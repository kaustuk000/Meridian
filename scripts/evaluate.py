"""
Zero-shot COCO image-text retrieval evaluation.
Two evaluators:
  CLIPCocoRetrievalEvaluator     — cosine-similarity baseline (raw HF CLIPModel,
                                    e.g. meridian_model.clip, for comparison)
  MeridianCocoRetrievalEvaluator — gate-weighted fusion of hyp + eucl spaces,
                                    same similarity formulation as
                                    gate_contrastive_loss during training

This version adds a `simulate_fp16_storage` option to MeridianCocoRetrievalEvaluator
that round-trips the computed embeddings (h_image, e_image, h_text, e_text, gates)
through fp16 right after encoding — simulating exactly what happens if the index
is stored on disk as fp16 — before running the same fp32 similarity/ranking math.
This isolates storage-precision loss from forward-pass numerics.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel

from meridian.lorentz import lorentz_distance
from meridian.data.evaluation import CocoCaptions, retrieval_collate_fn
from meridian.tokenizer import Tokenizer
from meridian.model import MeridianModel



# Shared helpers

tokenizer = Tokenizer()

def _build_ground_truth(samples):
    """samples: list of (image_id, image_path, caption_ids, captions)."""
    image_to_text_gt = defaultdict(set)
    text_to_image_gt = defaultdict(set)
    for image_id, _, caption_ids, _ in samples:
        image_to_text_gt[image_id].update(caption_ids)
        for cid in caption_ids:
            text_to_image_gt[cid].add(image_id)
    return image_to_text_gt, text_to_image_gt


def _flatten_captions(batch):
    """batch['captions']/batch['caption_ids'] are list[B] of list[~5]. Flatten in lockstep."""
    flat_texts, flat_ids = [], []
    for cids, caps in zip(batch["caption_ids"], batch["captions"]):
        flat_ids.extend(cids)
        flat_texts.extend(caps)
    return flat_texts, flat_ids


def _compute_recall(predictions: dict[int, list[int]], ground_truth: dict[int, set[int]], K: int):
    num_correct = 0.0
    for query_id, paired_ids in ground_truth.items():
        preds = predictions.get(query_id, [])
        if set(preds[:K]) & paired_ids:
            num_correct += 1.0
    return 100.0 * num_correct / len(ground_truth)


def _ranked_recall(scores, query_ids, candidate_ids, ground_truth, ks):
    """scores: [N_query, N_candidates], higher = more similar."""
    order = scores.argsort(dim=1, descending=True).cpu()
    candidate_ids_cpu = candidate_ids.cpu()
    predictions = {
        qid: candidate_ids_cpu[row].tolist()
        for qid, row in zip(query_ids.tolist(), order)
    }
    return {f"r{k}": _compute_recall(predictions, ground_truth, k) for k in ks}

def _tokenize_captions(captions, device):
    tokens = tokenizer(captions)

    max_len = max(len(t) for t in tokens)
    batch_size = len(tokens)

    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attn_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    eos_indices = torch.zeros(batch_size, dtype=torch.long)

    for i, t in enumerate(tokens):
        L = len(t)
        input_ids[i, :L] = t
        attn_mask[i, :L] = 1
        eos_indices[i] = L - 1

    return (
        input_ids.to(device),
        attn_mask.to(device),
        eos_indices.to(device),
    )


def _fp16_roundtrip(name: str, tensor: torch.Tensor, report: bool) -> torch.Tensor:
    """Casts to fp16 and back to fp32, simulating fp16 index storage. Optionally prints
    the quantization error introduced, so you can see which embedding space (hyperbolic
    vs euclidean) is more sensitive to fp16 storage."""
    fp16_tensor = tensor.half().float()
    if report:
        err = (tensor - fp16_tensor).abs()
        rel_err = (err / (tensor.abs() + 1e-8)).mean().item()
        norm = tensor.norm(dim=-1)
        print(f"  [{name:8s}] max_norm={norm.max().item():8.3f}  "
              f"mean_norm={norm.mean().item():8.3f}  "
              f"mean_abs_err={err.mean().item():.6f}  "
              f"mean_rel_err={rel_err:.6f}")
    return fp16_tensor


class CLIPCocoRetrievalEvaluator:
    """
    Zero-shot COCO retrieval for a frozen HF CLIPModel via cosine similarity.
    """

    def __init__(self, coco_root: str | Path, split: str = "test",
                 ks: tuple[int, ...] = (1, 5, 10), batch_size: int = 32,
                 num_workers: int = 4, device: str = "cuda"):
        self.dataset = CocoCaptions(coco_root, split=split)
        self.loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=retrieval_collate_fn,
        )
        self.ks = ks
        self.device = device
        self.image_to_text_gt, self.text_to_image_gt = _build_ground_truth(self.dataset.samples)

    @torch.inference_mode()
    def __call__(self, model: CLIPModel) -> dict[str, float]:
        model = model.eval().to(self.device)

        image_ids, text_ids = [], []
        image_feats, text_feats = [], []

        for batch in tqdm(self.loader, desc="CLIP: encoding COCO test set"):
            image_ids.extend(batch["image_id"])
            captions, caption_ids = _flatten_captions(batch)
            text_ids.extend(caption_ids)

            pixel_values = batch["image"].to(self.device)
            input_ids, attn_mask, _ = _tokenize_captions(captions, self.device)

            vision_outputs = model.vision_model(pixel_values=pixel_values, return_dict=True)
            img_feat = F.normalize(model.visual_projection(vision_outputs.pooler_output), dim=-1)

            text_outputs = model.text_model(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
            txt_feat = F.normalize(model.text_projection(text_outputs.pooler_output), dim=-1)

            image_feats.append(img_feat)
            text_feats.append(txt_feat)
        image_feats = torch.cat(image_feats, dim=0)
        text_feats = torch.cat(text_feats, dim=0)
        image_ids = torch.tensor(image_ids)
        text_ids = torch.tensor(text_ids)

        i2t_scores = image_feats @ text_feats.T
        t2i_scores = text_feats @ image_feats.T

        results = {}
        results.update({f"i2t_{k}": v for k, v in
                         _ranked_recall(i2t_scores, image_ids, text_ids, self.image_to_text_gt, self.ks).items()})
        results.update({f"t2i_{k}": v for k, v in
                         _ranked_recall(t2i_scores, text_ids, image_ids, self.text_to_image_gt, self.ks).items()})
        return results



# Meridian (gated dual-space)

class MeridianCocoRetrievalEvaluator:
    """
    Zero-shot COCO retrieval for Meridian, using the gate-weighted fusion of
    the hyperbolic and euclidean spaces — same formulation as
    gate_contrastive_loss, but over the full test set (not per-batch negatives).

    i2t (image is query) uses gate_image's (a_img, b_img) — per-image-row weight.
    t2i (text is query) uses gate_text's (a_txt, b_txt) — per-text-row weight.

    simulate_fp16_storage: if True, round-trips h_image/e_image/h_text/e_text
    and the gate weights through fp16 right after encoding — i.e. at the exact
    point they'd be written to an fp16 index file — before computing similarity
    scores. This isolates storage-precision loss from forward-pass numerics
    (the forward pass itself still runs in fp32, same as during training).
    """

    def __init__(self, coco_root: str | Path, split: str = "test",
                 ks: tuple[int, ...] = (1, 5, 10), batch_size: int = 36,
                 num_workers: int = 4, device: str = "cuda",
                 simulate_fp16_storage: bool = False,
                 report_fp16_error: bool = True):
        self.dataset = CocoCaptions(coco_root, split=split)
        self.loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=retrieval_collate_fn,
        )
        self.ks = ks
        self.device = device
        self.simulate_fp16_storage = simulate_fp16_storage
        self.report_fp16_error = report_fp16_error
        self.image_to_text_gt, self.text_to_image_gt = _build_ground_truth(self.dataset.samples)

    @torch.inference_mode()
    def __call__(self, model: MeridianModel) -> dict[str, float]:
        model = model.eval().to(self.device)

        image_ids, text_ids = [], []
        h_images, e_images, a_imgs, b_imgs = [], [], [], []
        h_texts, e_texts, a_txts, b_txts = [], [], [], []
        curv = scale_hyp = scale_eucl = None

        for batch in tqdm(self.loader, desc="Meridian: encoding COCO test set"):
            image_ids.extend(batch["image_id"])
            captions, caption_ids = _flatten_captions(batch)
            text_ids.extend(caption_ids)

            pixel_values = batch["image"].to(self.device)
            input_ids, attn_mask, eos_indices = _tokenize_captions(captions, self.device)
            img_out = model.encode_image(pixel_values)
            txt_out = model.encode_text(input_ids, attn_mask, eos_indices)

            h_images.append(img_out["h_image"].float())
            e_images.append(img_out["e_image"].float())
            a_imgs.append(img_out["a_img"].float())
            b_imgs.append(img_out["b_img"].float())

            h_texts.append(txt_out["h_text"].float())
            e_texts.append(txt_out["e_text"].float())
            a_txts.append(txt_out["a_txt"].float())
            b_txts.append(txt_out["b_txt"].float())

            # curv / scale_hyp / scale_eucl are global scalars
            if curv is None:
                curv = img_out["curv"]
                scale_hyp = img_out["scale_hyp"]
                scale_eucl = img_out["scale_eucl"]

        h_image = torch.cat(h_images, dim=0)
        e_image = torch.cat(e_images, dim=0)
        a_img = torch.cat(a_imgs, dim=0)
        b_img = torch.cat(b_imgs, dim=0)

        h_text = torch.cat(h_texts, dim=0)
        e_text = torch.cat(e_texts, dim=0)
        a_txt = torch.cat(a_txts, dim=0)
        b_txt = torch.cat(b_txts, dim=0)

        # --- fp16 index-storage simulation -----------------------------------
        # This is the point at which these tensors would be written to disk in
        # the index file, so this is exactly where the round-trip belongs.
        if self.simulate_fp16_storage:
            print("\nSimulating fp16 index storage (round-trip fp32 -> fp16 -> fp32):")
            h_image = _fp16_roundtrip("h_image", h_image, self.report_fp16_error)
            e_image = _fp16_roundtrip("e_image", e_image, self.report_fp16_error)
            h_text = _fp16_roundtrip("h_text", h_text, self.report_fp16_error)
            e_text = _fp16_roundtrip("e_text", e_text, self.report_fp16_error)
            a_img = _fp16_roundtrip("a_img", a_img, self.report_fp16_error)
            b_img = _fp16_roundtrip("b_img", b_img, self.report_fp16_error)
            a_txt = _fp16_roundtrip("a_txt", a_txt, self.report_fp16_error)
            b_txt = _fp16_roundtrip("b_txt", b_txt, self.report_fp16_error)
            print()
        # -----------------------------------------------------------------------

        image_ids = torch.tensor(image_ids)
        text_ids = torch.tensor(text_ids)

        with torch.autocast(self.device, dtype=torch.float32):
            hyp_sim_i2t = scale_hyp * (-lorentz_distance(h_image, h_text, curv))
            hyp_sim_t2i = scale_hyp * (-lorentz_distance(h_text, h_image, curv))
        eucl_sim_i2t = scale_eucl * (e_image @ e_text.T)
        eucl_sim_t2i = scale_eucl * (e_text @ e_image.T)

        i2t_scores = a_img.unsqueeze(1) * hyp_sim_i2t + b_img.unsqueeze(1) * eucl_sim_i2t
        t2i_scores = a_txt.unsqueeze(1) * hyp_sim_t2i + b_txt.unsqueeze(1) * eucl_sim_t2i

        #i2t_scores = eucl_sim_i2t
        #t2i_scores = eucl_sim_t2i

        results = {}
        results.update({f"i2t_{k}": v for k, v in
                         _ranked_recall(i2t_scores, image_ids, text_ids, self.image_to_text_gt, self.ks).items()})
        results.update({f"t2i_{k}": v for k, v in
                         _ranked_recall(t2i_scores, text_ids, image_ids, self.text_to_image_gt, self.ks).items()})
        return results


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--coco-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--eval-mode", choices=["meridian", "clip", "both"],
                        default="both")
    parser.add_argument("--clip-checkpoint", type=str, default=None)

    parser.add_argument("--image-hout", type=int, default=64)
    parser.add_argument("--image-eout", type=int, default=64)
    parser.add_argument("--text-hout", type=int, default=64)
    parser.add_argument("--text-eout", type=int, default=64)

    parser.add_argument("--fp16-storage", action="store_true",
                        help="Round-trip Meridian embeddings through fp16 after encoding, "
                             "to simulate fp16 index storage and measure its effect on recall.")
    parser.add_argument("--no-fp16-report", action="store_true",
                        help="Suppress the per-component quantization error printout "
                             "when --fp16-storage is set.")

    args = parser.parse_args()

    model = MeridianModel(
        image_hout=args.image_hout,
        image_eout=args.image_eout,
        text_hout=args.text_hout,
        text_eout=args.text_eout,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model = model.to(args.device)

    # CLIP baseline    
    if args.eval_mode in ["clip", "both"]:
        print("\n===== CLIP BASELINE =====")

        clip_evaluator = CLIPCocoRetrievalEvaluator(
            coco_root=args.coco_root,
            split=args.split,
            device=args.device,
        )

        if args.clip_checkpoint is not None:
            clip_model = CLIPModel.from_pretrained(args.clip_checkpoint)
            clip_model = clip_model.to(args.device)
        else:
            clip_model = model.clip

        clip_metrics = clip_evaluator(clip_model)

        for k, v in clip_metrics.items():
            print(f"{k}: {v:.3f}")


    # Meridian — fp32 baseline
    if args.eval_mode in ["meridian", "both"]:
        print("\n===== MERIDIAN (fp32) =====")

        meridian_evaluator = MeridianCocoRetrievalEvaluator(
            coco_root=args.coco_root,
            split=args.split,
            device=args.device,
            simulate_fp16_storage=False,
        )

        meridian_metrics = meridian_evaluator(model)

        for k, v in meridian_metrics.items():
            print(f"{k}: {v:.3f}")

        # Meridian — fp16 index-storage simulation, run only if requested
        if args.fp16_storage:
            print("\n===== MERIDIAN (fp16 index-storage round-trip) =====")

            meridian_evaluator_fp16 = MeridianCocoRetrievalEvaluator(
                coco_root=args.coco_root,
                split=args.split,
                device=args.device,
                simulate_fp16_storage=True,
                report_fp16_error=not args.no_fp16_report,
            )

            meridian_metrics_fp16 = meridian_evaluator_fp16(model)

            for k, v in meridian_metrics_fp16.items():
                delta = v - meridian_metrics[k]
                print(f"{k}: {v:.3f}   (Δ vs fp32: {delta:+.3f})")
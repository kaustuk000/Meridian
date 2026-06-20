import argparse
import functools
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from meridian.model import MeridianModel
from meridian.tokenizer import Tokenizer

from meridian.data.cc3m import ImageTextWebDataset
from meridian.data.transforms import build_eval_transform

# Constants matching CLIP/Meridian configurations
MAX_SEQ_LEN = 77
_EOT_TOKEN = 49407

class CustomIndexMapper:
    """
    Custom mapper designed specifically for index generation.
    It explicitly extracts the web URL and uses the official deterministic 
    evaluation transform from meridian.data.transforms.
    """
    def __init__(self):
        # Use the centrally defined evaluation transform
        self.image_transform = build_eval_transform()

    def __call__(self, sample: dict) -> dict:
        # 1. Safely extract text caption
        caption = (sample.get("txt") or "").strip()
        if not caption:
            caption = sample.get("json", {}).get("caption", "")

        # 2. Extract original image URL from the JSON metadata dictionary
        metadata = sample.get("json", {})
        url = metadata.get("url", "")

        return {
            "__key__": sample["__key__"],
            "image":   self.image_transform(sample["jpg"]),
            "text":    caption,
            "url":     url,
        }


def _collate_index(batch: list[dict], tokenizer: Tokenizer) -> dict:
    """
    Custom collate function that mirrors the tokenization mechanics of 
    _collate_cc3m, but explicitly preserves captions and URLs for the index file.
    """
    images   = torch.stack([s["image"] for s in batch])
    captions = [s["text"] for s in batch]
    urls     = [s["url"] for s in batch]
    keys     = [s["__key__"] for s in batch]

    token_list = tokenizer(captions)
    B = len(token_list)
    
    input_ids      = torch.zeros(B, MAX_SEQ_LEN, dtype=torch.long)
    attention_mask = torch.zeros(B, MAX_SEQ_LEN, dtype=torch.long)
    eos_indices    = torch.zeros(B, dtype=torch.long)

    for i, tokens in enumerate(token_list):
        seq = tokens[:MAX_SEQ_LEN].long()
        if len(tokens) > MAX_SEQ_LEN:
            seq = seq.clone()
            seq[-1] = _EOT_TOKEN

        n = len(seq)
        input_ids[i, :n]      = seq
        attention_mask[i, :n] = 1
        eos_indices[i]        = n - 1

    return {
        "pixel_values":   images,
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "eos_indices":    eos_indices,
        "captions":       captions,
        "urls":           urls,
        "__keys__":       keys
    }


def load_checkpoint_clean(checkpoint_path, device):
    """Loads weights and dynamically infers architecture dimensions from the checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    
    def _find_dim(prefix: str) -> int:
        for k, v in state_dict.items():
            if k.startswith(prefix) and k.endswith(".weight") and v.ndim == 2 and v.shape[1] == 128:
                return int(v.shape[0])
        raise RuntimeError(f"Could not infer dimension from checkpoint prefix '{prefix}'.")

    model = MeridianModel(
        image_hout=_find_dim("hyp_image_head.image_mlp."),
        image_eout=_find_dim("eucl_image_head.image_mlp."),
        text_hout=_find_dim("hyp_text_head.text_mlp."),
        text_eout=_find_dim("eucl_text_head.text_mlp.")
    ).to(device)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Build Meridian Index using native WebDataset streaming pipeline.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint.")
    parser.add_argument("--tarfiles-dir", default="meridian/data/cc3m_smoke", help="Directory with CC3M TAR shards.")
    parser.add_argument("--output-index", default="meridian/data/index/cc3m_index_32.pt", help="Where to save the index file.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint_clean(args.checkpoint, device)
    tokenizer = Tokenizer()

    # 1. Instantiate our custom index mapper
    mapper = CustomIndexMapper()

    # 2. Build the dataset using your native WebDataset wrapper class
    dataset = ImageTextWebDataset(
        tarfiles=f"{args.tarfiles_dir}/*.tar",
        mapper=mapper,
        buffer_size=0,            # Disable shuffle buffer for strict sequential extraction
        infinite_stream=False,    # End execution loop once shard files complete
        seed=42,
    )

    # 3. Create a custom DataLoader passing our specialized indexing collate function
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=functools.partial(_collate_index, tokenizer=tokenizer),
        pin_memory=True,
        drop_last=False,          # Keep last partial batch to ensure zero data loss during indexing
        persistent_workers=args.workers > 0,
    )

    index_data = {
        "h_image": [],"e_image": [],
        "h_text": [], "e_text": [],
        "a_image": [], "b_image": [],
        "a_text": [], "b_text": [],
        "captions": [],"urls": [],
    }

    print(f"Streaming evaluation shards from {args.tarfiles_dir}...")
    with torch.no_grad():
        for batch in tqdm(loader):
            pixel_values   = batch["pixel_values"].to(device)
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            eos_indices    = batch["eos_indices"].to(device)

            # Compute vectors using forward pass
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                eos_indices=eos_indices
            )

            # Gather embeddings to CPU storage
            index_data["h_image"].append(outputs["h_image"].cpu())
            index_data["e_image"].append(outputs["e_image"].cpu())
            index_data["h_text"].append(outputs["h_text"].cpu())
            index_data["e_text"].append(outputs["e_text"].cpu())
            index_data["a_image"].append(outputs["a_image"].cpu())
            index_data["b_image"].append(outputs["b_image"].cpu())
            index_data["a_text"].append(outputs["a_text"].cpu())
            index_data["b_text"].append(outputs["b_text"].cpu())
            
            # Safely capture the extracted text and URLs from our custom collate output
            index_data["captions"].extend(batch["captions"])
            index_data["urls"].extend(batch["urls"])

    print("Consolidating multi-batch representations...")
    final_index = {
        "h_image": torch.cat(index_data["h_image"], dim=0),
        "e_image": torch.cat(index_data["e_image"], dim=0),
        "h_text": torch.cat(index_data["h_text"], dim=0),
        "e_text": torch.cat(index_data["e_text"], dim=0),
        "a_image": torch.cat(index_data["a_image"], dim=0),
        "b_image": torch.cat(index_data["b_image"], dim=0),
        "a_text": torch.cat(index_data["a_text"], dim=0),
        "b_text": torch.cat(index_data["b_text"], dim=0),
        "captions": index_data["captions"],
        "urls": index_data["urls"]
    }

    print(f"Successfully generated index cache with {len(final_index['captions'])} elements.")
    
    # Save directory verification safety check
    Path(args.output_index).parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_index, args.output_index)
    print(f"Saved database registry to: {args.output_index}")

if __name__ == "__main__":
    main()
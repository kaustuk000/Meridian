import torch
from meridian.lorentz import lorentz_distance
from api.timing import timer, log_search


class MeridianSearchEngine:
    def __init__(self, hf_index_dict: dict, device: torch.device):
        self.device = device
        print(f"Loading Hugging Face index into {self.device}...")

        tensors = hf_index_dict["tensors"]
        metadata = hf_index_dict["metadata"]

        # Parse the JSON metadata
        self.index = {
            "captions": metadata.get("captions", []),
            "urls": metadata.get("urls", [])
        }

        # Move the safetensors to the correct device
        for key, val in tensors.items():
            self.index[key] = val.to(self.device)
        num_items = len(self.index["captions"])

        if "is_alive" not in self.index:
            self.index["is_alive"] = torch.ones(num_items, dtype=torch.bool, device=self.device)

        self.has_target_text = "h_text" in self.index and "e_text" in self.index
        self.has_target_image = "h_image" in self.index and "e_image" in self.index

        # Cache tensors once
        self.h_image = self.index["h_image"].float() if self.has_target_image else None
        self.e_image = self.index["e_image"].float() if self.has_target_image else None
        self.h_text = self.index["h_text"].float() if self.has_target_text else None
        self.e_text = self.index["e_text"].float() if self.has_target_text else None

        print(f"Engine Ready. Loaded {num_items} items.")

    def compute_scores(self, query_out: dict, query_has_text: bool, query_has_image: bool) -> torch.Tensor:
        curv = query_out.get("curv", torch.tensor(1.0, device=self.device)).float()
        scale_hyp = query_out.get("scale_hyp", torch.tensor(1.0, device=self.device)).float()
        scale_eucl = query_out.get("scale_eucl", torch.tensor(1.0, device=self.device)).float()

        def get_score_for_modality(q_hyp, q_eucl, q_a, q_b):
            hyp_parts = []
            eucl_parts = []

            if self.has_target_image:
                hyp_parts.append(scale_hyp * (-lorentz_distance(q_hyp, self.h_image, curv=curv)))
                eucl_parts.append(scale_eucl * (q_eucl @ self.e_image.T))

            if self.has_target_text:
                hyp_parts.append(scale_hyp * (-lorentz_distance(q_hyp, self.h_text, curv=curv)))
                eucl_parts.append(scale_eucl * (q_eucl @ self.e_text.T))

            combined_hyp = sum(hyp_parts) / len(hyp_parts)
            combined_eucl = sum(eucl_parts) / len(eucl_parts)

            return q_a.unsqueeze(1) * combined_hyp + q_b.unsqueeze(1) * combined_eucl

        if query_has_text and not query_has_image:
            return get_score_for_modality(
                query_out["h_text"],
                query_out["e_text"],
                query_out["a_txt"],
                query_out["b_txt"],
            )
        elif query_has_image and not query_has_text:
            return get_score_for_modality(
                query_out["h_image"],
                query_out["e_image"],
                query_out["a_img"],
                query_out["b_img"],
            )
        elif query_has_text and query_has_image:
            text_score = get_score_for_modality(
                query_out["h_text"],
                query_out["e_text"],
                query_out["a_txt"],
                query_out["b_txt"],
            )
            image_score = get_score_for_modality(
                query_out["h_image"],
                query_out["e_image"],
                query_out["a_img"],
                query_out["b_img"],
            )
            return 0.5 * (text_score + image_score)
        else:
            raise ValueError("At least one active query modality is required.")

    def search(self, query_payload: dict, query_has_text: bool, query_has_image: bool, topk: int = 9):
        with timer() as t_score:
            scores = self.compute_scores(query_payload, query_has_text, query_has_image).squeeze(0)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        with timer() as t_topk:
            k = min(topk, scores.shape[0])
            values, indices = torch.topk(scores, k=k, largest=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        log_search(
            score_ms=t_score["ms"],
            topk_ms=t_topk["ms"],
            n_candidates=k,
            index_size=scores.shape[0],
        )

        return [{"id": int(idx), "score": float(val)} for val, idx in zip(values.tolist(), indices.tolist())]
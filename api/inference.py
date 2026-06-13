import torch
import torch.nn.functional as F
from meridian.lorentz import lorentz_distance

def gated_score(hyp_logits, eucl_logits, q_a, q_b, cand_a, cand_b):
    """Computes the dynamically weighted cross-modal gating score matrix."""
    A = (q_a.unsqueeze(1) + cand_a.unsqueeze(0)) / 2.0
    B = (q_b.unsqueeze(1) + cand_b.unsqueeze(0)) / 2.0
    return (A * hyp_logits + B * eucl_logits).squeeze(0)

class MeridianSearchEngine:
    def __init__(self, index_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading index from {index_path} into {self.device}...")
        
        self.index = torch.load(index_path, map_location=self.device)
        num_items = len(self.index["captions"]) 
        
        if "is_alive" not in self.index:
            self.index["is_alive"] = torch.ones(num_items, dtype=torch.bool, device=self.device)
            
        self.has_target_text = "h_text" in self.index and "e_text" in self.index
        self.has_target_image = "h_image" in self.index and "e_image" in self.index
        
        print(f"Engine Ready. Loaded {num_items} items.")

    def compute_scores(self, query_out: dict, query_has_text: bool, query_has_image: bool) -> torch.Tensor:
        """Calculates similarities across query modalities and targets using the gated formulation."""
        curv = query_out.get("curv", torch.tensor(1.0, device=self.device)).float()
        scale = query_out.get("scale_eucl", torch.tensor(1.0, device=self.device)).float()

        q_a = query_out["a"].float().view(-1)
        q_b = query_out["b"].float().view(-1)
        cand_a = self.index["a"].float().view(-1)
        cand_b = self.index["b"].float().view(-1)

        def get_score_for_modality(q_hyp, q_eucl):
            hyp_list = []
            eucl_list = []

            if self.has_target_image:
                hyp_i = -lorentz_distance(q_hyp, self.index["h_image"].float(), curv=curv)
                eucl_i = scale * torch.matmul(F.normalize(q_eucl, p=2, dim=-1), self.index["e_image"].float().T)
                hyp_list.append(hyp_i)
                eucl_list.append(eucl_i)

            if self.has_target_text:
                hyp_t = -lorentz_distance(q_hyp, self.index["h_text"].float(), curv=curv)
                # FIXED: Changed index from "h_text" to "e_text"
                eucl_t = scale * torch.matmul(F.normalize(q_eucl, p=2, dim=-1), self.index["e_text"].float().T)
                hyp_list.append(hyp_t)
                eucl_list.append(eucl_t)

            combined_hyp = sum(hyp_list) / len(hyp_list)
            combined_eucl = sum(eucl_list) / len(eucl_list)
            _gate = gated_score(combined_hyp, combined_eucl, q_a, q_b, cand_a, cand_b)
            return combined_hyp + _gate + 0.5 * combined_eucl

        if query_has_text and not query_has_image:
            return get_score_for_modality(query_out["h_text"].float(), query_out["e_text"].float())
        elif query_has_image and not query_has_text:
            return get_score_for_modality(query_out["h_image"].float(), query_out["e_image"].float())
        elif query_has_text and query_has_image:
            text_score = get_score_for_modality(query_out["h_text"].float(), query_out["e_text"].float())
            image_score = get_score_for_modality(query_out["h_image"].float(), query_out["e_image"].float())
            return 0.5 * (text_score + image_score)
        else:
            raise ValueError("At least one active query modality is required.")

    def search(self, query_payload: dict, query_has_text: bool, query_has_image: bool, topk: int = 9):
        """
        Bridges the score calculation with the sorting logic required by the FastAPI endpoint.
        """
        # 1. Compute raw similarity matrix
        scores = self.compute_scores(query_payload, query_has_text, query_has_image)
        scores = scores.squeeze()

        # 2. Extract top ranked items
        k = min(topk, scores.shape[0])
        values, indices = torch.topk(scores, k=k, largest=True)

        # 3. Format array into the explicit dictionary shape app.py expects
        results = []
        for val, idx in zip(values.tolist(), indices.tolist()):
            results.append({
                "id": idx,
                "score": val
            })
        return results
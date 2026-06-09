import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from meridian.lorentz import lorentz_distance, exterior_angle, half_aperture_angle

class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process ranks and routes gradients back during backward pass.
    """
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        # Returns the gradient corresponding to the current GPU rank
        return all_gradients[dist.get_rank()]

def gather_features(features: torch.Tensor) -> torch.Tensor:
    """Safely gathers features if DDP is active, otherwise returns as-is."""
    if dist.is_available() and dist.is_initialized():
        gathered = GatherLayer.apply(features)
        return torch.cat(gathered, dim=0)
    return features

class MeridianLoss(nn.Module):
    def __init__(self, eucl_weight: float = 1.0, gate_weight: float = 1.0):
        """
        Multi-space joint loss module for the Meridian architecture.
        Combines Hyperbolic Contrastive, Hyperbolic Entailment, Euclidean Contrastive, and Gated Fusion objectives.
        """
        super().__init__()
        self.eucl_weight = eucl_weight
        self.gate_weight = gate_weight

    def forward(
        self, 
        h_image: torch.Tensor, 
        h_text: torch.Tensor, 
        e_image: torch.Tensor, 
        e_text: torch.Tensor, 
        a: torch.Tensor,         
        b: torch.Tensor,         
        curv: torch.Tensor, 
        scale_eucl: torch.Tensor,
        scale_hyp: torch.Tensor,
        entail_weight: torch.Tensor,
        alphas: list[torch.Tensor] = None,
        rank: int = 0
    ):
        """Calculates joint multi-space structural and alignment losses across batches."""
        # 1. ENFORCE FLOAT32 IMMEDIATELY AT THE ENTRY POINT
        # This completely guarantees numerical stability for hyperbolic math (cosh/sinh/distances)
        # regardless of whether the model's global context is running under FP16/BF16 AMP.
        h_image = h_image.float()
        h_text = h_text.float()
        e_image = e_image.float()
        e_text = e_text.float()
        
        # Enforce 1D structure via .view(-1) to preemptively neutralize linear projection shape bugs (e.g., (B, 1))
        a = a.float().view(-1)  
        b = b.float().view(-1)
        
        curv = curv.float()
        scale_eucl = scale_eucl.float()
        scale_hyp = scale_hyp.float()
        entail_weight = entail_weight.float() if isinstance(entail_weight, torch.Tensor) else entail_weight

        B_local = h_image.size(0)
        device = h_image.device

        # 2. Hyperbolic Entailment Loss (Strictly local 1-to-1 instance check)
        # Safe from NaNs now that inputs are explicitly converted to float32
        _angle = exterior_angle(h_text, h_image, curv)
        _aperture = half_aperture_angle(h_text, curv)
        entailment_loss = torch.clamp(_angle - _aperture, min=0.0).mean()

        # 3. Gather ALL features globally for cross-GPU contrastive pools
        all_h_image = gather_features(h_image)
        all_h_text = gather_features(h_text)
        all_e_image = gather_features(e_image)
        all_e_text = gather_features(e_text)
        all_a = gather_features(a).view(-1)
        all_b = gather_features(b).view(-1)

        # Targets represent the correct global indices for our local rows
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        targets = torch.arange(B_local, device=device) + (B_local * rank)

        # HYPERBOLIC BRANCH LOSSES
        # Row = local sample, Col = global gathered options -> Shape: (B_local, B_global)
        hyp_logits_per_image = - scale_hyp * lorentz_distance(h_image, all_h_text, curv)
        hyp_logits_per_text = - scale_hyp * lorentz_distance(h_text, all_h_image, curv)
        
        hyp_loss_img = F.cross_entropy(hyp_logits_per_image, targets)
        hyp_loss_txt = F.cross_entropy(hyp_logits_per_text, targets)
        hyp_contrastive_loss = (hyp_loss_img + hyp_loss_txt) / 2.0

        # EUCLIDEAN BRANCH LOSSES
        e_image_norm = F.normalize(e_image, p=2, dim=-1)
        e_text_norm = F.normalize(e_text, p=2, dim=-1)
        all_e_image_norm = F.normalize(all_e_image, p=2, dim=-1)
        all_e_text_norm = F.normalize(all_e_text, p=2, dim=-1)

        eucl_logits_per_image = scale_eucl * torch.matmul(e_image_norm, all_e_text_norm.t())
        eucl_logits_per_text = scale_eucl * torch.matmul(e_text_norm, all_e_image_norm.t())
        
        eucl_loss_img = F.cross_entropy(eucl_logits_per_image, targets)
        eucl_loss_txt = F.cross_entropy(eucl_logits_per_text, targets)
        eucl_contrastive_loss = (eucl_loss_img + eucl_loss_txt) / 2.0

        # GATED CONTRASTIVE LOSS
        # Match the local weights (rows) with the global weights (columns)
        # Broadcasting works seamlessly here because tensors are securely 1D before unsqueezing
        A_matrix = (a.unsqueeze(1) + all_a.unsqueeze(0)) / 2.0
        B_matrix = (b.unsqueeze(1) + all_b.unsqueeze(0)) / 2.0

        combined_logits_per_image = (A_matrix * hyp_logits_per_image) + (B_matrix * eucl_logits_per_image)
        combined_logits_per_text = (A_matrix * hyp_logits_per_text) + (B_matrix * eucl_logits_per_text)

        gate_loss_img = F.cross_entropy(combined_logits_per_image, targets)
        gate_loss_txt = F.cross_entropy(combined_logits_per_text, targets)
        gate_contrastive_loss = (gate_loss_img + gate_loss_txt) / 2.0

        # TOTAL COMPOSITE LOSS
        total_loss = (
            self.gate_weight * gate_contrastive_loss +
            hyp_contrastive_loss + 
            entailment_loss * entail_weight +
            self.eucl_weight * eucl_contrastive_loss
        )

        return total_loss, {
            "loss/total": total_loss.item(),
            "loss/gate_combined": gate_contrastive_loss.item(),
            "loss/hyp_contrastive": hyp_contrastive_loss.item(),
            "loss/entailment": entailment_loss.item(),
            "loss/euc_contrastive": eucl_contrastive_loss.item(),
            "hyperparams/entail_weight": entail_weight.item() if isinstance(entail_weight, torch.Tensor) else entail_weight,
            "gating/hyp_mean_weight": a.mean().item(),
            "gating/euc_mean_weight": b.mean().item()
        }
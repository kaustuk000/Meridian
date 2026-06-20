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
        return all_gradients[dist.get_rank()]


def gather_features(features: torch.Tensor) -> torch.Tensor:
    """Safely gathers features if DDP is active, otherwise returns as-is."""
    if dist.is_available() and dist.is_initialized():
        gathered = GatherLayer.apply(features)
        return torch.cat(gathered, dim=0)
    return features


class MeridianLoss(nn.Module):
    def __init__(self, hyp_weight: float = 0.3, eucl_weight: float = 0.3, gate_weight: float = 1.0, entail_weight: float = 0.15):
        """
        Multi-space joint loss for Meridian.

        gate_contrastive_loss is the PRIMARY objective (weight 1.0 by default) —
        it's the gated, per-query fused retrieval signal.

        hyp_contrastive_loss and eucl_contrastive_loss are small-weight
        "branch-alive" regularizers: they keep each individual space
        informative even on samples where the gate is currently routing
        away from it. They are not meant to independently solve retrieval.

        entailment_loss is scaled by the model's own (clamped, learnable)
        entail_weight, passed in per forward call.
        """
        super().__init__()
        self.hyp_weight = hyp_weight
        self.eucl_weight = eucl_weight
        self.entail_weight = entail_weight
        self.gate_weight = gate_weight

    def forward(
        self,
        h_image: torch.Tensor,
        h_text: torch.Tensor,
        e_image: torch.Tensor,
        e_text: torch.Tensor,
        a_img: torch.Tensor,
        b_img: torch.Tensor,
        a_txt: torch.Tensor,
        b_txt: torch.Tensor,
        curv: torch.Tensor,
        scale_hyp: torch.Tensor,
        scale_eucl: torch.Tensor,
        rank: int = 0
    ):
        """Calculates joint multi-space structural and alignment losses across batches."""

        # 1. Enforce float32 immediately — required for hyperbolic math stability.
        h_image = h_image.float()
        h_text = h_text.float()
        e_image = e_image.float()
        e_text = e_text.float()

        a_img = a_img.float().view(-1)
        b_img = b_img.float().view(-1)
        a_txt = a_txt.float().view(-1)
        b_txt = b_txt.float().view(-1)

        curv = curv.float()
        scale_hyp = scale_hyp.float()
        scale_eucl = scale_eucl.float()

        B_local = h_image.size(0)
        device = h_image.device

        # 2. HYPERBOLIC ENTAILMENT LOSS — strictly local 1-to-1 pairs, no gather needed.
        with torch.autocast(device.type, dtype=torch.float32):
            _angle = exterior_angle(h_text, h_image, curv)
            _aperture = half_aperture_angle(h_text, curv)
            entailment_loss = torch.clamp(_angle - _aperture, min=0.0).mean()

        # 3. Gather features globally for cross-GPU contrastive pools.
        all_h_image = gather_features(h_image)
        all_h_text = gather_features(h_text)
        all_e_image = gather_features(e_image)
        all_e_text = gather_features(e_text)

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        targets = torch.arange(B_local, device=device) + (B_local * rank)

        # HYPERBOLIC BRANCH
        with torch.autocast(device.type, dtype=torch.float32):
            hyp_logits_per_image = scale_hyp * (- lorentz_distance(h_image, all_h_text, curv))
            hyp_logits_per_text  = scale_hyp * (- lorentz_distance(h_text, all_h_image, curv))

        hyp_loss_img = F.cross_entropy(hyp_logits_per_image, targets)
        hyp_loss_txt = F.cross_entropy(hyp_logits_per_text, targets)
        hyp_contrastive_loss = (hyp_loss_img + hyp_loss_txt) / 2.0

        # EUCLIDEAN BRANCH
        eucl_logits_per_image = scale_eucl * torch.matmul(e_image, all_e_text.t())
        eucl_logits_per_text  = scale_eucl * torch.matmul(e_text, all_e_image.t())

        eucl_loss_img = F.cross_entropy(eucl_logits_per_image, targets)
        eucl_loss_txt = F.cross_entropy(eucl_logits_per_text, targets)
        eucl_contrastive_loss = (eucl_loss_img + eucl_loss_txt) / 2.0

        # GATED CONTRASTIVE LOSS
        combined_logits_per_image = (a_img.unsqueeze(1) * hyp_logits_per_image) + (b_img.unsqueeze(1) * eucl_logits_per_image)
        combined_logits_per_text  = (a_txt.unsqueeze(1) * hyp_logits_per_text)  + (b_txt.unsqueeze(1) * eucl_logits_per_text)

        gate_loss_img = F.cross_entropy(combined_logits_per_image, targets)
        gate_loss_txt = F.cross_entropy(combined_logits_per_text, targets)
        gate_contrastive_loss = (gate_loss_img + gate_loss_txt) / 2.0

        # TOTAL COMPOSITE LOSS
        total_loss = (
            self.gate_weight * gate_contrastive_loss
            + self.hyp_weight * hyp_contrastive_loss
            + self.eucl_weight * eucl_contrastive_loss
            + self.entail_weight * entailment_loss
        )

        return total_loss, {
            "loss/total": total_loss.item(),
            "loss/gate_combined": gate_contrastive_loss.item(),
            "loss/hyp_contrastive": hyp_contrastive_loss.item(),
            "loss/entailment": entailment_loss.item(),
            "loss/euc_contrastive": eucl_contrastive_loss.item(),
            "hyperparams/scale_hyp": scale_hyp.item(),
            "hyperparams/scale_eucl": scale_eucl.item(),
            "gating/img_hyp_weight": a_img.mean().item(),
            "gating/img_eucl_weight": b_img.mean().item(),
            "gating/txt_hyp_weight": a_txt.mean().item(),
            "gating/txt_eucl_weight": b_txt.mean().item(),
        }
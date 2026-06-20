"""
Meridian training script.
"""

import argparse
import time
import random
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from loguru import logger
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from meridian.model import MeridianModel
from meridian.losses import MeridianLoss
from meridian.tokenizer import Tokenizer
from meridian.data.cc3m import build_cc3m_dataloader
from meridian.optim import LinearWarmupCosineDecayLR, set_weight_decay_per_param

import gc

def gpu_memory_stats():
    if not torch.cuda.is_available():
        return {}

    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }

def gpu_cleanup():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class MeridianCheckpointManager:
    """
    Manages saving and loading training states safely across iterations.
    """

    def __init__(self, output_dir, model, optimizer, scheduler, scaler):
        self.output_dir = Path(output_dir)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler

    def save(self, iteration: int, is_final: bool = False):
        # VALIDATION: Check for NaN in model parameters before saving
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                logger.error(f"CHECKPOINT VALIDATION FAILED: Parameter '{name}' contains NaN values!")
                logger.error(f"Skipping checkpoint save to prevent corruption at iteration {iteration}")
                return False
        
        filename = "checkpoint_final.pt" if is_final else f"checkpoint_{iteration:07d}.pt"
        ckpt_path = self.output_dir / filename

        state_dict = {
            "iteration" : iteration,
            "model_state_dict" : self.model.state_dict(),
            "optimizer_state_dict" : self.optimizer.state_dict(),
            "scheduler_state_dict" : self.scheduler.state_dict(),
            "scaler_state_dict" : self.scaler.state_dict(),
        }
        torch.save(state_dict, ckpt_path)
        logger.info(f"Successfully saved checkpoint to: {ckpt_path}")

        # Keep a reference text file pointing to the latest checkpoint
        latest_path = self.output_dir / "latest_checkpoint.txt"
        with open(latest_path, "w") as f:
            f.write(str(filename))
        
        return True

    def resume(self, reset_scheduler: bool = False) -> int:
        latest_path = self.output_dir / "latest_checkpoint.txt"
        if not latest_path.exists():
            logger.warning(f"No checkpoint tracker found at {latest_path}. Starting from scratch.")
            return 0

        with open(latest_path, "r") as f:
            filename = f.read().strip()

        ckpt_path = self.output_dir / filename
        if not ckpt_path.exists():
            logger.error(f"Checkpoint file {ckpt_path} specified in tracker does not exist!")
            return 0

        logger.info(f"Resuming training state from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu") # Load all tensors onto the CPU
        
        # VALIDATION: Check for NaN in loaded checkpoint
        model_state = checkpoint["model_state_dict"]
        for param_name, param_tensor in model_state.items():
            if torch.isnan(param_tensor).any():
                logger.error(f"CHECKPOINT VALIDATION FAILED: Loaded checkpoint contains NaN in '{param_name}'")
                logger.error(f"Checkpoint {ckpt_path} is corrupted. Starting fresh from iteration 0.")
                return 0

        self.model.load_state_dict(checkpoint["model_state_dict"])
        #self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        try:
            self.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )
        except ValueError:
            logger.warning(
                "Skipping optimizer state because trainable parameters changed."
            )
        if reset_scheduler:
            logger.info("Skipping scheduler state restore (--reset-scheduler active).")
        else:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Scheduler last_epoch: {self.scheduler.last_epoch}")
        logger.info(f"Param group LRs: {[pg['lr'] for pg in self.optimizer.param_groups]}")
        return checkpoint["iteration"]

    def export_clean_weights(self, filename: str = "model_pure_weights.pt"):
        """
        Extracts only the network weights, completely stripping out training 
        metadata, optimizer states, and scaling histories to match clean 
        production-ready model distributions.
        """
        export_path = self.output_dir / filename
        
        # Pull the underlying model state dict directly
        clean_state = self.model.state_dict()
        
        # Strip any distributed training process prefixes if present
        clean_state = {k.replace("module.", ""): v for k, v in clean_state.items()}
        
        torch.save(clean_state, export_path)
        logger.info(f"Exported clean production weights to: {export_path}")

# CLI Argument Parser
parser = argparse.ArgumentParser(description = __doc__)

parser.add_argument("--output-dir", default = "./output_size64", help = "Directory to save checkpoints and logs.")
parser.add_argument("--resume", action = "store_true", help = "Automatically resume training from latest checkpoint.")
parser.add_argument("--checkpoint-period", type = int, default = 5000, help ="Save a checkpoint every N iterations.")
parser.add_argument("--log-period", type = int, default = 100, help = "Log metrics to stdout/TensorBoard every N steps.")
parser.add_argument("--seed", type = int, default = 42, help = "Random seed for reproducibility.")
parser.add_argument("--tarfiles-dir", default = "meridian/data/cc3m_smoke", help = "Directory containing CC3M TAR shards.")
# Model & Optimization Hyperparameters
parser.add_argument("--total-iterations", type = int, default = 100000, help = "Total training steps.")
parser.add_argument("--warmup-steps", type = int, default = 5000, help = "Linear LR warmup steps.")
parser.add_argument("--lr", type = float, default = 5e-4, help = "Peak learning rate.")
parser.add_argument("--weight-decay", type = float, default = 0.2, help = "AdamW weight decay configuration.")
parser.add_argument("--batch-size", type = int, default = 64, help = "Training batch size.")
parser.add_argument("--amp", action = "store_true", help = "Enable automatic mixed precision training.") 
parser.add_argument("--workers", type = int, default = 8, help = "Dataloader worker threads.")

# Architecture Specific Params
parser.add_argument("--image-hout", type = int, default = 64, help = "Hyperbolic out dimension for image.")
parser.add_argument("--image-eout", type = int, default = 64, help = "Euclidean out dimension for image.")
parser.add_argument("--text-hout", type = int, default = 64, help = "Hyperbolic out dimension for text.")
parser.add_argument("--text-eout", type = int, default = 64, help = "Euclidean out dimension for text.")

parser.add_argument("--eucl_weight", type = float, default = 0.3, help = "Euclidean contrastive loss weight.")
parser.add_argument("--hyp_weight", type = float, default = 0.7, help = "Hyperbolic contrastive loss weight.")
parser.add_argument("--gate_weight", type = float, default = 1.0, help = "Gated fusion loss weight.")

# ── Selective CLIP unfreeze ────────────────────────────────────────────────────
parser.add_argument(
    "--unfreeze-layers", type=int, default=0,
    help=(
        "Number of last CLIP transformer blocks to unfreeze in both the vision "
        "and text encoder (e.g. 1 = last block only, 2 = last two blocks). "
        "0 keeps CLIP fully frozen (default)."
    ),
)
parser.add_argument(
    "--unfreeze-at-step", type=int, default=None,
    help=(
        "Iteration at which to unfreeze the top CLIP blocks. "
        "If omitted and --unfreeze-layers > 0, layers are unfrozen immediately "
        "at the start of training (or on resume). "
        "Recommended: set to ~50-60%% of --total-iterations so the adapters "
        "have stabilised before touching CLIP weights."
    ),
)
parser.add_argument(
    "--unfreeze-lr-mult", type=float, default=0.1,
    help=(
        "LR multiplier applied to unfrozen CLIP parameters relative to --lr. "
        "Pretrained weights need a much smaller step size to avoid catastrophic "
        "forgetting. 0.1 (i.e. 10x smaller) is a safe default."
    ),
)

parser.add_argument(
    "--reset-scheduler", action="store_true",
    help=(
        "Discard saved scheduler state on resume and start a fresh cosine decay "
        "over the remaining steps. Use when extending --total-iterations beyond "
        "the original run."
    ),
)
parser.add_argument(
    "--reset-warmup-steps", type=int, default=500,
    help="Short re-warmup when --reset-scheduler is active (default: 500).",
)

parser.add_argument("--reset-lr", type = float, default = 1e-7, help = "Peak reset learning rate.")
# ──────────────────────────────────────────────────────────────────────────────


def main(_A: argparse.Namespace):
    # Environment and Logging Setup
    random.seed(_A.seed)
    np.random.seed(_A.seed)
    torch.manual_seed(_A.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_dir = Path(_A.output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)
    logger.add(output_dir / "training_log.txt", format = "{time} {level} {message}")

    logger.info("Command line configuration initialized:")
    for arg in vars(_A):
        logger.info(f"{arg:<20} : {getattr(_A, arg)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running execution pipeline natively on: {device}")

    # Architecture & Optimization Instantiation
    model = MeridianModel(
        image_hout = _A.image_hout, image_eout = _A.image_eout,
        text_hout = _A.text_hout, text_eout = _A.text_eout
    ).to(device)

    criterion = MeridianLoss(eucl_weight = _A.eucl_weight, hyp_weight = _A.hyp_weight, gate_weight = _A.gate_weight).to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # Protect geometric scalars and metadata layers from weight decay compression
    param_groups = set_weight_decay_per_param(
        model = model,
        weight_decay = _A.weight_decay,
        gain_bias_decay = 0.0,
        exclude_params = [
            "curv", "logit_scale_eucl", "logit_scale_hyp",
            "log_alpha_img", "log_alpha_txt", "entail_weight"
        ]
    )

    optimizer = torch.optim.AdamW(param_groups, lr = _A.lr, betas = (0.9, 0.98), eps = 1e-6)
    schedular = LinearWarmupCosineDecayLR(optimizer = optimizer, total_steps = _A.total_iterations, warmup_steps = _A.warmup_steps)
    scaler = GradScaler("cuda", enabled=_A.amp, init_scale = 2048.0)

    checkpoint_manager = MeridianCheckpointManager(
        output_dir = _A.output_dir, model = model, optimizer = optimizer, scheduler = schedular, scaler = scaler
        
    )
    start_iteration = (
    checkpoint_manager.resume(reset_scheduler=_A.reset_scheduler)
    if _A.resume else 0)

    if _A.resume and _A.reset_scheduler:
        remaining = _A.total_iterations - start_iteration
        if remaining <= 0:
            raise ValueError(
                f"--total-iterations ({_A.total_iterations}) must be greater than "
                f"the resumed checkpoint step ({start_iteration})."
            )

        for pg in optimizer.param_groups:
            pg["lr"] = _A.reset_lr
            pg["initial_lr"] = _A.reset_lr
            
        schedular = LinearWarmupCosineDecayLR(
            optimizer=optimizer,
            total_steps=remaining,
            warmup_steps=_A.reset_warmup_steps,
        )
        checkpoint_manager.scheduler = schedular          # keep reference in sync
        logger.info(
            f"[RESET-SCHED] Fresh cosine over {remaining} remaining steps "
            f"with {_A.reset_warmup_steps}-step re-warmup "
            f"(resume point: {start_iteration})."
        )
    tboard = SummaryWriter(log_dir = output_dir / "tensorboard")

    # ── Selective CLIP unfreeze helper ────────────────────────────────────────
    def do_unfreeze(at_step: int) -> None:
        """
        Unfreeze the top N CLIP blocks and register two new param groups with
        the existing optimizer:
          • wd group  — weight matrices (weight decay applies)
          • no-wd group — biases + LayerNorm gains/biases (no decay)

        The CLIP LR is intentionally low (--unfreeze-lr-mult * --lr) to avoid
        catastrophic forgetting of the pretrained representations.

        This function is idempotent: calling it when blocks are already
        unfrozen (e.g. after resume past the threshold) is safe because
        unfreeze_last_n_layers() only flips requires_grad on params that are
        still frozen.
        """
        new_params = model.unfreeze_last_n_layers(_A.unfreeze_layers)
        if not new_params:
            logger.info("do_unfreeze: no new parameters to add (already unfrozen?).")
            return

        # We need names to split wd / no-wd, so build a set of tensor ids first.
        new_param_ids = {id(p) for p in new_params}

        wd_params: list[nn.Parameter] = []
        no_wd_params: list[nn.Parameter] = []

        for name, param in model.clip.named_parameters():
            if id(param) not in new_param_ids:
                continue
            # Biases and 1-D tensors (LayerNorm weight/bias) get no decay.
            if param.ndim <= 1 or name.endswith(".bias"):
                no_wd_params.append(param)
            else:
                wd_params.append(param)

        clip_lr = _A.lr * _A.unfreeze_lr_mult

        if wd_params:
            optimizer.add_param_group({
                "params": wd_params,
                "lr": clip_lr,
                "weight_decay": _A.weight_decay,
            })
            schedular.base_lrs.append(clip_lr)
            schedular.lr_lambdas.append(schedular.lr_lambdas[-1])  

        if no_wd_params:
            optimizer.add_param_group({
                "params": no_wd_params,
                "lr": clip_lr,
                "weight_decay": 0.0,
            })
            schedular.base_lrs.append(clip_lr)
            schedular.lr_lambdas.append(schedular.lr_lambdas[-1])  
        

        logger.info(
            f"[UNFREEZE] Last {_A.unfreeze_layers} CLIP block(s) unfrozen. "
            f"Added {len(wd_params)} wd-params + {len(no_wd_params)} no-wd-params "
            f"at lr={clip_lr:.2e} (={_A.unfreeze_lr_mult}x peak lr)."
        )
        tboard.add_text(
            "Events/Unfreeze",
            f"Unfroze last {_A.unfreeze_layers} CLIP blocks at step {at_step}",
            at_step,
        )
    # ──────────────────────────────────────────────────────────────────────────

    # If we're resuming past the scheduled unfreeze step (or unfreeze was
    # requested without a specific step), activate it before entering the loop.
    if _A.unfreeze_layers > 0:
        unfreeze_step = _A.unfreeze_at_step  # may be None
        should_unfreeze_now = (
            unfreeze_step is None               # "unfreeze immediately" mode
            or start_iteration >= unfreeze_step # resuming past the threshold
        )
        if should_unfreeze_now:
            logger.info(
                f"[UNFREEZE] Activating immediately "
                f"(start_iteration={start_iteration}, "
                f"unfreeze_at_step={unfreeze_step})."
            )
            do_unfreeze(at_step=start_iteration)

    # Tokenizer
    tokenizer = Tokenizer()

    loader = build_cc3m_dataloader(
        tarfiles = f"{_A.tarfiles_dir}/*.tar",
        tokenizer = tokenizer,
        batch_size = _A.batch_size,
        num_workers = _A.workers,
        buffer_size = 5000,   
        infinite_stream = True,
        seed=42,
    )

    # Training Loop
    model.train()
    loader = iter(loader)
    logger.info(f"Beginning training pipeline from iteration {start_iteration + 1}")

    for iteration in range(start_iteration + 1, _A.total_iterations + 1):
        start_time = time.perf_counter()

        # ── Scheduled CLIP unfreeze ───────────────────────────────────────────
        if (
            _A.unfreeze_layers > 0
            and _A.unfreeze_at_step is not None
            and iteration == _A.unfreeze_at_step
            and not model._unfrozen_clip_modules   # guard: run exactly once
        ):
            logger.info(f"[UNFREEZE] Reached scheduled step {iteration}.")
            do_unfreeze(at_step=iteration)
            # Ensure newly unfrozen modules are in the right train/eval state.
            model.train()
        # ─────────────────────────────────────────────────────────────────────

        # Load batch
        batch = next(loader)

        pixel_values   = batch["pixel_values"].to(device)
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        eos_indices    = batch["eos_indices"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=_A.amp):
            outputs = model(
                pixel_values = pixel_values,
                input_ids = input_ids,
                attention_mask = attention_mask,
                eos_indices = eos_indices
            )

            loss, metrics = criterion(
                h_image = outputs["h_image"], h_text = outputs["h_text"],
                e_image = outputs["e_image"], e_text = outputs["e_text"],
                a_img = outputs["a_image"], b_img = outputs["b_image"],
                a_txt = outputs["a_text"], b_txt = outputs["b_text"],
                curv = outputs["curv"], scale_eucl = outputs["scale_eucl"], scale_hyp = outputs["scale_hyp"],
            )
            
        # EARLY DETECTION: Halt training immediately if NaN is detected
        if torch.isnan(loss):
            logger.error(f"NaN loss detected at iteration {iteration}!")
            logger.error(f"Outputs - Curv: {outputs['curv']},Scale_eucl: {outputs['scale_eucl']}")
            logger.error(f"Metrics: {metrics}")
            
            # DIAGNOSTIC: Log all learnable parameters to find the culprit
            logger.error("\n=== PARAMETER DIAGNOSTICS ===")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    has_nan = torch.isnan(param).any().item()
                    has_inf = torch.isinf(param).any().item()
                    val_mean = param.mean().item() if not (has_nan or has_inf) else "N/A"
                    val_max = param.max().item() if not (has_nan or has_inf) else "N/A"
                    logger.error(f"  {name}: NaN={has_nan}, Inf={has_inf}, Mean={val_mean}, Max={val_max}")
            
            # SAVE BATCH DATA FOR INSPECTION
            nan_batch_data = {
                "iteration": iteration,
                "pixel_values": pixel_values.cpu().detach(),
                "input_ids": input_ids.cpu().detach(),
                "attention_mask": attention_mask.cpu().detach(),
                "eos_indices": eos_indices.cpu().detach(),
                "outputs": {k: v.cpu().detach() if isinstance(v, torch.Tensor) else v for k, v in outputs.items()},
                "metrics": metrics,
            }
            
            nan_batch_path = Path(_A.output_dir) / f"nan_batch_iter{iteration}.pkl"
            with open(nan_batch_path, "wb") as f:
                pickle.dump(nan_batch_data, f)
            logger.error(f"Saved problematic batch data to: {nan_batch_path}")
            
            checkpoint_manager.save(iteration)
            raise RuntimeError(f"Training halted due to NaN loss at iteration {iteration}")

        scaler.scale(loss).backward()
        
        # GLOBAL GRADIENT CLIPPING: Prevent exploding gradients across all parameters
        # This is crucial for stability of learnable parameters like curvature
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        nan_in_grads = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters() if p.requires_grad
        )
        if nan_in_grads:
            logger.warning(
                f"Non-finite gradients at iter {iteration} — skipping update. "
                f"Curv={outputs['curv'].item():.4f}, "
                f"scale_eucl={outputs['scale_eucl'].item():.2f}"
            )
            optimizer.zero_grad(set_to_none=True)
            scaler.update(new_scale=scaler.get_scale() / 2)
            continue

        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()

        if new_scale >= old_scale:
            schedular.step()

        step_time = time.perf_counter() - start_time

        if iteration % _A.log_period == 0:
            with torch.no_grad():
                # Compute Routing Shannon Entropy to evaluate router confidence mapping
                # Probs array shape: [Batch, 2]                
                gating_probs_img = torch.stack([outputs["a_image"], outputs["b_image"]], dim = -1)
                gating_probs_txt = torch.stack([outputs["a_text"], outputs["b_text"]], dim = -1)
                eps = 1e-8

                routing_entropy_img = -torch.sum(gating_probs_img * torch.log(gating_probs_img + eps), dim = -1).mean().item()
                routing_entropy_txt = -torch.sum(gating_probs_txt * torch.log(gating_probs_txt + eps), dim = -1).mean().item()
                curv_val = outputs["curv"].item()
                temp_eucl = outputs["scale_eucl"].item() 
                temp_hyp = outputs["scale_hyp"].item()

                current_lr = schedular.get_last_lr()[0]

            # Output comprehensive console diagnostics tracking space balances
            if torch.cuda.is_available():
                mem = gpu_memory_stats()
            else:
                mem = {
                    "allocated_gb": 0.0,
                    "reserved_gb": 0.0,
                    "max_allocated_gb": 0.0,
                }

            # Derive CLIP fine-tune LR for logging (second param group after unfreeze)
            clip_lr_str = "frozen"
            if model._unfrozen_clip_modules and len(optimizer.param_groups) > 1:
                clip_lr_str = f"{optimizer.param_groups[-2]['lr']:.2e}"

            log_str = (
                f"Iter {iteration}/{_A.total_iterations} | "
                f"Loss: {loss.item():.4f} | "
                f"Time: {step_time:.3f}s | "
                f"LR: {current_lr:.2e} | "
                f"CLIP_LR: {clip_lr_str} | "
                f"Entropy_Img: {routing_entropy_img:.3f} | "
                f"Entropy_Txt: {routing_entropy_txt:.3f} | "
                f"Curv: {curv_val:.4f} | "
                f"GPU Alloc: {mem['allocated_gb']:.2f}GB | "
                f"GPU Reserved: {mem['reserved_gb']:.2f}GB | "
                f"GPU Peak: {mem['max_allocated_gb']:.2f}GB | "
                f"Gate Weights (Mean) "
                f"[A_Img (Hyp): {outputs['a_image'].mean().item():.3f} | "
                f"B_Img (Euc): {outputs['b_image'].mean().item():.3f} | "
                f"A_Txt (Hyp): {outputs['a_text'].mean().item():.3f} | "
                f"B_Txt (Euc): {outputs['b_text'].mean().item():.3f}] | "
            )

            logger.info(log_str)

            # Log granular metrics into TensorBoard tracking dashboards
            tboard.add_scalar("Train/Total_Loss", loss.item(), iteration)
            tboard.add_scalar("Train/Learning_Rate", current_lr, iteration)
            tboard.add_scalar("Router/Entropy_Image", routing_entropy_img, iteration)
            tboard.add_scalar("Router/Entropy_Text", routing_entropy_txt, iteration)
            tboard.add_scalar("Router/Mean_A_img_Hyp", outputs["a_image"].mean().item(), iteration)
            tboard.add_scalar("Router/Mean_B_img_Euc", outputs["b_image"].mean().item(), iteration)
            tboard.add_scalar("Router/Mean_A_txt_Hyp", outputs["a_text"].mean().item(), iteration)
            tboard.add_scalar("Router/Mean_B_txt_Euc", outputs["b_text"].mean().item(), iteration)
            tboard.add_scalar("Geometry/Curvature", curv_val, iteration)
            tboard.add_scalar("Geometry/Temperature_Euclidean", temp_eucl, iteration)
            tboard.add_scalar("Geometry/Temperature_Hyperbolic", temp_hyp, iteration)

            tboard.add_scalar("GPU/Allocated_GB", mem["allocated_gb"], iteration)
            tboard.add_scalar("GPU/Reserved_GB", mem["reserved_gb"], iteration)
            tboard.add_scalar("GPU/Peak_Allocated_GB", mem["max_allocated_gb"], iteration)

            for key, value in metrics.items():
                tboard.add_scalar(key, value, iteration)

        # PERIODIC STORAGE CHECKPOINTING 
        if iteration % _A.checkpoint_period == 0:
            checkpoint_manager.save(iteration)

   
    checkpoint_manager.save(_A.total_iterations, is_final=True)

    checkpoint_manager.export_clean_weights("meridian_final_weights.pt")
    tboard.close()
    del model
    del criterion
    del optimizer
    del schedular
    del scaler
    del loader
    del tokenizer
    del checkpoint_manager

    gpu_cleanup()
    
    logger.info("Training pipeline completely finished without optimization failures.")


if __name__ == "__main__":
    _Args = parser.parse_args()
    main(_Args)
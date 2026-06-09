"""
Meridian training script.
"""

import argparse
import time
import random
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

    def resume(self) -> int:
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

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

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

parser.add_argument("--output-dir", default = "./output", help = "Directory to save checkpoints and logs.")
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
parser.add_argument("--amp", action = "store_true", default = True, help = "Enable automatic mixed precision training.")
parser.add_argument("--workers", type = int, default = 6, help = "Dataloader worker threads.")

# Architecture Specific Params
parser.add_argument("--image-hout", type = int, default = 16, help = "Hyperbolic out dimension for image.")
parser.add_argument("--image-eout", type = int, default = 32, help = "Euclidean out dimension for image.")
parser.add_argument("--text-hout", type = int, default = 16, help = "Hyperbolic out dimension for text.")
parser.add_argument("--text-eout", type = int, default = 32, help = "Euclidean out dimension for text.")

def main(_A: argparse.Namespace):
    # Environment and Logging Setup
    random.seed(_A.seed)
    np.random.seed(_A.seed)
    torch.manual_seed(_A.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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

    criterion = MeridianLoss(eucl_weight = 1.0, gate_weight = 1.0).to(device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # Protect geometric scalars and metadata layers from weight decay compression
    param_groups = set_weight_decay_per_param(
        model = model,
        weight_decay = _A.weight_decay,
        gain_bias_decay = 0.0,
        exclude_params = [
            "curv", "logit_scale_hyp", "logit_scale_eucl",
            "log_alpha_img", "log_alpha_txt", "entail_weight"
        ]
    )

    optimizer = torch.optim.AdamW(param_groups, lr = _A.lr, betas = (0.9, 0.98), eps = 1e-6)
    schedular = LinearWarmupCosineDecayLR(optimizer = optimizer, total_steps = _A.total_iterations, warmup_steps = _A.warmup_steps)
    scaler = GradScaler("cuda", enabled=_A.amp)

    checkpoint_manager = MeridianCheckpointManager(
        output_dir = _A.output_dir, model = model, optimizer = optimizer, scheduler = schedular, scaler = scaler
        
    )
    start_iteration = checkpoint_manager.resume() if _A.resume else 0
    tboard = SummaryWriter(log_dir = output_dir / "tensorboard")


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
                a = outputs["a"], b = outputs["b"],
                curv = outputs["curv"], scale_eucl = outputs["scale_eucl"],
                scale_hyp = outputs["scale_hyp"], entail_weight = outputs["entail_weight"],
                alphas = outputs["alphas"],
            )

            scaler.scale(loss).backward()
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
                gating_probs = torch.stack([outputs["a"], outputs["b"]], dim = -1)
                eps = 1e-8
                routing_entropy = -torch.sum(gating_probs * torch.log(gating_probs + eps), dim = -1).mean().item()

                curv_val = outputs["curv"].item()
                temp_hyp = outputs["scale_hyp"].item()
                temp_euc = outputs["scale_eucl"].item() 

                alpha_img_h, alpha_txt_h, alpha_img_e, alpha_txt_e = [a.item() for a in outputs["alphas"]] 
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

            log_str = (
                f"Iter {iteration}/{_A.total_iterations} | "
                f"Loss: {loss.item():.4f} | "
                f"Time: {step_time:.3f}s | "
                f"LR: {current_lr:.2e} | "
                f"Entropy: {routing_entropy:.3f} | "
                f"Curv: {curv_val:.4f} | "
                f"GPU Alloc: {mem['allocated_gb']:.2f}GB | "
                f"GPU Reserved: {mem['reserved_gb']:.2f}GB | "
                f"GPU Peak: {mem['max_allocated_gb']:.2f}GB | "
                f"Gate Weights (Mean) "
                f"[A (Hyp): {outputs['a'].mean().item():.3f} | "
                f"B (Euc): {outputs['b'].mean().item():.3f}]"
            )

            logger.info(log_str)

            # Log granular metrics into TensorBoard tracking dashboards
            tboard.add_scalar("Train/Total_Loss", loss.item(), iteration)
            tboard.add_scalar("Train/Learning_Rate", current_lr, iteration)
            tboard.add_scalar("Router/Entropy", routing_entropy, iteration)
            tboard.add_scalar("Router/Mean_A_Hyp", outputs["a"].mean().item(), iteration)
            tboard.add_scalar("Router/Mean_B_Euc", outputs["b"].mean().item(), iteration)
            tboard.add_scalar("Geometry/Curvature", curv_val, iteration)
            tboard.add_scalar("Geometry/Temperature_Hyperbolic", temp_hyp, iteration)
            tboard.add_scalar("Geometry/Temperature_Euclidean", temp_euc, iteration)
            tboard.add_scalar("Alphas/Hyperbolic_Image", alpha_img_h, iteration)
            tboard.add_scalar("Alphas/Hyperbolic_Text", alpha_txt_h, iteration)
            tboard.add_scalar("Alphas/Euclidean_Image", alpha_img_e, iteration)
            tboard.add_scalar("Alphas/Euclidean_Text", alpha_txt_e, iteration)

            tboard.add_scalar(
                "GPU/Allocated_GB",
                mem["allocated_gb"],
                iteration
            )

            tboard.add_scalar(
                "GPU/Reserved_GB",
                mem["reserved_gb"],
                iteration
            )

            tboard.add_scalar(
                "GPU/Peak_Allocated_GB",
                mem["max_allocated_gb"],
                iteration
            )
            for key, value in metrics.items():
                tboard.add_scalar(key, value, iteration)
        # PERIODIC STORAGE CHECKPOINTING 
        if iteration % _A.checkpoint_period == 0:
            checkpoint_manager.save(iteration)

    # Save final operational weights at end of training
    checkpoint_manager.save(_A.total_iterations, is_final=True)

    # Export a clean snapshot matching production-only distribution files
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

    
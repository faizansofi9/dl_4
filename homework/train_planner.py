"""
 Train a waypoint‐prediction planner.

 Usage (typical):
     python -m homework.train_planner \
         --model_name mlp_planner \
         --num_epoch 60 \
         --hidden_dim 128 \
         --batch_size 128 \
         --lr 3e-4
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from .models import load_model, save_model
from .datasets.road_dataset import load_data as load_drive_data
from .metrics import PlannerMetric

# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------

def masked_smooth_l1(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Smooth‑L1 (Huber) averaged **only** over valid waypoints."""
    loss = nn.functional.smooth_l1_loss(pred, tgt, reduction="none")  # (B, n_wp, 2)
    loss = loss * mask.unsqueeze(-1).float()
    return loss.sum() / mask.sum().clamp_min(1)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    print("CUDA/MPS not available → using CPU")
    return torch.device("cpu")


# ----------------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------------

def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 3e-4,
    batch_size: int = 128,
    seed: int = 2024,
    **model_kwargs,
) -> None:
    """Main training entry‑point."""

    device = get_device()

    # Reproducibility ----------------------------------------------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Logging ------------------------------------------------------------------------
    ts = datetime.now().strftime("%m%d_%H%M%S")
    log_dir = Path(exp_dir) / f"{model_name}_{ts}"
    logger = tb.SummaryWriter(log_dir)

    # Model --------------------------------------------------------------------------
    model = load_model(model_name, **model_kwargs).to(device)
    print(model)

    # Data ---------------------------------------------------------------------------
    train_loader = load_drive_data(
        "drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2
    )
    val_loader = load_drive_data(
        "drive_data/val", shuffle=False, batch_size=batch_size, num_workers=2
    )

    # Optimiser & scheduler ----------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)

    train_metric, val_metric = PlannerMetric(), PlannerMetric()
    global_step = 0

    # --------------------------------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------------------------------

    for epoch in range(num_epoch):
        # ---- Training phase --------------------------------------------------------
        model.train()
        train_metric.reset()

        for batch in train_loader:
            track_left = batch["track_left"].to(device)
            track_right = batch["track_right"].to(device)
            waypoints = batch["waypoints"].to(device)
            mask = batch["waypoints_mask"].to(device)

            preds = model(track_left, track_right)

            loss = masked_smooth_l1(preds, waypoints, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_metric.add(preds, waypoints, mask)
            logger.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        scheduler.step()

        # ---- Validation phase ------------------------------------------------------
        model.eval()
        val_metric.reset()
        with torch.inference_mode():
            for batch in val_loader:
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                waypoints = batch["waypoints"].to(device)
                mask = batch["waypoints_mask"].to(device)

                preds = model(track_left, track_right)
                val_metric.add(preds, waypoints, mask)

        # ---- Logging ---------------------------------------------------------------
        for k, v in train_metric.compute().items():
            logger.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metric.compute().items():
            logger.add_scalar(f"val/{k}", v, epoch)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            tr = train_metric.compute()
            vl = val_metric.compute()
            print(
                f"Epoch {epoch+1:02d}/{num_epoch:02d} | "
                + " ".join(f"train_{k}={tr[k]:.3f}" for k in tr)
                + " | "
                + " ".join(f"val_{k}={vl[k]:.3f}" for k in vl)
            )

    # --------------------------------------------------------------------------------
    # Save ---------------------------------------------------------------------------
    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.pth")
    print(f"Model saved to {log_dir / f'{model_name}.pth'}")


 # ------------------------------------------------------------------------------------
 # CLI
 # ------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--hidden_dim", type=int, default=64)

    # Any additional model hyper‑parameters should be added here and will be captured
    # by **model_kwargs in the train() function.

    train(**vars(parser.parse_args()))

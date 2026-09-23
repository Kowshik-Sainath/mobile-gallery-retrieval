"""
FS-COCO Training Orchestrator Script.

FIX C applied:
  - Cosine annealing LR scheduler with linear warmup (1 epoch).
  - Validation loop runs at end of every epoch (InfoNCE on test split).
  - Best val_loss tracking — saves checkpoint_best.pt when val improves.
  - Early stopping after 3 epochs without validation improvement.
  - val_loss logged to checkpoint metadata.

Issue 4 FIX: torch.amp.GradScaler('cuda') replaces deprecated torch.cuda.amp.GradScaler.
Issue 5 FIX: TF/Protobuf warning flood suppressed before any imports.
"""

# Issue 5 FIX: suppress TF/Protobuf/timm warning flood BEFORE any imports
import os
import warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
import argparse
import time
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from src.data.fscoco_dataset import get_fscoco_dataloaders
from src.models.composite_model import TSBIRCompositeModel
from src.utils.checkpoint import CheckpointManager


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, train_loader, optimizer, scaler, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    total_infonce = 0.0
    total_rec = 0.0
    total_attn = 0.0

    pbar = tqdm(train_loader, desc=f"[Train] Epoch {epoch}")
    for step, batch in enumerate(pbar):
        sketch = batch['sketch'].to(device)
        captions = batch['caption']
        photo = batch['photo'].to(device)
        target_sketch = batch['target_sketch'].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast('cuda'):
            outputs = model(sketch, captions, photo, target_sketch)
            loss = outputs['loss']

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss    += loss.item()
        total_infonce += outputs['loss_infonce'].item()
        total_rec     += outputs['loss_rec'].item()

        pbar.set_postfix({
            'Loss':    f"{loss.item():.4f}",
            'InfoNCE': f"{outputs['loss_infonce'].item():.4f}",
            'LR':      f"{scheduler.get_last_lr()[0]:.2e}",
        })

    n = len(train_loader)
    avg_loss = total_loss / n
    print(
        f"[Train] Epoch {epoch} | "
        f"Avg Loss: {avg_loss:.4f} | "
        f"InfoNCE: {total_infonce/n:.4f} | "
        f"Rec: {total_rec/n:.4f}"
    )
    return avg_loss



# ---------------------------------------------------------------------------
# FIX C — Validation loop
# ---------------------------------------------------------------------------

def evaluate_diagnostic_subset(model, val_loader, device, max_samples=100):
    """
    Computes quick R@1 diagnostic on a small validation subset during training.
    Immediately catches representation collapse / drift without waiting 25 epochs.
    """
    model.eval()
    q_embs, p_embs = [], []
    collected = 0
    with torch.no_grad():
        for batch in val_loader:
            photos   = batch['photo'].to(device)
            sketches = batch['sketch'].to(device)
            texts    = batch['caption']
            token_ids = model.tokenizer(texts).to(device)
            t_emb = model.encode_text(token_ids)
            s_emb = model.encode_sketch(sketches)
            raw_comp = torch.cat([s_emb, t_emb], dim=-1)
            q_emb = F.normalize(model.composite_fusion(raw_comp), dim=-1)
            p_emb = model.encode_photo(photos)

            q_embs.append(q_emb.cpu())
            p_embs.append(p_emb.cpu())
            collected += len(photos)
            if collected >= max_samples:
                break

    Q = torch.cat(q_embs, dim=0)[:collected]
    P = torch.cat(p_embs, dim=0)[:collected]
    sim = torch.mm(Q, P.t())
    ranks = torch.argsort(sim, dim=-1, descending=True)
    targets = torch.arange(collected)
    r1 = (ranks[:, 0] == targets).float().mean().item() * 100.0
    return r1, collected


def validate_one_epoch(model, val_loader, device, epoch):
    """
    Computes validation InfoNCE loss on the test split.
    No gradient computation. Uses same forward pass as training.
    Also logs quick Diagnostic R@1 on 100 validation samples.
    """
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"[Val]   Epoch {epoch}")
        for batch in pbar:
            sketch = batch['sketch'].to(device)
            captions = batch['caption']
            photo = batch['photo'].to(device)
            target_sketch = batch['target_sketch'].to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(sketch, captions, photo, target_sketch)

            total_val_loss += outputs['loss_infonce'].item()
            pbar.set_postfix({'Val InfoNCE': f"{outputs['loss_infonce'].item():.4f}"})

    avg_val = total_val_loss / len(val_loader)
    try:
        r1_diag, n_diag = evaluate_diagnostic_subset(model, val_loader, device, max_samples=100)
        chance = 100.0 / n_diag
        print(f"[Val]   Epoch {epoch} | Avg Val InfoNCE: {avg_val:.4f} | Diagnostic R@1 ({n_diag} val samples): {r1_diag:.2f}% (Chance: {chance:.2f}%)")
    except Exception as e:
        print(f"[Val]   Epoch {epoch} | Avg Val InfoNCE: {avg_val:.4f} (Diagnostic R@1 skipped: {e})")

    return avg_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train T+SBIR Model on FS-COCO")
    parser.add_argument("--data_dir", type=str, default="fscoco")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=1,
                        help="Number of epochs for linear LR warmup.")
    parser.add_argument("--early_stop_patience", type=int, default=3,
                        help="Stop training if val loss does not improve for N epochs.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--fresh_start", action="store_true",
        help=(
            "Skip all checkpoint loading and start training from epoch 1. "
            "Use this when the LoRA config has changed (e.g. different target_modules) "
            "and resuming from old checkpoints would corrupt the model state. "
            "Old checkpoints are archived to checkpoints/archived_old_lora/ for safety."
        )
    )
    parser.add_argument(
        "--max_train_samples", type=int, default=None,
        help="Optional limit on number of train samples (useful for fast verification gates)."
    )
    parser.add_argument(
        "--loss_attn_weight", type=float, default=0.0,
        help="Weight for the positive-pair attention alignment loss (Step 4, default 0.0)."
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Device: {device}")

    # --- Data ---
    print(f"[Data] Loading FS-COCO from: {args.data_dir}")
    try:
        train_loader, val_loader = get_fscoco_dataloaders(
            args.data_dir, batch_size=args.batch_size
        )
        if args.max_train_samples and args.max_train_samples < len(train_loader.dataset):
            from torch.utils.data import Subset, DataLoader
            subset_indices = list(range(args.max_train_samples))
            train_subset = Subset(train_loader.dataset, subset_indices)
            train_loader = DataLoader(
                train_subset, batch_size=args.batch_size, shuffle=True,
                num_workers=train_loader.num_workers, drop_last=True
            )
        print(
            f"[Data] Train: {len(train_loader.dataset)} | "
            f"Val: {len(val_loader.dataset)} samples"
        )
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        sys.exit(1)

    # --- Model ---
    model = TSBIRCompositeModel(device=device, loss_attn_weight=args.loss_attn_weight)
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[Model] Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"[Model] MoCo queue size: {model.moco_queue.queue_size:,} negatives")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    # Issue 4 FIX: use non-deprecated form
    scaler = torch.amp.GradScaler('cuda')

    # --- FIX C: Cosine LR scheduler with linear warmup ---
    steps_per_epoch = len(train_loader)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # --- Checkpoint ---
    # Issue 1 FIX: extract current LoRA targets from model to detect ghost-weight mismatches
    checkpoint_mgr = CheckpointManager(checkpoint_dir=args.checkpoint_dir)
    _img_enc = getattr(getattr(model, 'backbone', model), 'image_encoder', None)
    _current_lora_targets = []
    if _img_enc is not None and hasattr(_img_enc, 'peft_config'):
        try:
            _first_cfg = next(iter(_img_enc.peft_config.values()))
            _current_lora_targets = sorted(_first_cfg.target_modules)
        except Exception:
            pass

    if args.fresh_start:
        # Archive stale checkpoints so they're not accidentally loaded,
        # but keep them on disk in case of rollback.
        import glob, shutil
        archive_dir = os.path.join(args.checkpoint_dir, "archived_old_lora")
        os.makedirs(archive_dir, exist_ok=True)
        stale_files = glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_*.pt"))
        if stale_files:
            print(
                f"[FreshStart] Archiving {len(stale_files)} old checkpoint(s) to "
                f"'{archive_dir}/' (LoRA config changed — cannot resume safely)."
            )
            for f in stale_files:
                shutil.move(f, os.path.join(archive_dir, os.path.basename(f)))
        print(
            f"[FreshStart] Starting clean training run with LoRA targets: {_current_lora_targets}"
        )
        model.populate_queue_with_real_photos(train_loader, max_samples=model.moco_queue.queue_size)
        start_epoch, global_step = 0, 0
    else:
        start_epoch, global_step = checkpoint_mgr.load_latest_checkpoint(
            model, optimizer, expected_lora_targets=_current_lora_targets
        )

    # --- Training loop ---
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(start_epoch + 1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, scheduler, device, epoch
        )

        # FIX C — Validate
        val_loss = validate_one_epoch(model, val_loader, device, epoch)

        # Save epoch checkpoint
        global_step += steps_per_epoch
        checkpoint_mgr.save_checkpoint(
            model, optimizer, epoch, global_step, train_loss, val_loss=val_loss,
            filename=f"checkpoint_epoch_{epoch}.pt"
        )
        checkpoint_mgr.save_checkpoint(
            model, optimizer, epoch, global_step, train_loss, val_loss=val_loss,
            filename="checkpoint_latest.pt"
        )

        # FIX C — Best model tracking
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_mgr.save_checkpoint(
                model, optimizer, epoch, global_step, train_loss, val_loss=val_loss,
                filename="checkpoint_best.pt"
            )
            print(f"[Val] [*] New best val loss: {best_val_loss:.4f} -> checkpoint_best.pt saved")
        else:
            patience_counter += 1
            print(
                f"[Val] No improvement ({patience_counter}/{args.early_stop_patience}). "
                f"Best so far: {best_val_loss:.4f}"
            )

        # FIX C — Early stopping
        if patience_counter >= args.early_stop_patience:
            print(
                f"[EarlyStop] Val loss did not improve for {args.early_stop_patience} "
                f"consecutive epochs. Stopping at epoch {epoch}."
            )
            break

    print(f"[Train] Completed. Best val InfoNCE: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

"""
FS-COCO Training Orchestrator Script.
Runs multi-loss contrastive + auxiliary sketch reconstruction training loop.
"""

import os
import sys
import argparse
import time
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.data.fscoco_dataset import get_fscoco_dataloaders
from src.models.composite_model import TSBIRCompositeModel
from src.utils.checkpoint import CheckpointManager

def train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, checkpoint_mgr, global_step):
    model.train()
    total_loss = 0.0
    total_infonce = 0.0
    total_rec = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
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
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_infonce += outputs['loss_infonce'].item()
        total_rec += outputs['loss_rec'].item()
        global_step += 1

        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}",
            'InfoNCE': f"{outputs['loss_infonce'].item():.4f}",
            'Rec': f"{outputs['loss_rec'].item():.4f}"
        })

        if checkpoint_mgr.should_save(global_step):
            checkpoint_mgr.save_checkpoint(model, optimizer, epoch, global_step, loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"--- Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f} ---")
    return avg_loss, global_step


def main():
    parser = argparse.ArgumentParser(description="Train T+SBIR Model on FS-COCO")
    parser.add_argument("--data_dir", type=str, default="fscoco", help="Path to FS-COCO root directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing T+SBIR Training on device: {device}")

    # 1. Load Data
    print(f"Loading FS-COCO dataset from: {args.data_dir}")
    try:
        train_loader, test_loader = get_fscoco_dataloaders(args.data_dir, batch_size=args.batch_size)
        print(f"Dataset successfully loaded: {len(train_loader.dataset)} train samples, {len(test_loader.dataset)} test samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # 2. Build Model
    model = TSBIRCompositeModel(device=device)
    model.to(device)

    # Filter trainable parameters (LoRA + STNet Heads)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable Parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler()
    checkpoint_mgr = CheckpointManager(checkpoint_dir=args.checkpoint_dir)

    start_epoch, global_step = checkpoint_mgr.load_latest_checkpoint(model, optimizer)

    # 3. Training Loop
    for epoch in range(start_epoch + 1, args.epochs + 1):
        _, global_step = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, checkpoint_mgr, global_step
        )
        # Final epoch checkpoint save
        checkpoint_mgr.save_checkpoint(model, optimizer, epoch, global_step, _, filename=f"checkpoint_epoch_{epoch}.pt")

    print("Training sequence completed successfully!")

if __name__ == "__main__":
    main()

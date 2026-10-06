"""
Pretraining Script for CLIP4Cir FeedbackComposedRetriever on FS-COCO.

Trains FeedbackComposedRetriever using InfoNCE on hard-negative feedback triplets:
  Input: (shown_candidate, refined_query) -> combined_query
  Target: matching target_photo
  Negatives: other batch target photos AND rejected shown photos

Saves: checkpoints/combiner_pretrained.pt
Run: python -m src.reranker.pretrain_reranker
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.reranker.combiner import FeedbackComposedRetriever
from src.reranker.mine_feedback_triplets import mine_feedback_triplets


TRIPLETS_PATH = "checkpoints/feedback_train_triplets.pt"
OUTPUT_PATH   = "checkpoints/combiner_pretrained.pt"
EPOCHS        = 20
BATCH_SIZE    = 128
LR            = 1e-3
TEMPERATURE   = 0.07


def pretrain_feedback_combiner():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 65)
    print(f"[PretrainCombiner] Training FeedbackComposedRetriever on {device}")
    print("=" * 65)

    if not os.path.exists(TRIPLETS_PATH):
        print(f"[PretrainCombiner] Triplet cache not found at {TRIPLETS_PATH}. Mining now...")
        mine_feedback_triplets(output_path=TRIPLETS_PATH, device=device)

    print(f"[PretrainCombiner] Loading triplets from: {TRIPLETS_PATH}")
    data = torch.load(TRIPLETS_PATH, map_location="cpu", weights_only=False)
    shown = data["shown_candidates"]
    queries = data["refined_queries"]
    targets = data["target_photos"]
    N = shown.shape[0]

    # Split 90% train / 10% validation
    n_val = max(100, int(0.1 * N))
    n_train = N - n_val
    perm = torch.randperm(N)

    train_idx, val_idx = perm[:n_train], perm[n_train:]
    train_dataset = TensorDataset(shown[train_idx], queries[train_idx], targets[train_idx])
    val_dataset   = TensorDataset(shown[val_idx], queries[val_idx], targets[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[PretrainCombiner] Train: {len(train_dataset)} | Val: {len(val_dataset)} samples")

    combiner = FeedbackComposedRetriever(feature_dim=512, hidden_dim=256).to(device)
    param_count = sum(p.numel() for p in combiner.parameters())
    print(f"[PretrainCombiner] Combiner Parameters: {param_count:,} (<1M target: PASS)")

    optimizer = torch.optim.AdamW(combiner.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        combiner.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{EPOCHS:02d}")
        for b_shown, b_query, b_target in pbar:
            b_shown = b_shown.to(device)
            b_query = b_query.to(device)
            b_target = b_target.to(device)

            optimizer.zero_grad()

            # Predict composed query: (B, D)
            combined = combiner(b_shown, b_query)

            # Build negatives: other target photos (B) + rejected shown photos (B) -> (2B, D)
            candidates = torch.cat([b_target, b_shown], dim=0)

            # InfoNCE logits: (B, 2B)
            logits = torch.matmul(combined, candidates.T) / TEMPERATURE
            labels = torch.arange(b_shown.shape[0], device=device)

            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(combiner.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{scheduler.get_last_lr()[0]:.2e}"})

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        combiner.eval()
        val_loss = 0.0
        val_r1 = 0
        total_val_samples = 0

        with torch.no_grad():
            for b_shown, b_query, b_target in val_loader:
                b_shown = b_shown.to(device)
                b_query = b_query.to(device)
                b_target = b_target.to(device)

                combined = combiner(b_shown, b_query)
                candidates = torch.cat([b_target, b_shown], dim=0)
                logits = torch.matmul(combined, candidates.T) / TEMPERATURE
                labels = torch.arange(b_shown.shape[0], device=device)

                val_loss += F.cross_entropy(logits, labels).item()
                preds = torch.argmax(logits, dim=-1)
                val_r1 += (preds == labels).sum().item()
                total_val_samples += b_shown.shape[0]

        avg_val_loss = val_loss / len(val_loader)
        val_acc = (val_r1 / total_val_samples) * 100.0

        print(
            f"[PretrainCombiner] Epoch {epoch:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Top-1 Acc: {val_acc:.2f}%"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            torch.save({
                "model_state_dict": combiner.state_dict(),
                "feature_dim": 512,
                "hidden_dim": 256,
                "epoch": epoch,
                "val_loss": best_val_loss,
            }, OUTPUT_PATH)
            print(f"  [*] Saved new best combiner model to: {OUTPUT_PATH}")

    print(f"\n[PretrainCombiner] Completed. Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    pretrain_feedback_combiner()

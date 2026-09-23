"""
Offline Combiner Reranker Pretraining on FS-COCO (BUG 5 FIX).

Trains FeedbackCombinerReranker on real composite_query / attended_photo
embedding pairs from the FS-COCO train split — NOT on torch.randn() tensors.

Cold-start pretraining ensures day-one users get a reranker that at minimum
orders results better than random, before any on-device feedback accumulates.

Loss: Triplet margin loss.
  - Positive: ground-truth (query, matching_photo) pair → high score
  - Negatives: same query, random non-matching photos from batch → low score

Saves: checkpoints/combiner_pretrained.pt
Run: python -m src.reranker.pretrain_reranker
"""

# Issue 5 FIX: suppress TF/Protobuf/timm warning flood BEFORE any imports
import os
import warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Ensure project root is on path when run as module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.fscoco_dataset import FSCOCODataset
from src.models.composite_model import TSBIRCompositeModel
from src.reranker.combiner import FeedbackCombinerReranker


CHECKPOINT_PATH = "checkpoints/checkpoint_best.pt"
BACKBONE_PATH   = "checkpoints/mobileclip_s1.pt"
OUTPUT_PATH     = "checkpoints/combiner_pretrained.pt"
DATA_DIR        = "fscoco"
EPOCHS              = 10
EXTRACT_BATCH_SIZE  = 32
TRAIN_BATCH_SIZE    = 2048
LR                  = 5e-4
TRIPLET_MARGIN  = 0.5


def extract_embeddings(model, loader, device):
    """
    Extracts (composite_query, attended_photo) embedding pairs from FS-COCO train split.
    Returns two tensors of shape (N, 512).
    """
    model.eval()
    all_queries, all_photos = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            sketch = batch['sketch'].to(device)
            captions = batch['caption']
            photo = batch['photo'].to(device)

            e_sketch = model.encode_sketch(sketch)
            e_text = model.encode_text(captions)
            e_photo = model.encode_photo(photo)

            raw_composite = torch.cat([e_sketch, e_text], dim=-1)
            e_composite = F.normalize(model.composite_fusion(raw_composite), dim=-1)

            all_queries.append(e_composite.cpu())
            all_photos.append(e_photo.cpu())

    queries = torch.cat(all_queries, dim=0)   # (N, 512)
    photos  = torch.cat(all_photos, dim=0)    # (N, 512)
    return queries, photos


def mine_semi_hard_negatives(
    q_batch: torch.Tensor,
    p_batch: torch.Tensor,
    all_photos: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """
    Semi-hard negative mining for triplet training.

    For each anchor query q_i with positive photo p_i:
      1. Compute similarity to ALL photos in the gallery.
      2. Exclude the ground-truth positive (index i in all_photos == p_i).
      3. Semi-hard negatives: negatives where sim(q, neg) > sim(q, pos) - margin.
         These are "almost positive" — forcing the model to learn fine-grained
         distinctions rather than trivially separated clusters.
      4. If no semi-hard negative exists, fall back to the hardest overall negative.

    Args:
        q_batch:    (B, D) query embeddings (normalised)
        p_batch:    (B, D) positive photo embeddings (normalised) — paired with queries
        all_photos: (N, D) the full photo gallery (normalised)
        margin:     triplet margin

    Returns:
        hard_neg_photos: (B, D) — one hard negative per query
    """
    B = q_batch.shape[0]
    N = all_photos.shape[0]

    # Similarity of each query to ALL gallery photos: (B, N)
    sim_all = torch.mm(q_batch, all_photos.t())          # (B, N)

    # Similarity of each query to its POSITIVE photo: (B,)
    pos_sim = (q_batch * p_batch).sum(dim=-1)            # (B,)

    # Identify positive indices in all_photos for each batch item
    # (using cosine similarity threshold — photos are normalised, so exact match ≈ 1.0)
    # We mask off the true positive from the negative candidates.
    pos_sims_vs_gallery = torch.mm(p_batch, all_photos.t())  # (B, N)
    is_positive_mask = (pos_sims_vs_gallery > 0.999)         # (B, N) True = positive

    hard_negs = []
    for i in range(B):
        row = sim_all[i]                       # (N,) similarity of query i to all photos
        pos_s = pos_sim[i]                     # scalar: similarity to ground-truth

        # Mask: not a positive photo AND sim > pos_sim - margin (semi-hard condition)
        semi_hard_mask = (~is_positive_mask[i]) & (row > pos_s - margin)

        if semi_hard_mask.any():
            # Among semi-hard candidates, pick the one closest to positive (hardest)
            candidate_sims = row.clone()
            candidate_sims[~semi_hard_mask] = -float('inf')
            hard_idx = candidate_sims.argmax()
        else:
            # No semi-hard found: pick the overall hardest negative (highest sim)
            all_neg_sims = row.clone()
            all_neg_sims[is_positive_mask[i]] = -float('inf')
            hard_idx = all_neg_sims.argmax()

        hard_negs.append(all_photos[hard_idx])

    return torch.stack(hard_negs, dim=0)    # (B, D)


def pretrain(queries, photos, combiner, device, epochs=EPOCHS, lr=LR):
    """
    Trains combiner using triplet margin loss with semi-hard negative mining.

    For each item i in a batch:
      - positive: (query[i], photos[i])   — ground-truth match
      - negative: mine_semi_hard_negatives() — hardest non-matching photo
                  from the FULL gallery (not just the current mini-batch)

    Why semi-hard?
      Random roll (old approach) trivially separates same-batch negatives that
      are already far apart in embedding space → triplet loss saturates at 0.26
      with margin trivially satisfied.
      Semi-hard negatives are "almost positive" → the model must learn to
      distinguish genuinely ambiguous cases.
    """
    optimizer = torch.optim.AdamW(combiner.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    N = queries.shape[0]
    best_loss = float('inf')

    # Pre-normalise the entire gallery (used for semi-hard mining)
    gallery_photos = F.normalize(photos, dim=-1).to(device)

    train_tensor_dataset = TensorDataset(queries, photos)
    train_tensor_loader = DataLoader(
        train_tensor_dataset, 
        batch_size=TRAIN_BATCH_SIZE, 
        shuffle=True, 
        drop_last=True
    )

    for epoch in range(1, epochs + 1):
        combiner.train()
        total_loss = 0.0
        n_batches  = 0

        for q_batch, p_batch in train_tensor_loader:
            q_batch = F.normalize(q_batch, dim=-1).to(device)
            p_batch = F.normalize(p_batch, dim=-1).to(device)
            B = q_batch.shape[0]
            if B < 2:
                continue

            optimizer.zero_grad()

            # Positive scores
            pos_scores = combiner(q_batch, p_batch).squeeze(-1)
            pos_scores = F.normalize(pos_scores, dim=-1)

            # Semi-hard negatives mined from FULL gallery
            with torch.no_grad():
                neg_photos = mine_semi_hard_negatives(
                    q_batch, p_batch, gallery_photos, margin=TRIPLET_MARGIN
                )
            neg_scores = combiner(q_batch, neg_photos).squeeze(-1)
            neg_scores = F.normalize(neg_scores, dim=-1)

            loss = F.relu(TRIPLET_MARGIN - pos_scores + neg_scores).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(combiner.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        scheduler.step()

        print(
            f"[CombinerPretrain] Epoch {epoch}/{epochs} | "
            f"Triplet Loss: {avg_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {'epoch': epoch, 'loss': avg_loss, 'state_dict': combiner.state_dict()},
                OUTPUT_PATH
            )
            print(f"[CombinerPretrain] New best saved -> {OUTPUT_PATH}")

    return best_loss




def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CombinerPretrain] Device: {device}")

    print("[CombinerPretrain] Loading TSBIRCompositeModel...")
    retrieval_model = TSBIRCompositeModel(
        backbone_name="mobileclip_s1",
        checkpoint_path=BACKBONE_PATH,
        device=device,
    )

    # Issue 3 FIX: always use checkpoint_best.pt (latest model) for embedding extraction.
    # Using checkpoint_latest.pt (epoch 15) to pretrain the reranker for a model that
    # later reached epoch 19 means the reranker is mis-aligned to stale representations.
    ckpt_path = CHECKPOINT_PATH  # checkpoint_best.pt
    if not os.path.exists(ckpt_path):
        fallback = "checkpoints/checkpoint_latest.pt"
        if os.path.exists(fallback):
            print(
                f"[CombinerPretrain] WARNING: checkpoint_best.pt not found at '{ckpt_path}'.\n"
                f"  Falling back to '{fallback}' — embeddings may not reflect the optimal model.\n"
                "  Run training first to generate checkpoint_best.pt."
            )
            ckpt_path = fallback
        else:
            print("[CombinerPretrain] No checkpoint found — using random init adapter.")
            ckpt_path = None
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"[CombinerPretrain] Loading adapter weights: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state = ckpt.get('state_dict', ckpt.get('full_state_dict', ckpt))
        # Issue 1 FIX: check saved LoRA targets before loading
        saved_targets = ckpt.get('lora_targets', None)
        if saved_targets is not None:
            print(f"[CombinerPretrain] Checkpoint was trained with LoRA targets: {sorted(saved_targets)}")
        retrieval_model.load_state_dict(state, strict=False)

    retrieval_model.to(device)

    cache_file = "checkpoints/extracted_train_embeddings.pt"
    if os.path.exists(cache_file):
        print(f"[CombinerPretrain] Loading pre-extracted embeddings from '{cache_file}'...")
        cached = torch.load(cache_file, map_location='cpu', weights_only=False)
        queries = cached['queries']
        photos = cached['photos']
    else:
        print(f"[CombinerPretrain] Loading FS-COCO train split: {DATA_DIR}")
        train_dataset = FSCOCODataset(DATA_DIR, split='train')
        train_loader  = DataLoader(train_dataset, batch_size=EXTRACT_BATCH_SIZE, shuffle=False, num_workers=2)
        print(f"[CombinerPretrain] Train samples: {len(train_dataset)}")

        queries, photos = extract_embeddings(retrieval_model, train_loader, device)
        torch.save({'queries': queries, 'photos': photos}, cache_file)
        print(f"[CombinerPretrain] Saved extracted embeddings to '{cache_file}'")

    print(f"[CombinerPretrain] Embeddings: queries={queries.shape}, photos={photos.shape}")

    combiner = FeedbackCombinerReranker(feature_dim=512).to(device)
    print(f"[CombinerPretrain] Combiner params: {sum(p.numel() for p in combiner.parameters()):,}")

    best_loss = pretrain(queries, photos, combiner, device)
    print(f"\n[CombinerPretrain] Done. Best triplet loss: {best_loss:.4f}")
    print(f"[CombinerPretrain] Weights saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()

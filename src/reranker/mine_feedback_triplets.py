"""
Hard Negative Feedback Triplet Mining Script for FS-COCO.

Mines realistic feedback triplets using the trained Part A dual-encoder model:
  For each training sample i (query_i, target_photo_i):
    - Compute retrieval similarity over all training gallery photos.
    - Identify the nearest false positive photo j != i (the photo a user would actually be shown erroneously).
    - Save triplet: (shown_candidate_j, refined_query_i, target_photo_i).

These triplets form the training set for the CLIP4Cir FeedbackComposedRetriever.
Saves: checkpoints/feedback_train_triplets.pt
"""

import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.fscoco_dataset import FSCOCODataset
from torch.utils.data import DataLoader
from src.models.composite_model import TSBIRCompositeModel


def mine_feedback_triplets(
    checkpoint_path: str = "checkpoints/checkpoint_best.pt",
    data_dir: str = "fscoco",
    output_path: str = "checkpoints/feedback_train_triplets.pt",
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 65)
    print("[MineFeedback] Mining Hard-Negative Feedback Triplets on FS-COCO Train Split")
    print("=" * 65)

    print(f"[MineFeedback] Device: {device}")
    model = TSBIRCompositeModel(device=device)

    if os.path.exists(checkpoint_path):
        print(f"[MineFeedback] Loading trained dual-encoder weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt.get("full_state_dict", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[MineFeedback] Weights loaded (unexpected: {len(unexpected)}, missing: {len(missing)}).")
    else:
        print(f"[MineFeedback] WARNING: Checkpoint {checkpoint_path} not found! Using random init.")

    model.to(device)
    model.eval()

    train_dataset = FSCOCODataset(data_dir, split="train")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    N = len(train_dataset)
    print(f"[MineFeedback] Extracting embeddings for {N} training samples...")

    all_queries = []
    all_photos = []

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Extracting Train Features"):
            sketch = batch["sketch"].to(device)
            captions = batch["caption"]
            photo = batch["photo"].to(device)

            e_sketch = model.encode_sketch(sketch)
            e_text = model.encode_text(captions)
            e_photo = model.encode_photo(photo)

            raw_comp = torch.cat([e_sketch, e_text], dim=-1)
            e_comp = F.normalize(model.composite_fusion(raw_comp), dim=-1)

            all_queries.append(e_comp.cpu())
            all_photos.append(e_photo.cpu())

    queries = torch.cat(all_queries, dim=0)   # (N, D)
    photos = torch.cat(all_photos, dim=0)     # (N, D)
    print(f"[MineFeedback] Extracted {queries.shape[0]} queries and {photos.shape[0]} photo embeddings.")

    print("[MineFeedback] Mining nearest wrong candidate per query...")
    # Process in chunks to avoid large (N, N) matrix OOM
    CHUNK = 512
    shown_candidates = []
    refined_queries = []
    target_photos = []

    for start_idx in tqdm(range(0, N, CHUNK), desc="Mining Hard Negatives"):
        end_idx = min(start_idx + CHUNK, N)
        q_chunk = queries[start_idx:end_idx]   # (C, D)
        targets_chunk = photos[start_idx:end_idx] # (C, D)

        # Cosine similarity against all photos: (C, N)
        sim_chunk = torch.matmul(q_chunk, photos.T)

        for i, row in enumerate(sim_chunk):
            global_idx = start_idx + i
            # Mask out the ground truth positive photo
            row[global_idx] = -float("inf")
            # Pick top-1 hardest false positive
            hard_neg_idx = torch.argmax(row).item()

            shown_candidates.append(photos[hard_neg_idx])
            refined_queries.append(q_chunk[i])
            target_photos.append(targets_chunk[i])

    shown_candidates = torch.stack(shown_candidates, dim=0)
    refined_queries = torch.stack(refined_queries, dim=0)
    target_photos = torch.stack(target_photos, dim=0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        "shown_candidates": shown_candidates,
        "refined_queries": refined_queries,
        "target_photos": target_photos,
    }, output_path)

    print(f"[MineFeedback] Successfully saved {len(shown_candidates)} triplets to: {output_path}")
    print(f"  shown_candidates: {shown_candidates.shape}")
    print(f"  refined_queries:  {refined_queries.shape}")
    print(f"  target_photos:    {target_photos.shape}")


if __name__ == "__main__":
    mine_feedback_triplets()

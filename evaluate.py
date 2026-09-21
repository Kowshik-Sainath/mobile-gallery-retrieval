"""
Evaluation Protocol Script for FS-COCO T+SBIR System.
Reports Recall@1, Recall@5, Recall@10, and NDCG@10 across modal combinations.

BUG 4 FIX: Each similarity mode now uses the correct embedding space:
  - Text-only / Sketch-only: query vs raw photo_embeds (clean frozen tower)
  - Composite (STNet Aligned): composite_query vs attended_photo_embeds
  - Diagnostic: prints R@1 gap between STNet-aligned vs raw-photo composite eval
    to quantify train/eval alignment quality.
"""

import os
import warnings
# Issue 5 FIX: suppress TF/Protobuf/timm warning flood BEFORE other imports
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.fscoco_dataset import FSCOCODataset
from torch.utils.data import DataLoader
from src.models.composite_model import TSBIRCompositeModel
from src.utils.metrics import compute_recall_at_k, compute_ndcg_at_k


def evaluate_retrieval(model, test_loader, device="cuda"):
    """
    Extracts embeddings across the full test set, then computes retrieval metrics.

    Similarity spaces used per mode (BUG 4 FIX — documented explicitly):
      Text-only        : text_embeds       vs photo_embeds       (both from frozen tower)
      Sketch-only      : sketch_embeds     vs photo_embeds       (sketch via LoRA-on, photo via LoRA-off)
      Late Fusion      : 0.5*sim_text + 0.5*sim_sketch           (pure baseline)
      STNet Aligned    : composite_query   vs attended_photo      (matches training loss target) ✓
      Diagnostic       : composite_query   vs photo_embeds        (misaligned; should be lower than STNet)
    """
    model.eval()

    sketch_embeds_list = []
    text_embeds_list = []
    composite_embeds_list = []
    photo_embeds_list = []
    attended_photo_embeds_list = []

    print("Extracting test set embeddings...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting Features"):
            sketch = batch['sketch'].to(device)
            captions = batch['caption']
            photo = batch['photo'].to(device)

            # Sketch: LoRA-adapted vision tower
            e_sketch = model.encode_sketch(sketch)
            # Text: frozen tower + text adapter
            e_text = model.encode_text(captions)
            # Photo (raw): frozen tower, adapters disabled
            e_photo = model.encode_photo(photo)
            # Photo (attended): sketch-guided attention over patch tokens
            photo_patches = model.encode_photo_patches(photo)
            attended_photo, _ = model.attention_pooling(e_sketch, photo_patches)
            attended_photo = F.normalize(attended_photo, dim=-1)
            # Composite query
            raw_composite = torch.cat([e_sketch, e_text], dim=-1)
            e_composite = F.normalize(model.composite_fusion(raw_composite), dim=-1)

            sketch_embeds_list.append(e_sketch.cpu().numpy())
            text_embeds_list.append(e_text.cpu().numpy())
            composite_embeds_list.append(e_composite.cpu().numpy())
            photo_embeds_list.append(e_photo.cpu().numpy())
            attended_photo_embeds_list.append(attended_photo.cpu().numpy())

    sketch_embeds = np.concatenate(sketch_embeds_list, axis=0)
    text_embeds = np.concatenate(text_embeds_list, axis=0)
    composite_embeds = np.concatenate(composite_embeds_list, axis=0)
    photo_embeds = np.concatenate(photo_embeds_list, axis=0)
    attended_photo_embeds = np.concatenate(attended_photo_embeds_list, axis=0)

    # --- Similarity matrices ---
    # BUG 4 FIX: text-only and sketch-only correctly compare against raw photo_embeds.
    # The model's text/sketch-only encoding paths do not use attention pooling during
    # training; raw photo_embeds is the correct retrieval gallery for these modes.
    sim_text = np.dot(text_embeds, photo_embeds.T)
    sim_sketch = np.dot(sketch_embeds, photo_embeds.T)

    # Composite: composite_query vs attended_photo (matches training InfoNCE target) ✓
    sim_composite = np.dot(composite_embeds, attended_photo_embeds.T)

    # Diagnostic: composite vs raw photo (the misaligned space — should score LOWER)
    # Prints the gap to quantify how much train/eval alignment matters.
    sim_composite_misaligned = np.dot(composite_embeds, photo_embeds.T)

    results = {}

    # 1. Text-only
    r_text = compute_recall_at_k(sim_text)
    ndcg_text = compute_ndcg_at_k(sim_text)
    results['Text-Only'] = {**r_text, 'NDCG@10': ndcg_text}

    # 2. Sketch-only
    r_sketch = compute_recall_at_k(sim_sketch)
    ndcg_sketch = compute_ndcg_at_k(sim_sketch)
    results['Sketch-Only'] = {**r_sketch, 'NDCG@10': ndcg_sketch}

    # 3. Late Fusion baseline
    sim_late = 0.5 * sim_sketch + 0.5 * sim_text
    r_late = compute_recall_at_k(sim_late)
    ndcg_late = compute_ndcg_at_k(sim_late)
    results['Composite (Late Fusion)'] = {**r_late, 'NDCG@10': ndcg_late}

    # 4. Composite STNet Aligned (correct space — matches training)
    r_comp = compute_recall_at_k(sim_composite)
    ndcg_comp = compute_ndcg_at_k(sim_composite)
    results['Composite (STNet Aligned)'] = {**r_comp, 'NDCG@10': ndcg_comp}

    # 5. Diagnostic: misaligned composite vs raw photo (for analysis only)
    r_misaligned = compute_recall_at_k(sim_composite_misaligned)
    results['[Diagnostic] Composite vs Raw Photo'] = {**r_misaligned, 'NDCG@10': 0.0}

    # BUG 4 Diagnostic: print the alignment gap
    gap = r_comp.get('R@1', 0) - r_misaligned.get('R@1', 0)
    print(
        f"\n[BUG4 Diagnostic] STNet-Aligned R@1: {r_comp.get('R@1', 0):.2f}% | "
        f"Misaligned R@1: {r_misaligned.get('R@1', 0):.2f}% | "
        f"Alignment Gap: {gap:+.2f}%"
    )
    if gap <= 0:
        print(
            "[Warning] STNet-Aligned R@1 is not higher than misaligned. "
            "This indicates attention pooling may not be adding signal. "
            "Check patch token diversity (BUG 3) and LoRA attachment (BUG 2)."
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate T+SBIR Model on FS-COCO Test Split")
    parser.add_argument("--data_dir", type=str, default="fscoco")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Evaluation on device: {device}")

    test_dataset = FSCOCODataset(args.data_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded FS-COCO Test Split: {len(test_dataset)} test samples.")

    model = TSBIRCompositeModel(device=device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        # FIX B: checkpoints now save adapter+head state_dict only (no full_state_dict).
        # Support both old (full_state_dict) and new (state_dict) formats.
        state = checkpoint.get('state_dict', checkpoint.get('full_state_dict', checkpoint))
        model.load_state_dict(state, strict=False)

    model.to(device)

    results = evaluate_retrieval(model, test_loader, device=device)

    print("\n" + "=" * 70)
    print(f"{'Evaluation Strategy':<35} | {'R@1':<7} | {'R@5':<7} | {'R@10':<7} | {'NDCG@10':<7}")
    print("-" * 70)
    for mode, metrics in results.items():
        print(
            f"{mode:<35} | {metrics['R@1']:<7.2f} | {metrics['R@5']:<7.2f} | "
            f"{metrics['R@10']:<7.2f} | {metrics['NDCG@10']:<7.2f}"
        )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

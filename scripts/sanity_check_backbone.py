"""
Sanity check script to verify whether the pretrained mobileclip_s1 backbone
actually contains valid pretrained weights with cross-modal alignment.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.backbone import load_mobileclip_backbone
from src.data.fscoco_dataset import FSCOCODataset


def run_sanity_check():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SanityCheck] Device: {device}")

    # 1. Load backbone with NO LoRA applied
    print("[SanityCheck] Loading base mobileclip_s1 backbone...")
    model, tokenizer, _ = load_mobileclip_backbone(
        model_name="mobileclip_s1",
        checkpoint_path="checkpoints/mobileclip_s1.pt",
        device=device
    )
    model.eval()

    # 2. Pick 5 clearly different real images with clearly different captions from FS-COCO
    print("[SanityCheck] Loading dataset samples from test split...")
    dataset = FSCOCODataset("fscoco", split="test")
    
    # Pick 5 spaced out samples to ensure variety
    indices = [0, 50, 100, 150, 200]
    indices = [idx for idx in indices if idx < len(dataset)]
    if len(indices) < 5:
        indices = list(range(min(5, len(dataset))))

    samples = [dataset[i] for i in indices]
    captions = [s["caption"] for s in samples]
    photos = torch.stack([s["photo"] for s in samples]).to(device)

    print("\n--- Selected Samples ---")
    for i, (idx, cap) in enumerate(zip(indices, captions)):
        print(f"Sample {i} (Index {idx}): \"{cap[:80]}...\"")

    # 3. Compute image and text embeddings directly from model
    with torch.no_grad():
        # Encode images
        img_feats = model.encode_image(photos)
        img_feats = F.normalize(img_feats, dim=-1)

        # Encode text
        text_tokens = tokenizer(captions).to(device)
        text_feats = model.encode_text(text_tokens)
        text_feats = F.normalize(text_feats, dim=-1)

    # 4. Compute cosine similarity matrix: text x image (5 x 5)
    # Row i = text i, Column j = image j
    sim_matrix = torch.matmul(text_feats, img_feats.T).cpu().numpy()

    print("\n--- Cosine Similarity Matrix (Rows: Text, Cols: Image) ---")
    header = "       " + " ".join([f"Img_{j:<3}" for j in range(len(samples))])
    print(header)
    for i in range(len(samples)):
        row_str = f"Txt_{i:<2} " + " ".join([f"{sim_matrix[i, j]:7.4f}" for j in range(len(samples))])
        print(row_str)

    diagonal = np.diag(sim_matrix)
    mask = ~np.eye(len(samples), dtype=bool)
    off_diagonal = sim_matrix[mask]

    mean_diag = np.mean(diagonal)
    min_diag = np.min(diagonal)
    mean_off = np.mean(off_diagonal)
    max_off = np.max(off_diagonal)

    # Rank check: for each text, is the correct image ranked #1?
    correct_rank1_hits = sum(np.argmax(sim_matrix[i]) == i for i in range(len(samples)))

    print("\n--- Summary Statistics ---")
    print(f"Diagonal (Correct Pairs)   : Mean = {mean_diag:.4f}, Min = {min_diag:.4f}")
    print(f"Off-Diagonal (Wrong Pairs) : Mean = {mean_off:.4f}, Max = {max_off:.4f}")
    print(f"Diagonal - Off-Diagonal Gap: {mean_diag - mean_off:.4f}")
    print(f"Rank-1 Matches             : {correct_rank1_hits} / {len(samples)}")

    # 5. Verdict
    # Pretrained MobileCLIP should have diagonal > off-diagonal and positive gap
    passed = (mean_diag > mean_off + 0.05) and (correct_rank1_hits >= 3)

    print("\n" + "=" * 50)
    if passed:
        print("[SanityCheck] GATE VERDICT: PASS")
        print("Pretrained weights are active and exhibit strong text-image alignment.")
    else:
        print("[SanityCheck] GATE VERDICT: FAIL")
        print("Weights do not exhibit meaningful text-image alignment!")
    print("=" * 50 + "\n")

    return passed


if __name__ == "__main__":
    passed = run_sanity_check()
    sys.exit(0 if passed else 1)

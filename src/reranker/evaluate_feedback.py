"""
Evaluation of Feedback-Driven Refinement on FS-COCO Test Split.

Evaluates the CLIP4Cir FeedbackComposedRetriever across three distinct regimes:
  1. Self-Correction Probe:
     Uses (shown_candidate_wrong_top1, composite_query) -> combined_query.
  2. Genuine New Feedback Probe:
     Initial query uses partial caption (first half of words + sketch).
     Refined query adds the remaining feedback text (full caption + sketch).
  3. Trivial Exclusion Baseline:
     Original query with the shown wrong candidate simply masked out (zero learning).
     The learned Combiner MUST beat this baseline to prove genuine semantic steering.

Run: python -m src.reranker.evaluate_feedback
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.fscoco_dataset import FSCOCODataset
from torch.utils.data import DataLoader
from src.models.composite_model import TSBIRCompositeModel
from src.reranker.combiner import FeedbackComposedRetriever
from src.utils.metrics import compute_recall_at_k


def split_caption_for_feedback(caption: str):
    """Splits caption into initial partial text and full text for simulated user feedback."""
    words = caption.strip().split()
    if len(words) <= 2:
        return caption, caption
    half_len = max(1, len(words) // 2)
    partial_caption = " ".join(words[:half_len])
    return partial_caption, caption


def evaluate_feedback_retrieval(
    data_dir: str = "fscoco",
    checkpoint_path: str = "checkpoints/checkpoint_best.pt",
    combiner_path: str = "checkpoints/combiner_pretrained.pt",
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 75)
    print("[FeedbackEval] Evaluating Feedback-Driven Query Composition & Refinement")
    print("=" * 75)
    print(f"[FeedbackEval] Device: {device}")

    # 1. Load dual-encoder model
    model = TSBIRCompositeModel(device=device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt.get("full_state_dict", ckpt))
        model.load_state_dict(state, strict=False)
        print(f"[FeedbackEval] Loaded backbone checkpoint: {checkpoint_path}")
    model.to(device).eval()

    # 2. Load trained Combiner
    combiner = FeedbackComposedRetriever(feature_dim=512, hidden_dim=256)
    if os.path.exists(combiner_path):
        c_ckpt = torch.load(combiner_path, map_location="cpu", weights_only=False)
        c_state = c_ckpt.get("model_state_dict", c_ckpt)
        combiner.load_state_dict(c_state, strict=False)
        print(f"[FeedbackEval] Loaded pretrained combiner from: {combiner_path}")
    else:
        print(f"[FeedbackEval] WARNING: {combiner_path} not found. Running with init weights.")
    combiner.to(device).eval()

    # 3. Load FS-COCO Test Split
    test_dataset = FSCOCODataset(data_dir, split="test")
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    N = len(test_dataset)
    print(f"[FeedbackEval] FS-COCO Test Set: {N} samples")

    test_cache_path = "checkpoints/test_eval_features.pt"
    if os.path.exists(test_cache_path):
        print(f"[FeedbackEval] Loading cached test features from: {test_cache_path}")
        t_cache = torch.load(test_cache_path, map_location="cpu", weights_only=False)
        photos = t_cache["photos"]
        full_queries = t_cache["full_queries"]
        part_queries = t_cache["part_queries"]
    else:
        # Extract all test embeddings
        print("[FeedbackEval] Extracting test split embeddings (full and partial)...")
        all_photos = []
        all_full_composites = []
        all_partial_composites = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Feature Extraction"):
                sketch = batch["sketch"].to(device)
                full_captions = batch["caption"]
                photo = batch["photo"].to(device)

                e_sketch = model.encode_sketch(sketch)
                e_photo  = model.encode_photo(photo)
                all_photos.append(e_photo.cpu())

                # Full composite query
                e_text_full = model.encode_text(full_captions)
                raw_full = torch.cat([e_sketch, e_text_full], dim=-1)
                comp_full = F.normalize(model.composite_fusion(raw_full), dim=-1)
                all_full_composites.append(comp_full.cpu())

                # Partial composite query for genuine feedback probe
                partial_caps = [split_caption_for_feedback(c)[0] for c in full_captions]
                e_text_part = model.encode_text(partial_caps)
                raw_part = torch.cat([e_sketch, e_text_part], dim=-1)
                comp_part = F.normalize(model.composite_fusion(raw_part), dim=-1)
                all_partial_composites.append(comp_part.cpu())

        photos       = torch.cat(all_photos, dim=0)               # (N, D)
        full_queries = torch.cat(all_full_composites, dim=0)      # (N, D)
        part_queries = torch.cat(all_partial_composites, dim=0)   # (N, D)
        torch.save({
            "photos": photos,
            "full_queries": full_queries,
            "part_queries": part_queries
        }, test_cache_path)

    # Precompute full gallery matrix for cosine scan
    gallery_matrix = photos.numpy()                           # (N, D)

    # -----------------------------------------------------------------------
    # PROBE 1 & BASELINE: Self-Correction on Initial Top-1 Misses
    # -----------------------------------------------------------------------
    print("\n[FeedbackEval] Running Probe 1: Self-Correction & Trivial Exclusion Baseline...")
    sim_initial = np.dot(full_queries.numpy(), gallery_matrix.T) # (N, N)

    initial_miss_indices = []
    for i in range(N):
        top1 = np.argmax(sim_initial[i])
        if top1 != i:
            initial_miss_indices.append((i, top1))

    n_misses = len(initial_miss_indices)
    miss_rate = (n_misses / N) * 100.0
    print(f"[FeedbackEval] Initial Top-1 Misses: {n_misses} / {N} ({miss_rate:.2f}%)")

    # Evaluate on the initial miss subset:
    # A. Initial baseline (no refinement): sim_initial[i]
    # B. Trivial exclusion baseline: sim_initial[i] with top1 masked to -inf
    # C. Learned Combiner: combiner(photo_top1, full_query) @ gallery.T (with photo_top1 masked)

    init_ranks, excl_ranks, comb_ranks = [], [], []

    with torch.no_grad():
        for i, wrong_top1 in initial_miss_indices:
            # GT target photo is i
            # Initial ranks
            row_init = sim_initial[i].copy()
            # Order descending
            init_order = np.argsort(row_init)[::-1]
            rank_init = np.where(init_order == i)[0][0] + 1
            init_ranks.append(rank_init)

            # Trivial exclusion baseline: mask wrong_top1
            row_excl = sim_initial[i].copy()
            row_excl[wrong_top1] = -1e9
            excl_order = np.argsort(row_excl)[::-1]
            rank_excl = np.where(excl_order == i)[0][0] + 1
            excl_ranks.append(rank_excl)

            # Learned Combiner
            sh_cand = photos[wrong_top1:wrong_top1+1].to(device)
            ref_q   = full_queries[i:i+1].to(device)
            c_query = combiner(sh_cand, ref_q).squeeze(0).cpu().numpy()

            row_comb = np.dot(c_query, gallery_matrix.T)
            row_comb[wrong_top1] = -1e9   # Explicitly filter shown candidate from results
            comb_order = np.argsort(row_comb)[::-1]
            rank_comb = np.where(comb_order == i)[0][0] + 1
            comb_ranks.append(rank_comb)

    init_r1 = sum(1 for r in init_ranks if r <= 1) / n_misses * 100.0
    init_r5 = sum(1 for r in init_ranks if r <= 5) / n_misses * 100.0
    init_r10 = sum(1 for r in init_ranks if r <= 10) / n_misses * 100.0

    excl_r1 = sum(1 for r in excl_ranks if r <= 1) / n_misses * 100.0
    excl_r5 = sum(1 for r in excl_ranks if r <= 5) / n_misses * 100.0
    excl_r10 = sum(1 for r in excl_ranks if r <= 10) / n_misses * 100.0

    comb_r1 = sum(1 for r in comb_ranks if r <= 1) / n_misses * 100.0
    comb_r5 = sum(1 for r in comb_ranks if r <= 5) / n_misses * 100.0
    comb_r10 = sum(1 for r in comb_ranks if r <= 10) / n_misses * 100.0

    # -----------------------------------------------------------------------
    # PROBE 2: Genuine New Feedback (Partial Query -> Refined Full Query)
    # -----------------------------------------------------------------------
    print("\n[FeedbackEval] Running Probe 2: Genuine New Feedback Simulation...")
    sim_part = np.dot(part_queries.numpy(), gallery_matrix.T)   # (N, N)

    part_miss_indices = []
    for i in range(N):
        top1 = np.argmax(sim_part[i])
        if top1 != i:
            part_miss_indices.append((i, top1))

    n_part_misses = len(part_miss_indices)
    p_init_ranks, p_excl_ranks, p_comb_ranks = [], [], []

    with torch.no_grad():
        for i, wrong_top1 in part_miss_indices:
            # Initial (partial query)
            row_p = sim_part[i].copy()
            p_order = np.argsort(row_p)[::-1]
            p_init_ranks.append(np.where(p_order == i)[0][0] + 1)

            # Trivial exclusion on partial query
            row_p_excl = sim_part[i].copy()
            row_p_excl[wrong_top1] = -1e9
            p_excl_order = np.argsort(row_p_excl)[::-1]
            p_excl_ranks.append(np.where(p_excl_order == i)[0][0] + 1)

            # Learned Combiner: (shown_candidate, refined_full_query)
            sh_cand = photos[wrong_top1:wrong_top1+1].to(device)
            ref_q   = full_queries[i:i+1].to(device)  # New information added
            c_query = combiner(sh_cand, ref_q).squeeze(0).cpu().numpy()

            row_c = np.dot(c_query, gallery_matrix.T)
            row_c[wrong_top1] = -1e9
            c_order = np.argsort(row_c)[::-1]
            p_comb_ranks.append(np.where(c_order == i)[0][0] + 1)

    p_init_r1 = sum(1 for r in p_init_ranks if r <= 1) / n_part_misses * 100.0
    p_init_r5 = sum(1 for r in p_init_ranks if r <= 5) / n_part_misses * 100.0
    p_init_r10 = sum(1 for r in p_init_ranks if r <= 10) / n_part_misses * 100.0

    p_excl_r1 = sum(1 for r in p_excl_ranks if r <= 1) / n_part_misses * 100.0
    p_excl_r5 = sum(1 for r in p_excl_ranks if r <= 5) / n_part_misses * 100.0
    p_excl_r10 = sum(1 for r in p_excl_ranks if r <= 10) / n_part_misses * 100.0

    p_comb_r1 = sum(1 for r in p_comb_ranks if r <= 1) / n_part_misses * 100.0
    p_comb_r5 = sum(1 for r in p_comb_ranks if r <= 5) / n_part_misses * 100.0
    p_comb_r10 = sum(1 for r in p_comb_ranks if r <= 10) / n_part_misses * 100.0

    # -----------------------------------------------------------------------
    # Output Verification Tables
    # -----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PROBE 1: SELF-CORRECTION (Initial Top-1 Misses, N=" + str(n_misses) + ")")
    print("=" * 78)
    print(f"{'Method':<35} | {'R@1 (%)':>9} | {'R@5 (%)':>9} | {'R@10 (%)':>9}")
    print("-" * 78)
    print(f"{'1. Initial Missed Retrieval':<35} | {init_r1:>9.2f} | {init_r5:>9.2f} | {init_r10:>9.2f}")
    print(f"{'2. Trivial Exclusion Baseline':<35} | {excl_r1:>9.2f} | {excl_r5:>9.2f} | {excl_r10:>9.2f}")
    print(f"{'3. Learned CLIP4Cir Combiner':<35} | {comb_r1:>9.2f} | {comb_r5:>9.2f} | {comb_r10:>9.2f}")
    print("-" * 78)
    print(f"{'Refinement Gain vs Initial (Delta-R@1)':<35} | {comb_r1 - init_r1:>+9.2f}%")
    print(f"{'Refinement Gain vs Trivial (Delta-R@1)':<35} | {comb_r1 - excl_r1:>+9.2f}%")
    print("=" * 78)

    print("\n" + "=" * 78)
    print("PROBE 2: GENUINE NEW FEEDBACK (Partial Query Misses, N=" + str(n_part_misses) + ")")
    print("=" * 78)
    print(f"{'Method':<35} | {'R@1 (%)':>9} | {'R@5 (%)':>9} | {'R@10 (%)':>9}")
    print("-" * 78)
    print(f"{'1. Initial Partial Query':<35} | {p_init_r1:>9.2f} | {p_init_r5:>9.2f} | {p_init_r10:>9.2f}")
    print(f"{'2. Trivial Exclusion Baseline':<35} | {p_excl_r1:>9.2f} | {p_excl_r5:>9.2f} | {p_excl_r10:>9.2f}")
    print(f"{'3. Learned Combiner (+ Feedback)':<35} | {p_comb_r1:>9.2f} | {p_comb_r5:>9.2f} | {p_comb_r10:>9.2f}")
    print("-" * 78)
    print(f"{'Refinement Gain vs Initial (Delta-R@1)':<35} | {p_comb_r1 - p_init_r1:>+9.2f}%")
    print(f"{'Refinement Gain vs Trivial (Delta-R@1)':<35} | {p_comb_r1 - p_excl_r1:>+9.2f}%")
    print("=" * 78 + "\n")

    # Gate Verification
    gate_probe1 = (comb_r1 > excl_r1)
    gate_probe2 = (p_comb_r1 > p_excl_r1)

    print(f"[Gate Check] Probe 1 beats Trivial Exclusion (Delta-R@1 = {comb_r1 - excl_r1:+.2f}%): {'PASS' if gate_probe1 else 'FAIL'}")
    print(f"[Gate Check] Probe 2 beats Trivial Exclusion (Delta-R@1 = {p_comb_r1 - p_excl_r1:+.2f}%): {'PASS' if gate_probe2 else 'FAIL'}")


if __name__ == "__main__":
    evaluate_feedback_retrieval()

"""
Evaluation Protocol Script for FS-COCO T+SBIR System.
Reports Recall@1, Recall@5, Recall@10, and NDCG@10 across modal combinations.

Component 5 (Eval/Train Alignment Fix):
  The key bug in the previous version: `attended_photo` was computed per-batch
  using THAT BATCH'S sketch to attend THAT BATCH'S paired photo patches. This means:
    - Gallery photo j's "attended embedding" was computed using sketch_j (its GT pair),
      NOT using the query sketch_i. The resulting sim_matrix diagonal was oracle-attended
      (i→i) while off-diagonals were incoherent cross-attention (sketch_j attending photo_k).
  This is mathematically wrong for retrieval evaluation.

  Correct O(N_query × N_gallery) protocol:
    Phase 1: Extract ALL gallery photo patch tensors → (N_gallery, 49, D). Stored CPU.
    Phase 2: Extract ALL query composite embeddings, sketch embeddings. Stored CPU.
    Phase 3: For each query i, attend sketch_i over EVERY gallery photo's patches j:
               attended_ij = attention_pooling(sketch_i, patches_j)
             Compute sim_matrix[i, j] = composite_query_i · attended_ij
    This is O(N_query × N_gallery) attention operations — correct for retrieval.
"""

import os
import warnings
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


# Number of gallery photos to process per attention batch (memory budget ~2GB GPU)
ATTENTION_CHUNK = 128


def evaluate_retrieval(model, test_loader, device="cuda"):
    """
    Correct O(N_query × N_gallery) evaluation for STNet-Aligned mode.

    Similarity spaces:
      Text-only      : text_embeds       vs photo_embeds       (frozen tower, no attention)
      Sketch-only    : sketch_embeds     vs photo_embeds       (LoRA-on sketch, no attention)
      Late Fusion    : 0.5*sim_text + 0.5*sim_sketch
      STNet Aligned  : composite_query_i vs attend(sketch_i, patches_j) for ALL j
                       ← matches training loss target, O(N×M) computation
      Diagnostic     : composite_query   vs photo_embeds (misaligned baseline)
    """
    model.eval()

    # ----------------------------------------------------------------
    # Phase 1: Extract all embeddings in a single pass
    # ----------------------------------------------------------------
    sketch_embeds_list    = []
    text_embeds_list      = []
    composite_embeds_list = []
    photo_embeds_list     = []
    gallery_patches_list  = []   # (B, 49, D) per batch — kept on CPU

    print("[Eval] Phase 1: Extracting embeddings and gallery patch tokens...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Feature Extraction"):
            sketch = batch['sketch'].to(device)
            captions = batch['caption']
            photo = batch['photo'].to(device)

            # Sketch: LoRA-adapted (enable → encode → disable)
            e_sketch = model.encode_sketch(sketch)
            # Text: frozen tower + text adapter
            e_text = model.encode_text(captions)
            # Photo (raw, frozen base): no LoRA
            e_photo = model.encode_photo(photo)
            # Gallery patch tokens: (B, 49, D), frozen base
            patches = model.encode_photo_patches(photo)   # (B, 49, D)
            # Composite query
            raw_composite = torch.cat([e_sketch, e_text], dim=-1)
            e_composite = F.normalize(model.composite_fusion(raw_composite), dim=-1)

            sketch_embeds_list.append(e_sketch.cpu())
            text_embeds_list.append(e_text.cpu())
            composite_embeds_list.append(e_composite.cpu())
            photo_embeds_list.append(e_photo.cpu())
            gallery_patches_list.append(patches.cpu())    # kept off GPU

    sketch_embeds    = torch.cat(sketch_embeds_list,    dim=0)   # (N, D)
    text_embeds      = torch.cat(text_embeds_list,      dim=0)   # (N, D)
    composite_embeds = torch.cat(composite_embeds_list, dim=0)   # (N, D)
    photo_embeds     = torch.cat(photo_embeds_list,     dim=0)   # (N, D)
    gallery_patches  = torch.cat(gallery_patches_list,  dim=0)   # (N, 49, D)

    N = sketch_embeds.shape[0]
    print(f"[Eval] Gallery size: {N} | Patch tokens stored: {gallery_patches.shape}")

    # ----------------------------------------------------------------
    # Phase 2: Non-attention similarity matrices (fast numpy matmul)
    # ----------------------------------------------------------------
    sketch_np    = sketch_embeds.numpy()
    text_np      = text_embeds.numpy()
    composite_np = composite_embeds.numpy()
    photo_np     = photo_embeds.numpy()

    sim_text    = np.dot(text_np,   photo_np.T)      # (N, N)
    sim_sketch  = np.dot(sketch_np, photo_np.T)      # (N, N)

    # Diagnostic: composite vs raw photo (misaligned — should be worse than STNet)
    sim_composite_misaligned = np.dot(composite_np, photo_np.T)   # (N, N)

    # ----------------------------------------------------------------
    # Phase 3: O(N_query × N_gallery) STNet-Aligned similarity matrix
    #
    # For each query i:
    #   For each gallery photo j (in chunks of ATTENTION_CHUNK):
    #     attended_ij = attention_pooling(sketch_i, patches_j)   # (1, D)
    #     sim[i, j]   = composite_i · F.normalize(attended_ij)
    #
    # This is the ONLY correct eval for sketch-guided attention pooling:
    # the attended photo representation is query-specific.
    # ----------------------------------------------------------------
    print("[Eval] Phase 3: O(N×N) STNet-Aligned similarity computation...")
    sim_stnet = torch.zeros(N, N, dtype=torch.float32)

    with torch.no_grad():
        for i in tqdm(range(N), desc="Query-attended Retrieval"):
            sketch_q    = sketch_embeds[i:i+1].to(device)      # (1, D)
            text_q      = text_embeds[i:i+1].to(device)        # (1, D)
            composite_q = composite_embeds[i:i+1].to(device)   # (1, D)

            # Process gallery in chunks to respect GPU memory
            row_chunks = []
            for j_start in range(0, N, ATTENTION_CHUNK):
                j_end = min(j_start + ATTENTION_CHUNK, N)
                chunk_size = j_end - j_start

                # patches_chunk: (chunk, 49, D) → GPU
                patches_chunk = gallery_patches[j_start:j_end].to(device)

                # Expand queries to match chunk dimension
                sketch_expanded = sketch_q.expand(chunk_size, -1)   # (chunk, D)
                text_expanded   = text_q.expand(chunk_size, -1)     # (chunk, D)

                # TASKformer attention over each gallery photo's patch tokens
                attended_chunk, _ = model.attention_pooling(
                    patches_chunk, sketch_expanded, text_expanded
                )                                                     # (chunk, D)
                attended_chunk = F.normalize(attended_chunk, dim=-1)  # (chunk, D)

                # Similarity: (1, D) × (D, chunk) → (1, chunk)
                sims = torch.mm(composite_q, attended_chunk.t()).squeeze(0)  # (chunk,)
                row_chunks.append(sims.cpu())

            sim_stnet[i] = torch.cat(row_chunks)

    sim_stnet_np = sim_stnet.numpy()

    # ----------------------------------------------------------------
    # Phase 4: Compute metrics for all modes
    # ----------------------------------------------------------------
    results = {}

    # 1. Text-only
    r_text   = compute_recall_at_k(sim_text)
    ndcg_text = compute_ndcg_at_k(sim_text)
    results['Text-Only'] = {**r_text, 'NDCG@10': ndcg_text}

    # 2. Sketch-only
    r_sketch   = compute_recall_at_k(sim_sketch)
    ndcg_sketch = compute_ndcg_at_k(sim_sketch)
    results['Sketch-Only'] = {**r_sketch, 'NDCG@10': ndcg_sketch}

    # 3. Late Fusion baseline
    sim_late = 0.5 * sim_sketch + 0.5 * sim_text
    r_late   = compute_recall_at_k(sim_late)
    ndcg_late = compute_ndcg_at_k(sim_late)
    results['Composite (Late Fusion)'] = {**r_late, 'NDCG@10': ndcg_late}

    # 4. STNet Aligned — O(N×M) correct eval (matches training loss target)
    r_stnet   = compute_recall_at_k(sim_stnet_np)
    ndcg_stnet = compute_ndcg_at_k(sim_stnet_np)
    results['Composite (STNet Aligned)'] = {**r_stnet, 'NDCG@10': ndcg_stnet}

    # 5. Diagnostic: misaligned composite vs raw photo
    r_misaligned = compute_recall_at_k(sim_composite_misaligned)
    results['[Diagnostic] Composite vs Raw Photo'] = {**r_misaligned, 'NDCG@10': 0.0}

    # Print alignment gap — STNet should beat misaligned if attention is working
    gap = r_stnet.get('R@1', 0) - r_misaligned.get('R@1', 0)
    print(
        f"\n[Eval Diagnostic] STNet-Aligned R@1: {r_stnet.get('R@1', 0):.2f}% | "
        f"Misaligned R@1: {r_misaligned.get('R@1', 0):.2f}% | "
        f"Alignment Gap: {gap:+.2f}%"
    )
    if gap <= 0:
        print(
            "[Warning] STNet-Aligned R@1 ≤ Misaligned R@1.\n"
            "  Possible causes:\n"
            "  - Patch token diversity issue (PatchTokenExtractor hook not firing)\n"
            "  - LoRA not adapting edge filters (check Tier 2 Conv2d attachment)\n"
            "  - MoCo queue not yet warm (train more epochs)"
        )

    return results


def main():
    global ATTENTION_CHUNK
    
    parser = argparse.ArgumentParser(description="Evaluate T+SBIR Model on FS-COCO Test Split")
    parser.add_argument("--data_dir",    type=str, default="fscoco")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--checkpoint",  type=str, default=None)
    parser.add_argument(
        "--attn_chunk", type=int, default=ATTENTION_CHUNK,
        help="Gallery chunk size for Phase 3 attention (reduce if OOM on GPU)"
    )
    parser.add_argument(
        "--use_hnsw", action="store_true",
        help=(
            "Load a pre-built HNSW index and evaluate fast-path retrieval latency. "
            "Build the index first with: python -m src.search.gallery_indexer, then "
            "python -m src.search.hnsw_index"
        )
    )
    parser.add_argument(
        "--hnsw_path", type=str, default="gallery_index/hnsw.usearch",
        help="Path to pre-built HNSW index (used with --use_hnsw)"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Eval] Device: {device}")

    test_dataset = FSCOCODataset(args.data_dir, split='test')
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"[Eval] FS-COCO Test Split: {len(test_dataset)} samples")

    model = TSBIRCompositeModel(device=device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[Eval] Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state = checkpoint.get('state_dict', checkpoint.get('full_state_dict', checkpoint))
        missing, unexpected = model.load_state_dict(state, strict=False)
        non_backbone_missing = [k for k in missing if 'lora_' in k or
                                any(t in k for t in ('attention_pooling', 'composite_fusion',
                                                      'text_adapter', 'sketch_decoder', 'moco'))]
        if non_backbone_missing:
            print(f"[Eval] Missing non-backbone keys: {non_backbone_missing}")

    model.to(device)

    # Override attention chunk if specified
    ATTENTION_CHUNK = args.attn_chunk

    # ----------------------------------------------------------------
    # HNSW fast-path evaluation (Phase 4)
    # ----------------------------------------------------------------
    if args.use_hnsw:
        import time, numpy as _np
        from src.search.hnsw_index import HNSWSearchIndex

        print(f"\n[Eval] HNSW Fast-Path Evaluation")
        index = HNSWSearchIndex.load(args.hnsw_path)
        print(f"[Eval] Loaded: {index}")

        # Extract all query composite embeddings
        model.eval()
        all_composites, all_labels = [], []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                sketch   = batch['sketch'].to(device)
                captions = batch['caption']
                e_sketch = model.encode_sketch(sketch)
                e_text   = model.encode_text(captions)
                raw      = torch.cat([e_sketch, e_text], dim=-1)
                e_comp   = F.normalize(model.composite_fusion(raw), dim=-1)
                all_composites.append(e_comp.cpu().numpy())
                all_labels.extend(range(i * args.batch_size,
                                        i * args.batch_size + sketch.shape[0]))

        queries_np = _np.vstack(all_composites).astype(_np.float32)   # (N, D)
        N_q = len(queries_np)

        # Benchmark: search all queries and measure latency
        t0 = time.perf_counter()
        indices, _ = index.search(queries_np, k=10)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Recall@1, @5, @10 using HNSW results
        r1 = sum(1 for i in range(N_q) if i in indices[i, :1])  / N_q * 100
        r5 = sum(1 for i in range(N_q) if i in indices[i, :5])  / N_q * 100
        r10= sum(1 for i in range(N_q) if i in indices[i, :10]) / N_q * 100

        print(f"\n{'=' * 60}")
        print(f"[HNSW] Recall@1:  {r1:.2f}%")
        print(f"[HNSW] Recall@5:  {r5:.2f}%")
        print(f"[HNSW] Recall@10: {r10:.2f}%")
        print(f"[HNSW] Total search latency ({N_q} queries): {elapsed_ms:.1f} ms")
        print(f"[HNSW] Per-query latency: {elapsed_ms/N_q*1000:.1f} µs")
        print(f"{'=' * 60}\n")
        return

    # ----------------------------------------------------------------
    # Standard O(N×N) eval
    # ----------------------------------------------------------------
    results = evaluate_retrieval(model, test_loader, device=device)

    print("\n" + "=" * 72)
    print(f"{'Evaluation Mode':<37} | {'R@1':>5} | {'R@5':>5} | {'R@10':>5} | {'NDCG@10':>7}")
    print("-" * 72)
    for mode, metrics in results.items():
        print(
            f"{mode:<37} | {metrics['R@1']:>5.2f} | {metrics['R@5']:>5.2f} | "
            f"{metrics['R@10']:>5.2f} | {metrics['NDCG@10']:>7.4f}"
        )
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()

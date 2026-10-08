import os
import sys
import json
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
from CSTBIR.models.stnet import STNet
from CSTBIR.clip import clip
from CSTBIR.evaluate_retrieval import compute_retrieval_metrics

def evaluate_split(
    model: STNet,
    split_name: str = "val",
    dataset_json: str = "CSTBIR/data/CSTBIR_dataset.json",
    images_dir: str = "CSTBIR/data/vg_images",
    sketches_dir: str = "CSTBIR/data/quickdraw_sketches",
    batch_size: int = 64,
    gallery_manifest: Optional[str] = None
) -> Dict[str, float]:
    device = model.device
    model.eval()
    
    print(f"\n" + "=" * 70)
    print(f"EVALUATING ON SPLIT: '{split_name.upper()}'")
    print("=" * 70)
    
    with open(dataset_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    queries = [x for x in data if x.get("split") == split_name]
    print(f"Total raw queries in '{split_name}': {len(queries)}")
    
    # 0. Strict Media Validation: Drop any query with missing/corrupt media (no zero-filling)
    valid_queries = []
    for q in queries:
        ip = os.path.join(images_dir, q["image"])
        sp = os.path.join(sketches_dir, q["sketch"])
        if (os.path.exists(ip) and os.path.getsize(ip) > 1000 and
            os.path.exists(sp) and os.path.getsize(sp) > 1000):
            valid_queries.append(q)
        else:
            print(f"[Eval Warning] Dropping query with invalid/missing media: img={q['image']}, sk={q['sketch']}")
            
    queries = valid_queries
    print(f"Total media-verified queries in '{split_name}': {len(queries)}")
    
    # 1. Unique Gallery Images (from manifest if provided, else unique query targets)
    if gallery_manifest and os.path.exists(gallery_manifest):
        print(f"Loading custom gallery manifest from {gallery_manifest}...")
        with open(gallery_manifest, "r", encoding="utf-8") as gf:
            gallery_image_names = json.load(gf)
    else:
        gallery_image_names = sorted(list(set(x["image"] for x in queries)))
        
    # Verify every gallery image exists with real data (>1000B)
    for name in gallery_image_names:
        p = os.path.join(images_dir, name)
        if not (os.path.exists(p) and os.path.getsize(p) > 1000):
            raise RuntimeError(f"Gallery image missing or corrupted: {p}! Evaluation cannot use zero tensors.")
            
    K = len(gallery_image_names)
    img_to_idx = {name: i for i, name in enumerate(gallery_image_names)}
    print(f"Total unique verified gallery images: {K}")
    
    # Filter queries whose ground truth image is in gallery
    queries = [q for q in queries if q["image"] in img_to_idx]
    gt_indices = [img_to_idx[q["image"]] for q in queries]
    Q = len(queries)
    print(f"Active evaluated queries: {Q}")
    
    # Preprocessors
    clip_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711))
    ])
    sketch_preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 2. Extract Gallery Image Tokens: (K, 197, 768) in batches
    print("\n--- Step 1: Pre-computing Gallery Image Tokens ---")
    t0 = time.time()
    gallery_tokens_list = []
    
    for i in range(0, K, batch_size):
        batch_names = gallery_image_names[i : i + batch_size]
        batch_imgs = []
        for name in batch_names:
            p = os.path.join(images_dir, name)
            with Image.open(p) as img:
                batch_imgs.append(clip_preprocess(img.convert("RGB")))
                
        img_tensors = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            tokens = model.extract_image_tokens(img_tensors) # (B, 197, 768)
        gallery_tokens_list.append(tokens)
        
    H_tilde_gallery = torch.cat(gallery_tokens_list, dim=0) # (K, 197, 768)
    print(f"Gallery tokens computed: {H_tilde_gallery.shape} in {time.time()-t0:.1f}s")
    
    # 3. Extract Query Text & Sketch Embeddings in batches
    print("\n--- Step 2: Pre-computing Query Embeddings ---")
    t0 = time.time()
    query_z_t_list = []
    query_h_S_list = []
    
    for i in range(0, Q, batch_size):
        batch_q = queries[i : i + batch_size]
        
        # Texts
        text_strs = [q["text"] for q in batch_q]
        text_toks = clip.tokenize(text_strs, truncate=True).to(device)
        
        # Sketches
        batch_sks = []
        for q in batch_q:
            sp = os.path.join(sketches_dir, q["sketch"])
            with Image.open(sp) as img:
                batch_sks.append(sketch_preprocess(img.convert("RGB")))
                
        sk_tensors = torch.stack(batch_sks).to(device)
        
        with torch.no_grad():
            h_T = model.encode_text(text_toks) # (B, 512)
            z_t = F.normalize(model.loss_ct_fn.txt_proj(h_T), p=2, dim=-1) # (B, 512)
            h_S = model.encode_sketch(sk_tensors) # (B, 768)
            
        query_z_t_list.append(z_t)
        query_h_S_list.append(h_S)
        
    all_z_t = torch.cat(query_z_t_list, dim=0) # (Q, 512)
    all_h_S = torch.cat(query_h_S_list, dim=0) # (Q, 768)
    print(f"Query embeddings computed: text={all_z_t.shape}, sketch={all_h_S.shape} in {time.time()-t0:.1f}s")
    
    # 4. Compute Query-to-Gallery Cosine Similarities & Rank
    print("\n--- Step 3: Computing Query-to-Gallery Similarity Matrix & Ranks ---")
    t0 = time.time()
    ranks = []
    
    # Compute in chunks of queries to conserve GPU memory
    q_chunk = 32
    for q_start in range(0, Q, q_chunk):
        q_end = min(q_start + q_chunk, Q)
        z_t_chunk = all_z_t[q_start:q_end] # (B_q, 512)
        h_S_chunk = all_h_S[q_start:q_end] # (B_q, 768)
        
        for idx_in_chunk in range(z_t_chunk.shape[0]):
            q_idx = q_start + idx_in_chunk
            sq = h_S_chunk[idx_in_chunk : idx_in_chunk + 1] # (1, 768)
            
            with torch.no_grad():
                # Cross-attention over all K gallery images with sketch sq
                # Process gallery in chunks if K is large
                z_i_chunks = []
                g_chunk = 256
                for g_start in range(0, K, g_chunk):
                    g_end = min(g_start + g_chunk, K)
                    sub_gallery = H_tilde_gallery[g_start:g_end] # (chunk, 197, 768)
                    sq_rep = sq.repeat(sub_gallery.shape[0], 1)
                    
                    h_I_avg, _, _ = model.cross_attention(sub_gallery, sq_rep)
                    sub_z_i = F.normalize(model.loss_ct_fn.img_proj(h_I_avg), p=2, dim=-1)
                    z_i_chunks.append(sub_z_i)
                    
                full_z_i = torch.cat(z_i_chunks, dim=0) # (K, 512)
                
                # Cosine similarity vector for query q: (K,)
                sims = (z_t_chunk[idx_in_chunk : idx_in_chunk + 1] @ full_z_i.T).squeeze(0) # (K,)
                sims_np = sims.cpu().numpy()
                
            # Descending ranking
            sorted_indices = np.argsort(-sims_np)
            gt_rank = int(np.where(sorted_indices == gt_indices[q_idx])[0][0]) + 1
            ranks.append(gt_rank)
            
        if (q_start + q_chunk) % 200 < q_chunk or q_end == Q:
            print(f"  Processed {min(q_end, Q)} / {Q} queries ({time.time()-t0:.1f}s)...")
            
    metrics = compute_retrieval_metrics(ranks)
    print(f"\nFinal Retrieval Results for {split_name.upper()}:")
    for k, v in metrics.items():
        if k != "total_queries":
            print(f"  {k:8s}: {v:6.2f}%" if "R@" in k else f"  {k:8s}: {v:6.2f}")
    return metrics

def run_table3_evaluation(model_path: str = None, split: str = "both"):
    print("=" * 75)
    print("TABLE 3 REPRODUCTION BENCHMARK: STNet on Test-1K & Test-5K")
    print("=" * 75)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = STNet(
        num_classes=258,
        device=device,
        pretrained_sketch=True,
        sketch_encoder_ckpt="CSTBIR/checkpoints/sketch_encoder_quickdraw_adapted.pt"
    ).to(device)
    if model_path and os.path.exists(model_path):
        print(f"Loading weights from {model_path}...")
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        print("Model checkpoint loaded.")
    else:
        print("Using base initialized STNet with pretrained CLIP.")
        
    metrics_1k = None
    metrics_5k = None
    
    if split in ["val", "both"]:
        # Evaluate Test-1K
        metrics_1k = evaluate_split(model, split_name="val")
    
    if split in ["test", "both"]:
        # Evaluate Test-5K (use official 5,000-image gallery if available)
        gallery_5k_manifest = "CSTBIR/data/test5k_gallery_5000.json"
        if os.path.exists(gallery_5k_manifest):
            print(f"\nUsing canonical 5,000-image gallery for Test-5K ({gallery_5k_manifest})...")
            metrics_5k = evaluate_split(model, split_name="test", gallery_manifest=gallery_5k_manifest)
        else:
            metrics_5k = evaluate_split(model, split_name="test")
    
    # Print Table 3 Comparison
    print("\n" + "=" * 80)
    print("TABLE 3 COMPARISON: Published AAAI 2024 Paper vs. Reimplementation")
    print("=" * 80)
    print(f"{'Split':<10} | {'Metric':<8} | {'Published Paper':>16} | {'Reimplemented':>16} | {'Delta':>10}")
    print("-" * 80)
    
    if metrics_1k is not None:
        paper_1k = {'R@10': 73.7, 'R@20': 80.6, 'R@50': 89.4, 'R@100': 93.5, 'MdR': 3.0}
        for m in ['R@10', 'R@20', 'R@50', 'R@100', 'MdR']:
            p_val = paper_1k[m]
            r_val = metrics_1k[m]
            delta = r_val - p_val
            print(f"{'Test-1K':<10} | {m:<8} | {p_val:16.1f} | {r_val:16.1f} | {delta:+10.1f}")
        
    if metrics_5k is not None:
        print("-" * 80)
        paper_5k = {'R@10': 38.7, 'R@20': 50.0, 'R@50': 64.6, 'R@100': 74.5, 'MdR': 20.5}
        for m in ['R@10', 'R@20', 'R@50', 'R@100', 'MdR']:
            p_val = paper_5k[m]
            r_val = metrics_5k[m]
            delta = r_val - p_val
            print(f"{'Test-5K':<10} | {m:<8} | {p_val:16.1f} | {r_val:16.1f} | {delta:+10.1f}")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs="?", default=None)
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    args = parser.parse_args()
    run_table3_evaluation(args.model_path, split=args.split)

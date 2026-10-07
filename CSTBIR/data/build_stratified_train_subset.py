"""
PART P-8: Stratified Diversity-Maximizing Training Subset Builder
=================================================================
Selects exactly 5,000 training queries stratified across:
  1. All 258 categories (100% coverage, 0 zero-query categories)
  2. Maximum image diversity (capping queries per image to 2-3)
  3. Downloads any missing Visual Genome images and QuickDraw sketches
  4. Verifies 100% decodability with PIL
  5. Saves to CSTBIR/data/train_subset_5k_stratified.json
"""

import os
import sys
import json
import time
import random
import requests
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import numpy as np

def download_vg_image(img_name: str, save_dir: str, session: requests.Session) -> bool:
    save_path = os.path.join(save_dir, img_name)
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        return True
        
    urls = [
        f"https://cs.stanford.edu/people/rak248/VG_100K/{img_name}",
        f"https://cs.stanford.edu/people/rak248/VG_100K_2/{img_name}"
    ]
    for url in urls:
        try:
            r = session.get(url, timeout=12)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(save_path, "wb") as f:
                    f.write(r.content)
                return True
        except Exception:
            continue
    return False

def download_quickdraw_sketch(sketch_name: str, save_dir: str, session: requests.Session) -> bool:
    save_path = os.path.join(save_dir, sketch_name)
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        return True
        
    base = os.path.splitext(sketch_name)[0]
    parts = base.rsplit("_", 1)
    if len(parts) != 2:
        return False
    cat, idx_str = parts[0], parts[1]
    try:
        idx = int(idx_str)
    except ValueError:
        return False
        
    offset = 80 + idx * 784
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{cat}.npy"
    headers = {"Range": f"bytes={offset}-{offset + 783}"}
    
    for attempt in range(3):
        try:
            r = session.get(url, headers=headers, timeout=12)
            if r.status_code in [200, 206] and len(r.content) == 784:
                arr = np.frombuffer(r.content, dtype=np.uint8).reshape(28, 28)
                img_arr = 255 - arr
                img = Image.fromarray(img_arr).resize((224, 224), Image.BILINEAR)
                img.save(save_path, quality=95)
                return True
        except Exception:
            time.sleep(0.05 * (attempt + 1))
    return False

def build_stratified_subset(seed: int = 42, target_queries: int = 5000):
    print("=" * 80)
    print("BUILDING DIVERSITY-STRATIFIED 5,000-QUERY TRAINING SUBSET")
    print("=" * 80)
    
    random.seed(seed)
    
    json_path = "CSTBIR/data/CSTBIR_dataset.json"
    classes_path = "CSTBIR/data/classes_258.json"
    vg_dir = "CSTBIR/data/vg_images"
    qd_dir = "CSTBIR/data/quickdraw_sketches"
    os.makedirs(vg_dir, exist_ok=True)
    os.makedirs(qd_dir, exist_ok=True)
    
    with open(classes_path, "r", encoding="utf-8") as f:
        classes_258 = json.load(f)
    print(f"Loaded {len(classes_258)} categories.")
    
    print(f"Loading queries from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    train_queries = [x for x in data if x.get("split") == "train"]
    print(f"Total training queries available: {len(train_queries)}")
    
    # Exclude images that belong to test or val splits to maintain zero data leakage
    eval_imgs = set(x["image"] for x in data if x.get("split") in ["val", "test"])
    train_queries_clean = [x for x in train_queries if x["image"] not in eval_imgs]
    print(f"Training queries strictly excluding val/test images: {len(train_queries_clean)}")
    
    # Group queries by category
    cat_to_queries = defaultdict(list)
    for q in train_queries_clean:
        cat_to_queries[q["label"]].append(q)
        
    # Check category counts
    small_cats = {c: len(cat_to_queries[c]) for c in classes_258 if len(cat_to_queries[c]) < 20}
    print(f"Categories with <20 queries: {len(small_cats)}")
    
    selected_queries = []
    selected_query_ids = set()
    img_counts = Counter()
    
    # 1. Take all available queries for rare categories (<20 queries)
    for cat in sorted(small_cats.keys()):
        for q in cat_to_queries[cat]:
            selected_queries.append(q)
            selected_query_ids.add(q.get("query_id", id(q)))
            img_counts[q["image"]] += 1
            
    remaining_target = target_queries - len(selected_queries)
    regular_cats = [c for c in classes_258 if c not in small_cats]
    print(f"Regular categories: {len(regular_cats)} | Remaining query budget: {remaining_target}")
    
    base_per_cat = remaining_target // len(regular_cats)
    rem = remaining_target % len(regular_cats)
    
    # Deterministic category order
    shuffled_regular = sorted(regular_cats)
    random.shuffle(shuffled_regular)
    
    for i, cat in enumerate(shuffled_regular):
        n_needed = base_per_cat + (1 if i < rem else 0)
        avail = [q for q in cat_to_queries[cat] if q.get("query_id", id(q)) not in selected_query_ids]
        random.shuffle(avail)
        
        # Group by image
        img_map = defaultdict(list)
        for q in avail:
            img_map[q["image"]].append(q)
            
        cat_selected = []
        # Sort images by current global query count to maximize image diversity
        sorted_imgs = sorted(img_map.keys(), key=lambda img: (img_counts[img], random.random()))
        
        # Cap to at most 2 queries per image
        cap = 2
        for img in sorted_imgs:
            for q in img_map[img][:cap]:
                cat_selected.append(q)
                selected_query_ids.add(q.get("query_id", id(q)))
                img_counts[img] += 1
                if len(cat_selected) == n_needed:
                    break
            if len(cat_selected) == n_needed:
                break
                
        # If still needed (very rare), allow cap up to 4
        if len(cat_selected) < n_needed:
            for img in sorted_imgs:
                for q in img_map[img][cap:4]:
                    if q.get("query_id", id(q)) not in selected_query_ids:
                        cat_selected.append(q)
                        selected_query_ids.add(q.get("query_id", id(q)))
                        img_counts[img] += 1
                        if len(cat_selected) == n_needed:
                            break
                if len(cat_selected) == n_needed:
                    break
                    
        selected_queries.extend(cat_selected)
        
    print(f"\nFinal Selected Queries: {len(selected_queries)}")
    unique_imgs = sorted(list(set(x["image"] for x in selected_queries)))
    unique_sks = sorted(list(set(x["sketch"] for x in selected_queries)))
    cat_cov = Counter(x["label"] for x in selected_queries)
    
    print(f"Unique Images in Subset: {len(unique_imgs)} (vs. 238 previously, +{len(unique_imgs)/238:.1f}x)")
    print(f"Unique Sketches in Subset: {len(unique_sks)}")
    print(f"Categories Covered: {len(cat_cov)} / 258 ({len(cat_cov)/258*100:.1f}%) (vs. 139 previously)")
    print(f"Zero-Query Categories: {sum(1 for c in classes_258 if cat_cov[c] == 0)}")
    print(f"Max Queries Per Image: {max(img_counts.values())} (vs. 98 previously)")
    print(f"Mean Queries Per Image: {len(selected_queries)/len(unique_imgs):.2f} (vs. 21.0 previously)")
    
    # -------------------------------------------------------------
    # 2. Download missing media with ThreadPoolExecutor
    # -------------------------------------------------------------
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
    session.mount("https://", adapter)
    
    # Missing images
    missing_imgs = [img for img in unique_imgs if not (os.path.exists(os.path.join(vg_dir, img)) and os.path.getsize(os.path.join(vg_dir, img)) > 1000)]
    print(f"\nImages needing download: {len(missing_imgs)} / {len(unique_imgs)}")
    
    if missing_imgs:
        t0 = time.time()
        dl_success = 0
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(download_vg_image, img, vg_dir, session): img for img in missing_imgs}
            for f in as_completed(futures):
                if f.result():
                    dl_success += 1
                if dl_success % 500 == 0 and dl_success > 0:
                    print(f"  Downloaded {dl_success} / {len(missing_imgs)} images ({time.time()-t0:.1f}s)...")
        print(f"Image download complete: {dl_success} / {len(missing_imgs)} in {time.time()-t0:.1f}s.")
        
    # Missing sketches
    missing_sks = [sk for sk in unique_sks if not (os.path.exists(os.path.join(qd_dir, sk)) and os.path.getsize(os.path.join(qd_dir, sk)) > 1000)]
    print(f"\nSketches needing download: {len(missing_sks)} / {len(unique_sks)}")
    
    if missing_sks:
        t0 = time.time()
        dl_success_sk = 0
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(download_quickdraw_sketch, sk, qd_dir, session): sk for sk in missing_sks}
            for f in as_completed(futures):
                if f.result():
                    dl_success_sk += 1
                if dl_success_sk % 1000 == 0 and dl_success_sk > 0:
                    print(f"  Downloaded {dl_success_sk} / {len(missing_sks)} sketches ({time.time()-t0:.1f}s)...")
        print(f"Sketch download complete: {dl_success_sk} / {len(missing_sks)} in {time.time()-t0:.1f}s.")
        
    # -------------------------------------------------------------
    # 3. Verify All Media with PIL
    # -------------------------------------------------------------
    print("\n--- Verifying All Media with PIL ---")
    bad_imgs = []
    for img in unique_imgs:
        p = os.path.join(vg_dir, img)
        if not os.path.exists(p) or os.path.getsize(p) <= 1000:
            bad_imgs.append(img)
            continue
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception:
            bad_imgs.append(img)
            
    bad_sks = []
    for sk in unique_sks:
        p = os.path.join(qd_dir, sk)
        if not os.path.exists(p) or os.path.getsize(p) <= 1000:
            bad_sks.append(sk)
            continue
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception:
            bad_sks.append(sk)
            
    print(f"Verified Images: {len(unique_imgs) - len(bad_imgs)} / {len(unique_imgs)} valid (Bad: {len(bad_imgs)})")
    print(f"Verified Sketches: {len(unique_sks) - len(bad_sks)} / {len(unique_sks)} valid (Bad: {len(bad_sks)})")
    
    # Filter out any query with bad media if any
    if bad_imgs or bad_sks:
        bad_img_set = set(bad_imgs)
        bad_sk_set = set(bad_sks)
        selected_queries = [
            q for q in selected_queries
            if q["image"] not in bad_img_set and q["sketch"] not in bad_sk_set
        ]
        print(f"Clean valid queries after verification: {len(selected_queries)}")
        
    # -------------------------------------------------------------
    # 4. Save to JSON
    # -------------------------------------------------------------
    out_path = "CSTBIR/data/train_subset_5k_stratified.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selected_queries, f, indent=2)
    print(f"\n[SUCCESS] Saved stratified subset ({len(selected_queries)} queries) to {out_path}.")
    
    return out_path

if __name__ == "__main__":
    build_stratified_subset()

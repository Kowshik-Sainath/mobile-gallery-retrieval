"""
PART P-7 (Step 3) / PART P-8: Canonical 5,000-Image Gallery Builder for Test-5K
==============================================================================
Pads the 2,214 query-referenced target images of Test-5K to exactly 5,000 gallery
images using real Visual Genome distractor images (strictly excluding any overlap
with the stratified training subset or validation set).
"""

import os
import sys
import json
import time
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

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

def build_5000_gallery(seed: int = 42, target_gallery_size: int = 5000):
    print("=" * 80)
    print("BUILDING OFFICIAL 5,000-IMAGE GALLERY FOR TEST-5K")
    print("=" * 80)
    
    random.seed(seed)
    
    dataset_json = "CSTBIR/data/CSTBIR_dataset.json"
    train_stratified_json = "CSTBIR/data/train_subset_5k_stratified.json"
    vg_dir = "CSTBIR/data/vg_images"
    os.makedirs(vg_dir, exist_ok=True)
    
    with open(dataset_json, "r", encoding="utf-8") as f:
        all_data = json.load(f)
        
    test_queries = [x for x in all_data if x.get("split") == "test"]
    test_target_imgs = sorted(list(set(x["image"] for x in test_queries)))
    print(f"Test-5K query-referenced target images: {len(test_target_imgs)}")
    
    val_queries = [x for x in all_data if x.get("split") == "val"]
    val_imgs = set(x["image"] for x in val_queries)
    
    # Load training images to strictly prevent leakage
    train_imgs = set()
    if os.path.exists(train_stratified_json):
        with open(train_stratified_json, "r", encoding="utf-8") as f:
            train_stratified = json.load(f)
        train_imgs = set(x["image"] for x in train_stratified)
        print(f"Loaded {len(train_imgs)} images from stratified training subset to exclude.")
        
    excluded_imgs = set(test_target_imgs) | val_imgs | train_imgs
    print(f"Total excluded images (Test targets + Val + Train): {len(excluded_imgs)}")
    
    # All train images from CSTBIR_dataset
    all_train_imgs = sorted(list(set(x["image"] for x in all_data if x.get("split") == "train")))
    distractor_pool = [img for img in all_train_imgs if img not in excluded_imgs]
    print(f"Candidate Visual Genome distractor images available: {len(distractor_pool)}")
    
    num_distractors_needed = target_gallery_size - len(test_target_imgs)
    print(f"Distractors needed to reach {target_gallery_size}: {num_distractors_needed}")
    
    # Build valid gallery iteratively
    valid_gallery = set()
    
    # 1. Target images must all be present
    for img in test_target_imgs:
        p = os.path.join(vg_dir, img)
        if os.path.exists(p) and os.path.getsize(p) > 1000:
            try:
                with Image.open(p) as im:
                    im.verify()
                valid_gallery.add(img)
            except Exception:
                pass
    print(f"Verified test target images present: {len(valid_gallery)} / {len(test_target_imgs)}")
    assert len(valid_gallery) == len(test_target_imgs), "Some test target images are missing or corrupt!"
    
    # 2. Check already downloaded distractors
    rng = random.Random(seed)
    shuffled_pool = distractor_pool[:]
    rng.shuffle(shuffled_pool)
    
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
    session.mount("https://", adapter)
    
    pool_idx = 0
    t0 = time.time()
    
    while len(valid_gallery) < target_gallery_size and pool_idx < len(shuffled_pool):
        needed = target_gallery_size - len(valid_gallery)
        candidate_batch = shuffled_pool[pool_idx : pool_idx + needed + 10]
        pool_idx += len(candidate_batch)
        
        # Download missing candidates in batch
        to_download = [
            img for img in candidate_batch
            if not (os.path.exists(os.path.join(vg_dir, img)) and os.path.getsize(os.path.join(vg_dir, img)) > 1000)
        ]
        
        if to_download:
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = {executor.submit(download_vg_image, img, vg_dir, session): img for img in to_download}
                for f in as_completed(futures):
                    f.result()
                    
        # Verify with PIL
        for img in candidate_batch:
            if len(valid_gallery) >= target_gallery_size:
                break
            p = os.path.join(vg_dir, img)
            if os.path.exists(p) and os.path.getsize(p) > 1000:
                try:
                    with Image.open(p) as im:
                        im.verify()
                    valid_gallery.add(img)
                except Exception:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
                        
        print(f"Current verified gallery size: {len(valid_gallery)} / {target_gallery_size} ({time.time()-t0:.1f}s)")
        
    full_5000_gallery = sorted(list(valid_gallery))
    assert len(full_5000_gallery) == target_gallery_size, f"Failed to acquire {target_gallery_size} valid images!"
    
    out_manifest = "CSTBIR/data/test5k_gallery_5000.json"
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(full_5000_gallery, f, indent=2)
    print(f"\n[SUCCESS] Canonical 5,000-image gallery verified 100% with PIL and saved to {out_manifest}.")
    return out_manifest

if __name__ == "__main__":
    build_5000_gallery()

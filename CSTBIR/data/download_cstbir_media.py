import os
import io
import sys
import json
import time
import requests
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_vg_image(img_name: str, save_dir: str, session: requests.Session) -> bool:
    save_path = os.path.join(save_dir, img_name)
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        return True
        
    urls = [
        f"https://cs.stanford.edu/people/rak248/VG_100K_2/{img_name}",
        f"https://cs.stanford.edu/people/rak248/VG_100K/{img_name}"
    ]
    for url in urls:
        try:
            r = session.get(url, timeout=10)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(save_path, "wb") as f:
                    f.write(r.content)
                return True
        except Exception:
            continue
    return False

def download_quickdraw_sketch(sketch_name: str, save_dir: str, session: requests.Session) -> bool:
    save_path = os.path.join(save_dir, sketch_name)
    if os.path.exists(save_path) and os.path.getsize(save_path) > 100:
        return True
        
    # Format: {category}_{idx}.jpg
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
    
    try:
        r = session.get(url, headers=headers, timeout=10)
        if r.status_code in [200, 206] and len(r.content) == 784:
            arr = np.frombuffer(r.content, dtype=np.uint8).reshape(28, 28)
            # Invert: white background (255), black strokes (0 to 255)
            img_arr = 255 - arr
            img = Image.fromarray(img_arr).resize((224, 224), Image.BILINEAR)
            img.save(save_path, quality=95)
            return True
    except Exception:
        pass
    return False

def main():
    print("=" * 75)
    print("Downloading Official Test-1K Media (Visual Genome + QuickDraw)")
    print("=" * 75)
    
    vg_dir = "CSTBIR/data/vg_images"
    qd_dir = "CSTBIR/data/quickdraw_sketches"
    os.makedirs(vg_dir, exist_ok=True)
    os.makedirs(qd_dir, exist_ok=True)
    
    with open("CSTBIR/data/CSTBIR_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Get all val (Test-1K) queries
    val_queries = [x for x in data if x['split'] == 'val']
    val_images = sorted(list(set(x['image'] for x in val_queries)))
    val_sketches = sorted(list(set(x['sketch'] for x in val_queries)))
    
    print(f"Test-1K requires {len(val_images)} Visual Genome images and {len(val_sketches)} QuickDraw sketches.")
    
    # 1. Download Images in parallel
    print("\n--- 1. Downloading Visual Genome Gallery Images (1000 images) ---")
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
    session.mount("https://", adapter)
    
    t0 = time.time()
    success_imgs = 0
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(download_vg_image, img, vg_dir, session): img for img in val_images}
        for future in as_completed(futures):
            if future.result():
                success_imgs += 1
            if success_imgs % 100 == 0:
                print(f"  Downloaded {success_imgs} / {len(val_images)} images ({time.time()-t0:.1f}s)...")
                
    print(f"Completed Visual Genome image downloads: {success_imgs} / {len(val_images)} successful in {time.time()-t0:.1f}s.")
    
    # 2. Download Sketches in parallel via Range requests
    print("\n--- 2. Downloading QuickDraw Query Sketches (999 sketches) ---")
    t0 = time.time()
    success_sks = 0
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(download_quickdraw_sketch, sk, qd_dir, session): sk for sk in val_sketches}
        for future in as_completed(futures):
            if future.result():
                success_sks += 1
            if success_sks % 100 == 0:
                print(f"  Downloaded {success_sks} / {len(val_sketches)} sketches ({time.time()-t0:.1f}s)...")
                
    print(f"Completed QuickDraw sketch downloads: {success_sks} / {len(val_sketches)} successful in {time.time()-t0:.1f}s.")
    
    # Verify disk counts
    disk_imgs = len(os.listdir(vg_dir))
    disk_sks = len(os.listdir(qd_dir))
    print(f"\nFinal disk counts in CSTBIR/data:")
    print(f"  vg_images:           {disk_imgs} images")
    print(f"  quickdraw_sketches:  {disk_sks} sketches")
    assert success_imgs >= 990, f"Too many image download failures: {success_imgs} < 990"
    assert success_sks >= 990, f"Too many sketch download failures: {success_sks} < 990"
    print("\n[DATA PREPARATION: PASS] All Test-1K gallery images and sketches downloaded and verified.")

if __name__ == "__main__":
    main()

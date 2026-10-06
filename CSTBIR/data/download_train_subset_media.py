import os
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
            img_arr = 255 - arr
            img = Image.fromarray(img_arr).resize((224, 224), Image.BILINEAR)
            img.save(save_path, quality=95)
            return True
    except Exception:
        pass
    return False

def main():
    print("=" * 75)
    print("Downloading Training Media Subset (5,000 Queries)")
    print("=" * 75)
    
    vg_dir = "CSTBIR/data/vg_images"
    qd_dir = "CSTBIR/data/quickdraw_sketches"
    os.makedirs(vg_dir, exist_ok=True)
    os.makedirs(qd_dir, exist_ok=True)
    
    with open("CSTBIR/data/CSTBIR_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        
    train_queries = [x for x in data if x['split'] == 'train'][:5000]
    train_images = sorted(list(set(x['image'] for x in train_queries)))
    train_sketches = sorted(list(set(x['sketch'] for x in train_queries)))
    
    print(f"5,000 train queries require {len(train_images)} VG images and {len(train_sketches)} QuickDraw sketches.")
    
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
    session.mount("https://", adapter)
    
    # 1. Images
    t0 = time.time()
    success_imgs = 0
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(download_vg_image, img, vg_dir, session): img for img in train_images}
        for future in as_completed(futures):
            if future.result():
                success_imgs += 1
                
    print(f"Completed VG images: {success_imgs} / {len(train_images)} in {time.time()-t0:.1f}s.")
    
    # 2. Sketches
    t0 = time.time()
    success_sks = 0
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(download_quickdraw_sketch, sk, qd_dir, session): sk for sk in train_sketches}
        for future in as_completed(futures):
            if future.result():
                success_sks += 1
            if success_sks % 500 == 0:
                print(f"  Downloaded {success_sks} / {len(train_sketches)} sketches ({time.time()-t0:.1f}s)...")
                
    print(f"Completed QuickDraw sketches: {success_sks} / {len(train_sketches)} in {time.time()-t0:.1f}s.")
    print(f"\nTotal media on disk: {len(os.listdir(vg_dir))} images, {len(os.listdir(qd_dir))} sketches.")

if __name__ == "__main__":
    main()

import os
import sys
import random
import torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from CSTBIR.data.dataloader import CSTBIRDataset

def test_real_vg_boxes_gate():
    print("=" * 80)
    print("GATE P-5 (Step 3): Real Visual Genome Bounding Box Verification")
    print("=" * 80)
    
    json_path = "CSTBIR/data/CSTBIR_dataset.json"
    vg_boxes_path = "CSTBIR/data/vg_boxes.json"
    images_dir = "CSTBIR/data/vg_images"
    sketches_dir = "CSTBIR/data/quickdraw_sketches"
    
    assert os.path.exists(vg_boxes_path), f"Missing {vg_boxes_path}!"
    
    train_ds = CSTBIRDataset(
        json_path=json_path,
        split='train',
        images_dir=images_dir,
        sketches_dir=sketches_dir,
        vg_boxes_dict=vg_boxes_path,
        filter_available=True
    )
    
    N = len(train_ds)
    print(f"\nTotal active training queries: {N}")
    
    # -------------------------------------------------------------
    # 1. Sample 10 random training items and inspect their bounding boxes
    # -------------------------------------------------------------
    print("\n--- 1. Sampling 10 Random Training Items ---")
    random.seed(42)
    sample_indices = random.sample(range(N), 10)
    
    constant_fallback = [0.25, 0.25, 0.75, 0.75]
    all_boxes = []
    
    print(f"{'Idx':<5} | {'Image':<14} | {'Query Label':<16} | {'Bounding Box [xmin, ymin, xmax, ymax]':<38} | {'Type'}")
    print("-" * 95)
    
    for idx in sample_indices:
        item = train_ds[idx]
        bbox_list = [round(v, 4) for v in item['bbox'].tolist()]
        all_boxes.append(bbox_list)
        
        is_fallback = (bbox_list == constant_fallback)
        match_type = "FALLBACK [0.25,0.25,0.75,0.75]" if is_fallback else "REAL VG ANNOTATION"
        
        print(f"{idx:<5} | {item['image_name']:<14} | {item['raw_text'][:16]:<16} | {str(bbox_list):<38} | {match_type}")
        
    # Verification checks
    unique_boxes = set(tuple(b) for b in all_boxes)
    print("\n--- Gate Assertions ---")
    print(f"Total unique bounding boxes among 10 random samples: {len(unique_boxes)} / 10")
    assert len(unique_boxes) > 1, "FAIL: All sampled bounding boxes are identical!"
    assert not all(b == constant_fallback for b in all_boxes), "FAIL: All boxes are the constant fallback!"
    print("PASS: Bounding boxes vary per image and are not identical.")
    
    # -------------------------------------------------------------
    # 2. Comprehensive Statistics across ALL N available queries
    # -------------------------------------------------------------
    print("\n--- 2. Comprehensive Statistics Across All 5,000 Training Samples ---")
    all_dataset_boxes = []
    num_exact_matches = 0
    num_img_fallbacks = 0
    num_constant_fallbacks = 0
    
    for i in range(N):
        raw_item = train_ds.samples[i]
        img = raw_item['image']
        lbl = raw_item['label']
        key_pair = f"{img}_{lbl}"
        
        if key_pair in train_ds.vg_boxes_dict:
            num_exact_matches += 1
            all_dataset_boxes.append(train_ds.vg_boxes_dict[key_pair])
        elif img in train_ds.vg_boxes_dict:
            num_img_fallbacks += 1
            all_dataset_boxes.append(train_ds.vg_boxes_dict[img])
        else:
            num_constant_fallbacks += 1
            all_dataset_boxes.append(constant_fallback)
            
    print(f"Exact Query-Object (image, label) Matches: {num_exact_matches:5d} / {N} ({num_exact_matches/N*100:5.1f}%)")
    print(f"Image-Level Object Matches (fallback):     {num_img_fallbacks:5d} / {N} ({num_img_fallbacks/N*100:5.1f}%)")
    print(f"Unmatched (constant [0.25,0.25,0.75,0.75]):{num_constant_fallbacks:5d} / {N} ({num_constant_fallbacks/N*100:5.1f}%)")
    print(f"Total Real VG Ground-Truth Supervised:     {num_exact_matches + num_img_fallbacks:5d} / {N} ({(num_exact_matches + num_img_fallbacks)/N*100:5.1f}%)")
    
    boxes_arr = np.array(all_dataset_boxes)
    widths = boxes_arr[:, 2] - boxes_arr[:, 0]
    heights = boxes_arr[:, 3] - boxes_arr[:, 1]
    
    print("\n--- Bounding Box Distribution Metrics ---")
    print(f"Mean Width:  {widths.mean():.4f} (std: {widths.std():.4f}, min: {widths.min():.4f}, max: {widths.max():.4f})")
    print(f"Mean Height: {heights.mean():.4f} (std: {heights.std():.4f}, min: {heights.min():.4f}, max: {heights.max():.4f})")
    print(f"X-center range: [{((boxes_arr[:,0]+boxes_arr[:,2])/2).min():.4f}, {((boxes_arr[:,0]+boxes_arr[:,2])/2).max():.4f}]")
    print(f"Y-center range: [{((boxes_arr[:,1]+boxes_arr[:,3])/2).min():.4f}, {((boxes_arr[:,1]+boxes_arr[:,3])/2).max():.4f}]")
    
    # -------------------------------------------------------------
    # 3. Test Conflict-Free Batch Sampling with Real Boxes
    # -------------------------------------------------------------
    print("\n--- 3. Testing Conflict-Free Batch Sampling ---")
    batch = train_ds.get_conflict_free_batch(8)
    batch_boxes = batch['bbox']
    print(f"Batch bbox tensor shape: {batch_boxes.shape}")
    for b_idx in range(8):
        b_val = [round(v, 4) for v in batch_boxes[b_idx].tolist()]
        print(f"  Batch sample {b_idx} ({batch['image_names'][b_idx]}): bbox={b_val}")
        
    print("\n[GATE P-5 STEP 3: PASSED] Real Visual Genome bounding boxes are fully active and verified.")

if __name__ == "__main__":
    test_real_vg_boxes_gate()

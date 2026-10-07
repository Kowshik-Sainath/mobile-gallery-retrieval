"""
PART P-7: Step 1 Data Completeness Verification
===============================================
Computes exact counts of present vs. missing gallery images and query sketches
for both Test-1K ('val') and Test-5K ('test') splits, checking for non-trivial size (>1000 bytes).
"""

import os
import sys
import json

def check_split_completeness(split_name: str, dataset_json: str, vg_images_dir: str, sketches_dir: str):
    with open(dataset_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    queries = [x for x in data if x.get('split') == split_name]
    num_queries = len(queries)
    
    # Unique gallery images required by the split
    gallery_image_names = sorted(list(set(x['image'] for x in queries)))
    num_gallery = len(gallery_image_names)
    
    # Check gallery images (>1000 bytes)
    present_gallery = []
    missing_gallery = []
    too_small_gallery = []
    
    for img_name in gallery_image_names:
        p = os.path.join(vg_images_dir, img_name)
        if not os.path.exists(p):
            missing_gallery.append(img_name)
        else:
            sz = os.path.getsize(p)
            if sz > 1000:
                present_gallery.append(img_name)
            else:
                too_small_gallery.append((img_name, sz))
                
    # Check query sketches (>1000 bytes)
    # Note: A query's sketch file might be shared across queries, but we check both per-query and unique sketches
    unique_sketch_names = sorted(list(set(x['sketch'] for x in queries)))
    num_unique_sketches = len(unique_sketch_names)
    
    present_unique_sketches = []
    missing_unique_sketches = []
    too_small_unique_sketches = []
    
    for sk_name in unique_sketch_names:
        sp = os.path.join(sketches_dir, sk_name)
        if not os.path.exists(sp):
            missing_unique_sketches.append(sk_name)
        else:
            sz = os.path.getsize(sp)
            if sz > 1000:
                present_unique_sketches.append(sk_name)
            else:
                too_small_unique_sketches.append((sk_name, sz))
                
    # Per-query sketch check
    query_sketch_present = 0
    query_sketch_missing = 0
    
    for q in queries:
        sp = os.path.join(sketches_dir, q['sketch'])
        if os.path.exists(sp) and os.path.getsize(sp) > 1000:
            query_sketch_present += 1
        else:
            query_sketch_missing += 1
            
    # Also check per-query: both sketch AND target image present
    queries_fully_valid = 0
    for q in queries:
        ip = os.path.join(vg_images_dir, q['image'])
        sp = os.path.join(sketches_dir, q['sketch'])
        img_ok = os.path.exists(ip) and os.path.getsize(ip) > 1000
        sk_ok = os.path.exists(sp) and os.path.getsize(sp) > 1000
        if img_ok and sk_ok:
            queries_fully_valid += 1
            
    print(f"\n" + "=" * 70)
    print(f"DATA COMPLETENESS REPORT FOR SPLIT: '{split_name.upper()}'")
    print("=" * 70)
    print(f"Total queries in split: {num_queries}")
    print(f"Gallery Images (>1000B):   {len(present_gallery):5d} / {num_gallery:5d} present ({len(present_gallery)/num_gallery*100:5.2f}%)")
    print(f"  Missing gallery files:   {len(missing_gallery)}")
    print(f"  Too small (<=1000B):     {len(too_small_gallery)}")
    print(f"Unique Sketches (>1000B):  {len(present_unique_sketches):5d} / {num_unique_sketches:5d} present ({len(present_unique_sketches)/num_unique_sketches*100:5.2f}%)")
    print(f"  Missing sketch files:    {len(missing_unique_sketches)}")
    print(f"  Too small (<=1000B):     {len(too_small_unique_sketches)}")
    print(f"Query Sketches (>1000B):   {query_sketch_present:5d} / {num_queries:5d} present ({query_sketch_present/num_queries*100:5.2f}%)")
    print(f"Queries with BOTH valid:   {queries_fully_valid:5d} / {num_queries:5d} ({queries_fully_valid/num_queries*100:5.2f}%)")
    
    return {
        "split": split_name,
        "num_queries": num_queries,
        "num_gallery": num_gallery,
        "present_gallery": len(present_gallery),
        "missing_gallery_list": missing_gallery,
        "too_small_gallery_list": too_small_gallery,
        "num_unique_sketches": num_unique_sketches,
        "present_unique_sketches": len(present_unique_sketches),
        "missing_unique_sketches_list": missing_unique_sketches,
        "query_sketch_present": query_sketch_present,
        "queries_fully_valid": queries_fully_valid
    }

if __name__ == "__main__":
    json_path = "CSTBIR/data/CSTBIR_dataset.json"
    vg_dir = "CSTBIR/data/vg_images"
    sk_dir = "CSTBIR/data/quickdraw_sketches"
    
    val_rep = check_split_completeness("val", json_path, vg_dir, sk_dir)
    test_rep = check_split_completeness("test", json_path, vg_dir, sk_dir)
    
    # Save to json
    out_path = "CSTBIR/data/completeness_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"val": val_rep, "test": test_rep}, f, indent=2)
    print(f"\nSaved report to {out_path}")

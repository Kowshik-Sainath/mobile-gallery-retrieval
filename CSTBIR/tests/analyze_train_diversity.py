"""
PART P-8: Step 1 & 2 Image Diversity Analysis
=============================================
Analyzes the active 5,000-query training subset:
  - Unique image count
  - Image ID clustering
  - Per-category query & image distribution across all 258 classes
  - Zero-representation categories count
  - Queries-per-image distribution
  - Confirmation of contiguous slicing [:5000]
"""

import os
import sys
import json
from collections import Counter

def analyze_diversity():
    print("=" * 80)
    print("PART P-8: Training Image & Category Diversity Analysis")
    print("=" * 80)
    
    json_path = "CSTBIR/data/CSTBIR_dataset.json"
    classes_path = "CSTBIR/data/classes_258.json"
    images_dir = "CSTBIR/data/vg_images"
    sketches_dir = "CSTBIR/data/quickdraw_sketches"
    
    with open(classes_path, "r", encoding="utf-8") as f:
        classes_258 = json.load(f)
    print(f"Total official categories: {len(classes_258)}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
        
    all_train = [x for x in all_data if x.get("split") == "train"]
    print(f"Total raw training queries in dataset: {len(all_train)}")
    
    # 1. Contiguous slice check (from download_train_subset_media.py)
    slice_5000 = all_train[:5000]
    slice_imgs = sorted(list(set(x["image"] for x in slice_5000)))
    print(f"\n--- Contiguous Slice [:5000] Check ---")
    print(f"Total queries in [:5000]: {len(slice_5000)}")
    print(f"Total unique images in [:5000]: {len(slice_imgs)}")
    
    # Check currently active samples in CSTBIRDataset
    local_imgs = set(os.listdir(images_dir))
    local_sks = set(os.listdir(sketches_dir))
    
    active_samples = [
        x for x in all_train
        if x["image"] in local_imgs and x["sketch"] in local_sks
    ]
    print(f"\n--- Currently Active Local Training Samples in Dataloader ---")
    print(f"Total local-media verified training samples: {len(active_samples)}")
    
    active_5k = active_samples[:5000]
    unique_active_imgs = sorted(list(set(x["image"] for x in active_5k)))
    print(f"Unique images in active 5,000 queries: {len(unique_active_imgs)}")
    
    # 2. Image ID clustering analysis
    img_ids = []
    for img_name in unique_active_imgs:
        stem = os.path.splitext(img_name)[0]
        try:
            img_ids.append(int(stem))
        except ValueError:
            pass
            
    img_ids.sort()
    print(f"\n--- Image ID Clustering ---")
    if img_ids:
        print(f"Image ID range: min={img_ids[0]}, max={img_ids[-1]}, span={img_ids[-1] - img_ids[0] + 1}")
        diffs = [img_ids[i+1] - img_ids[i] for i in range(len(img_ids)-1)]
        print(f"Median ID gap: {sorted(diffs)[len(diffs)//2]}, Mean gap: {sum(diffs)/len(diffs):.1f}")
        consecutive = sum(1 for d in diffs if d == 1)
        print(f"Directly consecutive IDs (gap==1): {consecutive} / {len(diffs)}")
        
    # 3. Queries-per-image distribution
    img_query_counts = Counter(x["image"] for x in active_5k)
    counts = list(img_query_counts.values())
    print(f"\n--- Queries Per Image Distribution ---")
    print(f"Min queries/image:  {min(counts)}")
    print(f"Max queries/image:  {max(counts)}")
    print(f"Mean queries/image: {sum(counts)/len(counts):.1f}")
    print(f"Top 5 most queried images: {img_query_counts.most_common(5)}")
    
    # 4. Category distribution across 258 classes
    class_query_counts = Counter(x["label"] for x in active_5k)
    class_img_sets = {}
    for x in active_5k:
        class_img_sets.setdefault(x["label"], set()).add(x["image"])
        
    zero_query_classes = [c for c in classes_258 if c not in class_query_counts]
    covered_classes = [c for c in classes_258 if c in class_query_counts]
    
    print(f"\n--- Category Distribution Across 258 Classes ---")
    print(f"Categories with >=1 query:  {len(covered_classes)} / 258 ({len(covered_classes)/258*100:.1f}%)")
    print(f"Categories with 0 queries:   {len(zero_query_classes)} / 258 ({len(zero_query_classes)/258*100:.1f}%)")
    
    sorted_covered = sorted(covered_classes, key=lambda c: class_query_counts[c], reverse=True)
    print(f"\nTop 10 most frequent categories in active training subset:")
    for c in sorted_covered[:10]:
        print(f"  {c:<18}: {class_query_counts[c]:4d} queries ({len(class_img_sets[c])} images)")
        
    print(f"\nBottom 10 least frequent categories (with >=1 query):")
    for c in sorted_covered[-10:]:
        print(f"  {c:<18}: {class_query_counts[c]:4d} queries ({len(class_img_sets[c])} images)")
        
    # 5. Overlap with Test-1K (val) and Test-5K (test) categories
    with open("CSTBIR/data/CSTBIR_dataset.json", "r", encoding="utf-8") as f:
        val_queries = [x for x in all_data if x.get("split") == "val"]
        test_queries = [x for x in all_data if x.get("split") == "test"]
        
    val_classes = set(x["label"] for x in val_queries)
    test_classes = set(x["label"] for x in test_queries)
    
    unseen_val_classes = val_classes - set(covered_classes)
    unseen_test_classes = test_classes - set(covered_classes)
    
    print(f"\n--- Evaluation Category Coverage Impact ---")
    print(f"Test-1K unique categories:  {len(val_classes)}")
    print(f"Test-1K categories UNSEEN in training: {len(unseen_val_classes)} / {len(val_classes)} ({len(unseen_val_classes)/len(val_classes)*100:.1f}%)")
    print(f"Test-5K unique categories:  {len(test_classes)}")
    print(f"Test-5K categories UNSEEN in training: {len(unseen_test_classes)} / {len(test_classes)} ({len(unseen_test_classes)/len(test_classes)*100:.1f}%)")
    
    report = {
        "num_active_queries": len(active_5k),
        "num_unique_images": len(unique_active_imgs),
        "contiguous_slice_confirmed": (slice_imgs == unique_active_imgs),
        "queries_per_image": {
            "min": min(counts),
            "max": max(counts),
            "mean": sum(counts)/len(counts)
        },
        "category_coverage": {
            "total_classes": 258,
            "covered_classes": len(covered_classes),
            "zero_query_classes": len(zero_query_classes),
            "zero_query_class_names": zero_query_classes,
            "unseen_val_classes_count": len(unseen_val_classes),
            "unseen_test_classes_count": len(unseen_test_classes)
        }
    }
    
    with open("CSTBIR/data/train_diversity_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved diversity report to CSTBIR/data/train_diversity_report.json")
    return report

if __name__ == "__main__":
    analyze_diversity()

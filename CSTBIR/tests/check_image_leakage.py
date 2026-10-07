"""
PART P-9 (Step 1): Image-Filename Overlap & Data Leakage Audit
=============================================================
Computes exact image-filename intersections between:
1. Stratified 5K training set (train_subset_5k_stratified.json) vs:
   - Test-1K Gallery (val split, 1,000 images)
   - Test-5K Query Targets (test split targets, 2,214 images)
   - Test-5K Canonical Gallery (test5k_gallery_5000.json, 5,000 images)
2. Old contiguous 238-image training set vs:
   - Test-1K Gallery (val split, 1,000 images)
   - Test-5K Query Targets (test split targets, 2,214 images)
   - Test-5K Canonical Gallery (test5k_gallery_5000.json, 5,000 images)
"""

import os
import json
from typing import Set, Dict, Any

def audit_leakage() -> Dict[str, Any]:
    dataset_path = "CSTBIR/data/CSTBIR_dataset.json"
    stratified_path = "CSTBIR/data/train_subset_5k_stratified.json"
    gallery_5k_path = "CSTBIR/data/test5k_gallery_5000.json"
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
        
    with open(stratified_path, "r", encoding="utf-8") as f:
        stratified_data = json.load(f)
        
    val_queries = [x for x in all_data if x.get("split") == "val"]
    test_queries = [x for x in all_data if x.get("split") == "test"]
    old_train_queries = [x for x in all_data if x.get("split") == "train"][:5000]
    
    # Image sets
    val_gallery_imgs: Set[str] = set(x["image"] for x in val_queries)
    test_targets_imgs: Set[str] = set(x["image"] for x in test_queries)
    
    with open(gallery_5k_path, "r", encoding="utf-8") as f:
        test_5k_canonical_gallery: Set[str] = set(json.load(f))
        
    stratified_train_imgs: Set[str] = set(x["image"] for x in stratified_data)
    old_train_imgs: Set[str] = set(x["image"] for x in old_train_queries)
    
    print("=" * 80)
    print("PART P-9: IMAGE LEAKAGE & OVERLAP AUDIT")
    print("=" * 80)
    print(f"Stratified Train Images:         {len(stratified_train_imgs)}")
    print(f"Old Contiguous Train Images:     {len(old_train_imgs)}")
    print(f"Test-1K Gallery Images:          {len(val_gallery_imgs)}")
    print(f"Test-5K Query Target Images:     {len(test_targets_imgs)}")
    print(f"Test-5K Canonical Gallery (5000):{len(test_5k_canonical_gallery)}")
    print("-" * 80)
    
    # Intersections for Stratified Train Set
    strat_vs_val = stratified_train_imgs.intersection(val_gallery_imgs)
    strat_vs_test_targets = stratified_train_imgs.intersection(test_targets_imgs)
    strat_vs_test_5k = stratified_train_imgs.intersection(test_5k_canonical_gallery)
    
    print("\n--- [A] STRATIFIED TRAINING SET (3,324 Unique Images) ---")
    print(f"  Overlap with Test-1K Gallery:           {len(strat_vs_val)} images (Overlap: {len(strat_vs_val) > 0})")
    print(f"  Overlap with Test-5K Targets:           {len(strat_vs_test_targets)} images (Overlap: {len(strat_vs_test_targets) > 0})")
    print(f"  Overlap with Test-5K Canonical Gallery: {len(strat_vs_test_5k)} images (Overlap: {len(strat_vs_test_5k) > 0})")
    
    # Intersections for Old 238-Image Train Set
    old_vs_val = old_train_imgs.intersection(val_gallery_imgs)
    old_vs_test_targets = old_train_imgs.intersection(test_targets_imgs)
    old_vs_test_5k = old_train_imgs.intersection(test_5k_canonical_gallery)
    
    print("\n--- [B] OLD CONTIGUOUS TRAINING SET (238 Unique Images) ---")
    print(f"  Overlap with Test-1K Gallery:           {len(old_vs_val)} images (Overlap: {len(old_vs_val) > 0})")
    print(f"  Overlap with Test-5K Targets:           {len(old_vs_test_targets)} images (Overlap: {len(old_vs_test_targets) > 0})")
    print(f"  Overlap with Test-5K Canonical Gallery: {len(old_vs_test_5k)} images (Overlap: {len(old_vs_test_5k) > 0})")
    print("=" * 80)
    
    report = {
        "stratified_set": {
            "num_images": len(stratified_train_imgs),
            "overlap_test1k_gallery": len(strat_vs_val),
            "overlap_test5k_targets": len(strat_vs_test_targets),
            "overlap_test5k_canonical_gallery": len(strat_vs_test_5k),
            "is_clean": len(strat_vs_val) == 0 and len(strat_vs_test_targets) == 0 and len(strat_vs_test_5k) == 0
        },
        "old_contiguous_set": {
            "num_images": len(old_train_imgs),
            "overlap_test1k_gallery": len(old_vs_val),
            "overlap_test5k_targets": len(old_vs_test_targets),
            "overlap_test5k_canonical_gallery": len(old_vs_test_5k),
            "is_clean": len(old_vs_val) == 0 and len(old_vs_test_targets) == 0 and len(old_vs_test_5k) == 0
        }
    }
    
    out_path = "CSTBIR/data/leakage_audit_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nAudit report saved to {out_path}.")
    return report

if __name__ == "__main__":
    audit_leakage()

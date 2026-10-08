import os
import sys
import json
import random
from collections import Counter
from typing import Dict, List, Any

def build_sketch_classification_set(seed: int = 42):
    print("=" * 80)
    print("STEP 2: BUILDING QUICKDRAW SKETCH CLASSIFICATION PRETRAINING SET")
    print("=" * 80)
    
    dataset_path = "CSTBIR/data/CSTBIR_dataset.json"
    classes_path = "CSTBIR/data/classes_258.json"
    sketches_dir = "CSTBIR/data/quickdraw_sketches"
    
    with open(classes_path, "r", encoding="utf-8") as f:
        classes_list = json.load(f)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes_list)}
    print(f"Loaded {len(classes_list)} categories from {classes_path}.")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    print(f"Total raw queries across all splits in CSTBIR_dataset.json: {len(all_data):,}")
    
    # 1. Deduplicate by sketch filename across train + val + test
    sketch_to_label: Dict[str, str] = {}
    for item in all_data:
        sk_name = item.get("sketch")
        lbl = item.get("label")
        if sk_name and lbl and sk_name not in sketch_to_label:
            sketch_to_label[sk_name] = lbl
            
    print(f"Unique (sketch, label) pairs extracted: {len(sketch_to_label):,}")
    
    # 2. Verify media on disk (>1000 bytes) and valid class mapping
    valid_samples: List[Dict[str, Any]] = []
    missing_files = 0
    corrupt_files = 0
    unmapped_labels = 0
    
    for sk_name, lbl in sketch_to_label.items():
        if lbl not in class_to_idx:
            unmapped_labels += 1
            continue
            
        sk_path = os.path.join(sketches_dir, sk_name)
        if not os.path.exists(sk_path):
            missing_files += 1
            continue
            
        if os.path.getsize(sk_path) <= 1000:
            corrupt_files += 1
            continue
            
        valid_samples.append({
            "sketch": sk_name,
            "label": lbl,
            "class_idx": class_to_idx[lbl]
        })
        
    print(f"Verified valid sketch files on disk: {len(valid_samples):,}")
    print(f"  Missing files:    {missing_files}")
    print(f"  <=1000 bytes:     {corrupt_files}")
    print(f"  Unmapped labels:  {unmapped_labels}")
    
    # 3. Class distribution stats
    class_counts = Counter(s["class_idx"] for s in valid_samples)
    num_classes_represented = len(class_counts)
    counts = list(class_counts.values())
    min_count = min(counts) if counts else 0
    max_count = max(counts) if counts else 0
    mean_count = sum(counts) / len(counts) if counts else 0.0
    
    print("\n--- Class Representation Statistics ---")
    print(f"Total unique sketches verified:        {len(valid_samples):,}")
    print(f"Classes represented (out of 258):      {num_classes_represented} / {len(classes_list)}")
    print(f"Min examples per represented class:    {min_count}")
    print(f"Max examples per represented class:    {max_count}")
    print(f"Mean examples per represented class:   {mean_count:.1f}")
    
    if num_classes_represented < 200:
        print(f"\n[WARNING] Only {num_classes_represented} of 258 classes represented (< 200 threshold)!")
    else:
        print(f"\n[GATE 2 PASS CRITERIA MET] {num_classes_represented} of 258 classes represented (>= 200).")
        
    # 4. Stratified or seeded 90/10 split
    rng = random.Random(seed)
    # Shuffle deterministically
    shuffled = valid_samples[:]
    rng.shuffle(shuffled)
    
    split_idx = int(0.9 * len(shuffled))
    train_split = shuffled[:split_idx]
    val_split = shuffled[split_idx:]
    
    print(f"\n90/10 Split:")
    print(f"  Training sketches:   {len(train_split):,}")
    print(f"  Validation sketches: {len(val_split):,}")
    
    # Verify both splits cover classes well
    train_classes = len(set(s["class_idx"] for s in train_split))
    val_classes = len(set(s["class_idx"] for s in val_split))
    print(f"  Train classes covered: {train_classes} / {num_classes_represented}")
    print(f"  Val classes covered:   {val_classes} / {num_classes_represented}")
    
    # Save datasets
    train_out = "CSTBIR/data/sketch_classification_train.json"
    val_out = "CSTBIR/data/sketch_classification_val.json"
    
    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(train_split, f, indent=2)
    with open(val_out, "w", encoding="utf-8") as f:
        json.dump(val_split, f, indent=2)
        
    print(f"\nSaved training set to:   {train_out}")
    print(f"Saved validation set to: {val_out}")
    print("=" * 80)
    
    return {
        "total_sketches": len(valid_samples),
        "classes_represented": num_classes_represented,
        "min_per_class": min_count,
        "max_per_class": max_count,
        "mean_per_class": mean_count,
        "train_count": len(train_split),
        "val_count": len(val_split)
    }

if __name__ == "__main__":
    build_sketch_classification_set()

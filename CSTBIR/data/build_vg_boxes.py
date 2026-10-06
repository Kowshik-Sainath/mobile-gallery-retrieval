import os
import json
import zipfile
import time
from typing import Dict, List, Tuple, Optional

def normalize_label(label: str) -> str:
    return label.lower().replace('_', ' ').strip()

def match_object_box(
    objects: List[dict],
    label_str: str,
    img_width: int,
    img_height: int
) -> Optional[List[float]]:
    """
    Finds the best matching bounding box for label_str in the image objects.
    Returns normalized [x_min, y_min, x_max, y_max] in [0, 1].
    """
    lbl = normalize_label(label_str)
    best_match = None
    best_priority = 0
    max_area = -1
    
    # Simple singular/plural variations
    singular = lbl[:-1] if lbl.endswith('s') else lbl
    plural = lbl + 's' if not lbl.endswith('s') else lbl
    
    for obj in objects:
        names = [n.lower().strip() for n in obj.get('names', [])]
        synsets = [s.lower().strip() for s in obj.get('synsets', [])]
        
        priority = 0
        # Priority 3: Exact name match
        if lbl in names or singular in names or plural in names:
            priority = 3
        # Priority 2: Synset match
        elif any(lbl in s or singular in s for s in synsets):
            priority = 2
        # Priority 1: Substring word token match
        elif any(lbl in n or n in lbl for n in names):
            priority = 1
            
        if priority > 0:
            area = obj.get('w', 0) * obj.get('h', 0)
            if priority > best_priority or (priority == best_priority and area > max_area):
                best_priority = priority
                max_area = area
                best_match = obj
                
    if best_match is not None:
        x = float(best_match['x'])
        y = float(best_match['y'])
        w = float(best_match['w'])
        h = float(best_match['h'])
        
        x_min = max(0.0, min(1.0, x / img_width))
        y_min = max(0.0, min(1.0, y / img_height))
        x_max = max(0.0, min(1.0, (x + w) / img_width))
        y_max = max(0.0, min(1.0, (y + h) / img_height))
        
        # Ensure valid non-degenerate box
        if x_max > x_min and y_max > y_min:
            return [round(x_min, 4), round(y_min, 4), round(x_max, 4), round(y_max, 4)]
            
    return None

def build_vg_boxes_dict(
    image_data_zip: str = "CSTBIR/data/image_data.json.zip",
    objects_zip: str = "CSTBIR/data/objects.json.zip",
    dataset_json: str = "CSTBIR/data/CSTBIR_dataset.json",
    output_path: str = "CSTBIR/data/vg_boxes.json"
) -> Dict[str, List[float]]:
    print("=" * 75)
    print("STEP 1: Building Visual Genome Bounding Box Database for CSTBIR")
    print("=" * 75)
    
    # 1. Load image dimensions
    t0 = time.time()
    print("Loading image dimensions from image_data.json.zip...")
    with zipfile.ZipFile(image_data_zip) as z:
        with z.open("image_data.json") as f:
            img_data = json.load(f)
    img_dims = {d["image_id"]: (d["width"], d["height"]) for d in img_data}
    print(f"Loaded dimensions for {len(img_dims)} images in {time.time()-t0:.1f}s.")
    
    # 2. Load dataset queries to know which images we need
    print(f"\nLoading CSTBIR queries from {dataset_json}...")
    with open(dataset_json, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"Total dataset entries: {len(queries)}")
    
    # Filter to unique images present in CSTBIR
    needed_image_ids = set()
    for q in queries:
        try:
            img_id = int(q["image"].split(".")[0])
            needed_image_ids.add(img_id)
        except ValueError:
            pass
    print(f"Total unique images referenced in CSTBIR: {len(needed_image_ids)}")
    
    # 3. Stream/Parse objects.json.zip filtering for needed images
    t0 = time.time()
    print("\nLoading and filtering objects from objects.json.zip...")
    with zipfile.ZipFile(objects_zip) as z:
        with z.open("objects.json") as f:
            all_vg_objects = json.load(f)
            
    vg_objects_by_id = {}
    for entry in all_vg_objects:
        img_id = entry.get("image_id")
        if img_id in needed_image_ids:
            vg_objects_by_id[img_id] = entry.get("objects", [])
            
    print(f"Matched objects for {len(vg_objects_by_id)} / {len(needed_image_ids)} CSTBIR images in {time.time()-t0:.1f}s.")
    del all_vg_objects  # Free memory
    
    # 4. Match objects for each query
    print("\nMatching bounding boxes for CSTBIR queries...")
    t0 = time.time()
    vg_boxes_dict = {}
    
    splits_stats = {
        "train": {"total": 0, "matched": 0},
        "val": {"total": 0, "matched": 0},
        "test": {"total": 0, "matched": 0}
    }
    
    for q in queries:
        split = q.get("split")
        if split in splits_stats:
            splits_stats[split]["total"] += 1
            
        img_name = q["image"]
        label = q.get("label", "")
        key_pair = f"{img_name}_{label}"
        
        try:
            img_id = int(img_name.split(".")[0])
        except ValueError:
            continue
            
        if img_id not in img_dims or img_id not in vg_objects_by_id:
            continue
            
        w, h = img_dims[img_id]
        box = match_object_box(vg_objects_by_id[img_id], label, w, h)
        
        if box is not None:
            vg_boxes_dict[key_pair] = box
            # Also store image-level fallback if not yet set
            if img_name not in vg_boxes_dict:
                vg_boxes_dict[img_name] = box
                
            if split in splits_stats:
                splits_stats[split]["matched"] += 1
                
    print(f"Matching completed in {time.time()-t0:.1f}s.")
    print("\n--- Matching Statistics by Split ---")
    for s, stats in splits_stats.items():
        tot = stats["total"]
        m = stats["matched"]
        pct = (m / tot * 100) if tot > 0 else 0
        print(f"  Split '{s:5s}': {m:6d} / {tot:6d} queries matched ({pct:.1f}%)")
        
    print(f"\nTotal distinct box keys generated: {len(vg_boxes_dict)}")
    
    # 5. Save to disk
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vg_boxes_dict, f, indent=2)
    print(f"Saved bounding box dictionary to {output_path} ({os.path.getsize(output_path)/1e6:.1f} MB)")
    
    return vg_boxes_dict

if __name__ == "__main__":
    build_vg_boxes_dict()

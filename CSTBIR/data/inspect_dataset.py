import json
from collections import Counter

print("Loading CSTBIR_dataset.json...")
with open("CSTBIR/data/CSTBIR_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Type of data:", type(data))

if isinstance(data, dict):
    print("Dict keys:", list(data.keys()))
    for k in list(data.keys())[:5]:
        v = data[k]
        print(f"Key: {k}, type: {type(v)}, len: {len(v) if hasattr(v, '__len__') else 'N/A'}")
elif isinstance(data, list):
    print(f"Total entries: {len(data)}")
    print("Sample entry 0:", data[0])
    print("Sample entry 1:", data[1])
    
    # Check splits
    splits = Counter(d.get("split") for d in data)
    print("\nSplit distribution in CSTBIR_dataset.json:")
    for s, count in splits.most_common():
        print(f"  Split '{s}': {count} queries")
        
    # Check unique images and categories per split
    print("\nDetailed split statistics:")
    for s in splits.keys():
        sub = [d for d in data if d.get("split") == s]
        unique_imgs = len(set(d.get("image_filename") or d.get("image_id") or d.get("image") for d in sub))
        unique_sketches = len(set(d.get("sketch_filename") or d.get("sketch_id") or d.get("sketch") for d in sub))
        unique_cats = len(set(d.get("object_category") or d.get("category") or d.get("class") for d in sub))
        print(f"  Split '{s}': {len(sub)} queries | {unique_imgs} unique images | {unique_sketches} unique sketches | {unique_cats} categories")

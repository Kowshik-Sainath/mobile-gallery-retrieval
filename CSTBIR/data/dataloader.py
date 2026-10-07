import os
import io
import json
import time
import random
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

try:
    from CSTBIR.clip import clip
except ImportError:
    import clip

def safe_load_rgb(path: str, retries: int = 5) -> Image.Image:
    """Safe image reader avoiding Windows OneDrive file-handle leaks."""
    for attempt in range(retries):
        try:
            with open(path, "rb") as f:
                data = f.read()
            bio = io.BytesIO(data)
            with Image.open(bio) as img:
                return img.convert("RGB")
        except OSError:
            time.sleep(0.02 * (attempt + 1))
    with Image.open(path) as img:
        return img.convert("RGB")

class CSTBIRDataset(Dataset):
    """
    Robust Dataloader for CSTBIR Dataset (AAAI 2024).
    
    Parses CSTBIR_dataset.json as released by authors:
        Total queries: 1,989,873 across 258 object classes.
        Splits:
            - 'train':      1,887,884 queries (~97K images, ~478K sketches)
            - 'val':        1,000 queries (Test-1K gallery set)
            - 'test':       4,001 queries (Test-5K set)
            - 'val_large':  96,988 queries (~5K images)
    """
    def __init__(
        self,
        json_path: str,
        split: str = 'train',
        images_dir: Optional[str] = None,
        sketches_dir: Optional[str] = None,
        sketch_embeddings_dict: Optional[Dict[str, torch.Tensor]] = None,
        vg_boxes_dict: Optional[Dict[str, List[float]]] = None,
        classes_path: Optional[str] = None,
        preprocess = None,
        sketch_transform = None,
        filter_available: bool = True
    ):
        super().__init__()
        self.split = split
        self.images_dir = images_dir
        self.sketches_dir = sketches_dir
        self.sketch_embeddings_dict = sketch_embeddings_dict
        # Load VG Bounding Boxes dictionary (real Visual Genome spatial annotations)
        if isinstance(vg_boxes_dict, str) and os.path.exists(vg_boxes_dict):
            print(f"[CSTBIRDataset] Loading real VG bounding boxes from {vg_boxes_dict}...")
            with open(vg_boxes_dict, 'r', encoding='utf-8') as f:
                self.vg_boxes_dict = json.load(f)
        elif vg_boxes_dict is not None and len(vg_boxes_dict) > 0:
            self.vg_boxes_dict = vg_boxes_dict
        else:
            default_boxes_path = os.path.join(os.path.dirname(__file__), "vg_boxes.json")
            if os.path.exists(default_boxes_path):
                print(f"[CSTBIRDataset] Auto-loading real VG bounding boxes from {default_boxes_path}...")
                with open(default_boxes_path, 'r', encoding='utf-8') as f:
                    self.vg_boxes_dict = json.load(f)
            else:
                self.vg_boxes_dict = {}
        
        # Default CLIP and Sketch Preprocessors
        self.preprocess = preprocess or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.sketch_transform = sketch_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.target_sketch_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
        
        # Load classes list
        if classes_path and os.path.exists(classes_path):
            with open(classes_path, 'r', encoding='utf-8') as f:
                self.classes = json.load(f)
        else:
            default_classes_path = os.path.join(os.path.dirname(__file__), "classes_258.json")
            if os.path.exists(default_classes_path):
                with open(default_classes_path, 'r', encoding='utf-8') as f:
                    self.classes = json.load(f)
            else:
                self.classes = []
                
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # Load dataset json
        print(f"[CSTBIRDataset] Loading {json_path} for split '{split}'...")
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # Filter by split
        self.samples = [d for d in raw_data if d.get('split') == split]
        print(f"[CSTBIRDataset] Split '{split}' has {len(self.samples)} raw queries.")
        
        # Filter available media if requested and directories provided
        if filter_available and self.images_dir and os.path.exists(self.images_dir) and self.sketches_dir and os.path.exists(self.sketches_dir):
            avail_imgs = set(os.listdir(self.images_dir))
            avail_sks = set(os.listdir(self.sketches_dir))
            filtered = [
                d for d in self.samples
                if d['image'] in avail_imgs and d['sketch'] in avail_sks
            ]
            if len(filtered) > 0:
                self.samples = filtered
                print(f"[CSTBIRDataset] Retained {len(self.samples)} queries with verified local media files.")
            else:
                print(f"[CSTBIRDataset] Warning: No queries matched local media files; keeping raw split.")
        
        # Build uniqueness indices for conditional conflict-free sampling
        self.image_to_indices: Dict[str, List[int]] = {}
        self.text_to_indices: Dict[str, List[int]] = {}
        for idx, item in enumerate(self.samples):
            img = item['image']
            txt = item['text']
            self.image_to_indices.setdefault(img, []).append(idx)
            self.text_to_indices.setdefault(txt, []).append(idx)

        # In-memory RAM caches for instant tensor access
        self._img_cache: Dict[str, torch.Tensor] = {}
        self._sketch_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        text_str = item['text']
        img_name = item['image']
        sketch_name = item['sketch']
        label_str = item['label']
        class_idx = self.class_to_idx.get(label_str, 0)
        
        # 1. Tokenize query text with CLIP BPE tokenizer
        tokenized_text = clip.tokenize(text_str, truncate=True).squeeze(0)  # (77,)
        
        # 2. Image loading / preprocessing with caching
        if img_name in self._img_cache:
            processed_img = self._img_cache[img_name]
        elif self.images_dir and os.path.exists(os.path.join(self.images_dir, img_name)):
            img_path = os.path.join(self.images_dir, img_name)
            raw_img = safe_load_rgb(img_path)
            processed_img = self.preprocess(raw_img)
            self._img_cache[img_name] = processed_img
        else:
            processed_img = torch.zeros(3, 224, 224)
            
        # 3. Sketch handling (input tensor + reconstruction target) with caching
        sketch_embed = None
        if sketch_name in self._sketch_cache:
            sketch_img, target_sketch = self._sketch_cache[sketch_name]
        elif self.sketch_embeddings_dict and sketch_name in self.sketch_embeddings_dict:
            sketch_embed = self.sketch_embeddings_dict[sketch_name]
            sketch_img = torch.zeros(3, 224, 224)
            target_sketch = torch.zeros(1, 224, 224)
        elif self.sketches_dir and os.path.exists(os.path.join(self.sketches_dir, sketch_name)):
            raw_sketch = safe_load_rgb(os.path.join(self.sketches_dir, sketch_name))
            sketch_img = self.sketch_transform(raw_sketch)
            target_sketch = self.target_sketch_transform(raw_sketch)
            self._sketch_cache[sketch_name] = (sketch_img, target_sketch)
        else:
            sketch_embed = torch.zeros(768)
            sketch_img = torch.zeros(3, 224, 224)
            target_sketch = torch.zeros(1, 224, 224)
            
        # 4. Bounding box for L_OD: [x_min, y_min, x_max, y_max] normalized in [0, 1]
        # Query-specific object match: check (img_name, label_str) first, then image fallback
        key_pair = f"{img_name}_{label_str}"
        if key_pair in self.vg_boxes_dict:
            bbox = self.vg_boxes_dict[key_pair]
        elif (img_name, label_str) in self.vg_boxes_dict:
            bbox = self.vg_boxes_dict[(img_name, label_str)]
        elif img_name in self.vg_boxes_dict:
            bbox = self.vg_boxes_dict[img_name]
        else:
            bbox = [0.25, 0.25, 0.75, 0.75]
        bbox_tensor = torch.tensor(bbox, dtype=torch.float32)
        
        return {
            'text': tokenized_text,
            'image': processed_img,
            'sketch_embed': sketch_embed,
            'sketch_img': sketch_img,
            'target_sketch': target_sketch,
            'label': class_idx,
            'bbox': bbox_tensor,
            'image_name': img_name,
            'sketch_name': sketch_name,
            'raw_text': text_str
        }

    def get_conflict_free_batch(self, batch_size: int, rng: Optional[random.Random] = None) -> dict:
        """
        Samples a batch where no two queries share the same image or text,
        matching conditional sampling described in Section: STNET Training.
        """
        rand_fn = rng.randint if rng is not None else random.randint
        selected_indices = []
        selected_images = set()
        selected_texts = set()
        
        attempts = 0
        max_attempts = batch_size * 50
        while len(selected_indices) < batch_size and attempts < max_attempts:
            attempts += 1
            idx = rand_fn(0, len(self.samples) - 1)
            img = self.samples[idx]['image']
            txt = self.samples[idx]['text']
            
            if img in selected_images or txt in selected_texts:
                continue
                
            selected_images.add(img)
            selected_texts.add(txt)
            selected_indices.append(idx)
            
        # Collate items
        batch_items = [self.__getitem__(i) for i in selected_indices]
        
        collated = {
            'text': torch.stack([b['text'] for b in batch_items], dim=0),
            'image': torch.stack([b['image'] for b in batch_items], dim=0),
            'sketch_embed': torch.stack([b['sketch_embed'] for b in batch_items], dim=0) if batch_items[0]['sketch_embed'] is not None else None,
            'sketch_img': torch.stack([b['sketch_img'] for b in batch_items], dim=0) if batch_items[0]['sketch_img'] is not None else None,
            'target_sketch': torch.stack([b['target_sketch'] for b in batch_items], dim=0),
            'label': torch.tensor([b['label'] for b in batch_items], dtype=torch.long),
            'bbox': torch.stack([b['bbox'] for b in batch_items], dim=0),
            'image_names': [b['image_name'] for b in batch_items],
            'sketch_names': [b['sketch_name'] for b in batch_items]
        }
        return collated

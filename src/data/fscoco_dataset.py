"""
FS-COCO Dataset Loader & Adapter.
Refactored from pinakinathc/fscoco to support unified (sketch, caption, photo) retrieval.
"""

import os
from glob import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from .data_utils import get_transforms

class FSCOCODataset(Dataset):
    """
    FS-COCO Dataset for Freehand Scene Sketch + Text + Photo Triples.
    Official split: 9,525 train / 475 test (or custom split via val_unseen_user.txt / val_normal.txt).
    """
    def __init__(self, root_dir, split='train', image_size=224, test_split_file='val_unseen_user.txt'):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.photo_transform, self.sketch_transform, self.target_sketch_transform = get_transforms(image_size)
        
        # Determine actual root path
        if not os.path.exists(os.path.join(root_dir, 'images')) and os.path.exists(os.path.join(root_dir, 'fscoco', 'images')):
            self.dataset_root = os.path.join(root_dir, 'fscoco')
        elif not os.path.exists(os.path.join(root_dir, 'images')) and os.path.exists(os.path.join(root_dir, 'fscoco', 'fscoco', 'images')):
            self.dataset_root = os.path.join(root_dir, 'fscoco', 'fscoco')
        else:
            self.dataset_root = root_dir

        self.samples = []
        self._load_samples(test_split_file)

    def _load_samples(self, test_split_file):
        test_ids = set()
        split_path = os.path.join(self.dataset_root, test_split_file)
        if os.path.exists(split_path):
            with open(split_path, 'r', encoding='utf-8') as f:
                test_ids = set(line.strip() for line in f if line.strip())

        # Scan text directory for valid triples
        text_dir = os.path.join(self.dataset_root, 'text')
        if not os.path.exists(text_dir):
            raise FileNotFoundError(f"FS-COCO text directory not found at {text_dir}")

        for subfolder in os.listdir(text_dir):
            sub_text_path = os.path.join(text_dir, subfolder)
            if not os.path.isdir(sub_text_path):
                continue
            
            for txt_file in os.listdir(sub_text_path):
                if not txt_file.endswith('.txt'):
                    continue
                img_id = os.path.splitext(txt_file)[0]
                
                # Filter based on train / test split
                is_test = img_id in test_ids
                if self.split == 'train' and is_test:
                    continue
                if self.split == 'test' and not is_test:
                    continue
                
                txt_full_path = os.path.join(sub_text_path, txt_file)
                
                # Check for image file (.jpg or .png)
                img_full_path = os.path.join(self.dataset_root, 'images', subfolder, f"{img_id}.jpg")
                if not os.path.exists(img_full_path):
                    img_full_path = os.path.join(self.dataset_root, 'images', subfolder, f"{img_id}.png")

                # Check for sketch file (.jpg or .png)
                sketch_full_path = os.path.join(self.dataset_root, 'raster_sketches', subfolder, f"{img_id}.jpg")
                if not os.path.exists(sketch_full_path):
                    sketch_full_path = os.path.join(self.dataset_root, 'raster_sketches', subfolder, f"{img_id}.png")

                # Verify file existence
                if os.path.exists(img_full_path) and os.path.exists(sketch_full_path):
                    self.samples.append({
                        'id': img_id,
                        'text_path': txt_full_path,
                        'photo_path': img_full_path,
                        'sketch_path': sketch_full_path
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load text caption
        with open(sample['text_path'], 'r', encoding='utf-8') as f:
            caption = f.read().strip()

        # Load images
        photo_img = Image.open(sample['photo_path']).convert('RGB')
        sketch_img = Image.open(sample['sketch_path']).convert('RGB')

        photo_tensor = self.photo_transform(photo_img)
        sketch_tensor = self.sketch_transform(sketch_img)
        target_sketch_tensor = self.target_sketch_transform(sketch_img)

        return {
            'id': sample['id'],
            'sketch': sketch_tensor,
            'caption': caption,
            'photo': photo_tensor,
            'target_sketch': target_sketch_tensor
        }

def get_fscoco_dataloaders(root_dir, batch_size=32, num_workers=2):
    train_dataset = FSCOCODataset(root_dir, split='train')
    test_dataset = FSCOCODataset(root_dir, split='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    return train_loader, test_loader

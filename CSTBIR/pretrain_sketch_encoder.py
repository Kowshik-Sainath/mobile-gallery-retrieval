import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
from CSTBIR.models.sketch_encoder import SketchEncoder

class SketchClassificationDataset(Dataset):
    def __init__(self, json_path: str, sketches_dir: str, transform=None):
        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)
        self.sketches_dir = sketches_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.samples[idx]
        p = os.path.join(self.sketches_dir, item["sketch"])
        with Image.open(p) as img:
            sketch = img.convert("RGB")
        if self.transform:
            sketch = self.transform(sketch)
        return sketch, item["class_idx"]

def pretrain_sketch_encoder(
    train_json: str = "CSTBIR/data/sketch_classification_train.json",
    val_json: str = "CSTBIR/data/sketch_classification_val.json",
    sketches_dir: str = "CSTBIR/data/quickdraw_sketches",
    num_classes: int = 258,
    batch_size: int = 32,
    lr: float = 1e-5,
    epochs: int = 8,
    save_path: str = "CSTBIR/checkpoints/sketch_encoder_quickdraw_adapted.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print("=" * 80)
    print("STEP 3: PRETRAINING SKETCH ENCODER (ViT-Base ImageNet-21K -> QuickDraw 258)")
    print("=" * 80)
    print(f"Device: {device} | Batch Size: {batch_size} | Base LR: {lr} | Epochs: {epochs}")
    random_chance = 100.0 / num_classes
    print(f"Random-Chance Baseline (1/{num_classes}): {random_chance:.2f}%\n")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 1. Transforms
    sketch_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 2. Datasets & Loaders
    train_ds = SketchClassificationDataset(train_json, sketches_dir, transform=sketch_transform)
    val_ds = SketchClassificationDataset(val_json, sketches_dir, transform=sketch_transform)
    print(f"Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,}")
    
    num_workers = 2 if os.name == 'nt' else 4
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # 3. Model: Real ImageNet-21K pretrained ViT-Base
    print("\nInitializing SketchEncoder with pretrained=True (ImageNet-21K)...")
    model = SketchEncoder(num_classes=num_classes, pretrained=True).to(device)
    
    # 4. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader))
    
    best_val_acc = 0.0
    history: List[Dict[str, float]] = []
    
    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for sketches, targets in pbar:
            sketches = sketches.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            logits = model(sketches)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item() * sketches.size(0)
            preds = logits.argmax(dim=-1)
            correct_train += (preds == targets).sum().item()
            total_train += sketches.size(0)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.0 * correct_train / total_train:.2f}%"
            })
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = 100.0 * correct_train / total_train
        
        # Validation evaluation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for sketches, targets in val_loader:
                sketches = sketches.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                logits = model(sketches)
                loss = criterion(logits, targets)
                
                val_loss += loss.item() * sketches.size(0)
                preds = logits.argmax(dim=-1)
                correct_val += (preds == targets).sum().item()
                total_val += sketches.size(0)
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = 100.0 * correct_val / total_val
        epoch_time = time.time() - t0
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc,
            "time_sec": epoch_time
        })
        
        is_best = epoch_val_acc > best_val_acc
        if is_best:
            best_val_acc = epoch_val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_acc": best_val_acc,
                "epoch": epoch + 1,
                "num_classes": num_classes,
                "history": history
            }, save_path)
            
        print(f"\n[Epoch {epoch+1}/{epochs} Summary ({epoch_time:.1f}s)]")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:5.2f}%")
        print(f"  Val Loss:   {epoch_val_loss:.4f} | Val Acc:   {epoch_val_acc:5.2f}% (Chance: {random_chance:.2f}%) {'[NEW BEST - SAVED]' if is_best else ''}")
        
    print("\n" + "=" * 80)
    print("GATE 3 TRAJECTORY REPORT:")
    print("=" * 80)
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Train Acc (%)':<14} | {'Val Loss':<10} | {'Val Acc (%)':<14} | {'vs Chance':<12}")
    print("-" * 80)
    for h in history:
        mult = h['val_acc'] / random_chance
        print(f"{h['epoch']:<6} | {h['train_loss']:<10.4f} | {h['train_acc']:<14.2f} | {h['val_loss']:<10.4f} | {h['val_acc']:<14.2f} | {mult:6.1f}x chance")
    print("=" * 80)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Random chance: {random_chance:.2f}%)")
    print(f"Checkpoint saved to: {save_path}")
    print("=" * 80)
    
    # Save training history json
    hist_json = "CSTBIR/checkpoints/sketch_pretrain_history.json"
    with open(hist_json, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        
    assert best_val_acc > 10.0, f"Gate 3 Failed: Best val accuracy {best_val_acc:.2f}% is too low!"
    return best_val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    pretrain_sketch_encoder(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

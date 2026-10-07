import sys
import os
import yaml
import time
import random
import argparse
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
from CSTBIR.models.stnet import STNet
from CSTBIR.data.dataloader import CSTBIRDataset
from CSTBIR.evaluate_retrieval import evaluate_gallery_ranking

def set_seed(seed: int = 42):
    """Fix random seeds for 100% reproducible training and batch sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_stnet(
    config_path: str = "CSTBIR/configs/stnet_train.yaml",
    lambda_od: float = 1.0,
    seed: int = 42,
    run_name: Optional[str] = None
) -> Tuple[Dict[str, List[float]], str]:
    # 0. Set seed for deterministic initialization & batch sequences
    set_seed(seed)
    batch_rng = random.Random(seed)
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n" + "=" * 70)
    print(f"=== STNet Training (AAAI 2024 Reimplementation) ===")
    print(f"Run Name:  {run_name if run_name else 'default'}")
    print(f"lambda_od: {lambda_od} | Seed: {seed} | Device: {device}")
    print("=" * 70)
    
    base_save_dir = cfg["training"]["save_dir"]
    save_dir = os.path.join(base_save_dir, run_name) if run_name else base_save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 1. Initialize Model
    model = STNet(
        clip_model_name=cfg["model"]["clip_model_name"],
        num_classes=cfg["model"]["num_classes"],
        device=device,
        pretrained_sketch=False
    )
    
    # 2. Datasets
    json_path = cfg["data"]["dataset_json_path"]
    vg_boxes_path = cfg["data"].get("vg_boxes_path", "CSTBIR/data/vg_boxes.json")
    train_ds = CSTBIRDataset(
        json_path=json_path,
        split=cfg["data"]["train_split"],
        images_dir=cfg["data"]["images_dir"],
        sketches_dir=cfg["data"]["sketches_dir"],
        vg_boxes_dict=vg_boxes_path,
        classes_path=cfg["data"]["classes_path"]
    )
    
    val_ds = CSTBIRDataset(
        json_path=json_path,
        split=cfg["data"]["val_split"],
        images_dir=cfg["data"]["images_dir"],
        sketches_dir=cfg["data"]["sketches_dir"],
        vg_boxes_dict=vg_boxes_path,
        classes_path=cfg["data"]["classes_path"]
    )
    
    batch_size = cfg["model"]["batch_size"]
    learning_rate = float(cfg["model"]["learning_rate"])
    weight_decay = float(cfg["model"]["weight_decay"])
    epochs = cfg["training"]["epochs"]
    
    # 3. Optimizer & Scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    total_steps = len(train_ds) // batch_size * epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    print(f"Optimizer: Adam | Base LR: {learning_rate} | Batch Size: {batch_size} | Epochs: {epochs}")
    print(f"Training split queries: {len(train_ds)} | Test-1K queries: {len(val_ds)}")
    
    # History logs for all loss terms
    history = {
        'loss_total': [],
        'loss_ct': [],
        'loss_cls_t': [],
        'loss_cls_i': [],
        'loss_od': [],
        'loss_od_weighted': [],
        'loss_sr': []
    }
    
    steps_per_epoch = cfg["training"].get("steps_per_epoch", min(len(train_ds) // batch_size, 1000))
    best_ckpt_path = os.path.join(save_dir, "stnet_best.pt")
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = {k: 0.0 for k in history.keys()}
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
        
        for step in pbar:
            batch = train_ds.get_conflict_free_batch(batch_size=batch_size, rng=batch_rng)
            
            text_tokens = batch['text'].to(device)
            images = batch['image'].to(device)
            sketches = batch['sketch_img'].to(device) if batch['sketch_img'] is not None else None
            sketch_embeds = batch['sketch_embed'].to(device) if batch['sketch_embed'] is not None else None
            gt_boxes = batch['bbox'].to(device)
            gt_labels = batch['label'].to(device)
            target_sketches = batch['target_sketch'].to(device) if 'target_sketch' in batch and batch['target_sketch'] is not None else None
            
            optimizer.zero_grad()
            
            outputs = model(
                text_tokens=text_tokens,
                images=images,
                sketches=sketches,
                sketch_embeds=sketch_embeds,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                target_sketch_imgs=target_sketches,
                lambda_od=lambda_od
            )
            
            loss_total = outputs['loss_total']
            loss_total.backward()
            
            if cfg["training"].get("gradient_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["gradient_clip"])
                
            optimizer.step()
            scheduler.step()
            
            # Record individual losses
            for k in epoch_losses.keys():
                epoch_losses[k] += outputs[k].item()
                
            pbar.set_postfix({
                'L_tot': f"{outputs['loss_total'].item():.2f}",
                'L_ct': f"{outputs['loss_ct'].item():.2f}",
                'L_cls': f"{(outputs['loss_cls_t'] + outputs['loss_cls_i']).item():.2f}",
                'L_od': f"{outputs['loss_od'].item():.2f}",
                'L_sr': f"{outputs['loss_sr'].item():.2f}"
            })
            
        # End of epoch summary
        for k in epoch_losses.keys():
            avg_loss = epoch_losses[k] / steps_per_epoch
            history[k].append(avg_loss)
            
        print(f"\n[Epoch {epoch+1}/{epochs} Summary (lambda_od={lambda_od})]")
        print(f"  L_total:     {history['loss_total'][-1]:.4f}")
        print(f"  L_CT:        {history['loss_ct'][-1]:.4f}")
        print(f"  L_CLS^T:     {history['loss_cls_t'][-1]:.4f}")
        print(f"  L_CLS^I:     {history['loss_cls_i'][-1]:.4f}")
        print(f"  L_OD (raw):  {history['loss_od'][-1]:.4f}")
        print(f"  L_OD (wtd):  {history['loss_od_weighted'][-1]:.4f}")
        print(f"  L_SR:        {history['loss_sr'][-1]:.4f}")
        
        # Save epoch checkpoint
        ckpt_path = os.path.join(save_dir, f"stnet_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'lambda_od': lambda_od,
            'seed': seed
        }, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")
        
        # Save best model based on contrastive retrieval loss (L_CT)
        if history['loss_ct'][-1] == min(history['loss_ct']):
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'lambda_od': lambda_od,
                'seed': seed
            }, best_ckpt_path)
            print(f"  * New best L_CT ({history['loss_ct'][-1]:.4f})! Saved to {best_ckpt_path}")
            
        # Save history json
        import json
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as hf:
            json.dump(history, hf, indent=2)
            
    print(f"\nTraining completed successfully for {run_name if run_name else 'default'}.")
    return history, best_ckpt_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="CSTBIR/configs/stnet_train.yaml")
    parser.add_argument("--lambda_od", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()
    train_stnet(config_path=args.config, lambda_od=args.lambda_od, seed=args.seed, run_name=args.run_name)

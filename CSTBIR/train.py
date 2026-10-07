import sys
import os
import yaml
import time
import random
import argparse
import json
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("."))
from CSTBIR.models.stnet import STNet
from CSTBIR.data.dataloader import CSTBIRDataset
from CSTBIR.clip import clip
from CSTBIR.evaluate_retrieval import compute_retrieval_metrics

def set_seed(seed: int = 42):
    """Fix random seeds for 100% reproducible training and batch sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class DiagnosticEvaluator:
    """
    Evaluates STNet on a held-out Test-1K diagnostic subset every epoch.
    Pre-caches all images into CPU RAM to minimize disk I/O overhead.
    """
    def __init__(
        self,
        json_path: str = "CSTBIR/data/CSTBIR_dataset.json",
        images_dir: str = "CSTBIR/data/vg_images",
        sketches_dir: str = "CSTBIR/data/quickdraw_sketches",
        num_queries: int = 200,
        device: str = "cuda"
    ):
        self.device = device
        with open(json_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
            
        val_queries = [x for x in all_data if x.get("split") == "val"]
        self.val_gallery_names = sorted(list(set(x["image"] for x in val_queries)))
        self.img_to_idx = {name: i for i, name in enumerate(self.val_gallery_names)}
        
        # Verify valid media files
        valid_val = []
        for q in val_queries:
            ip = os.path.join(images_dir, q["image"])
            sp = os.path.join(sketches_dir, q["sketch"])
            if os.path.exists(ip) and os.path.getsize(ip) > 1000 and os.path.exists(sp) and os.path.getsize(sp) > 1000:
                valid_val.append(q)
                
        self.diag_queries = valid_val[:num_queries]
        self.gt_indices = [self.img_to_idx[q["image"]] for q in self.diag_queries]
        
        self.clip_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.sketch_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print(f"\n[DiagnosticEvaluator] Pre-caching {len(self.val_gallery_names)} gallery images in RAM...")
        gal_imgs = []
        for name in self.val_gallery_names:
            p = os.path.join(images_dir, name)
            with Image.open(p) as img:
                gal_imgs.append(self.clip_preprocess(img.convert("RGB")))
        self.gallery_img_tensors = torch.stack(gal_imgs)
        
        self.query_text_tokens = clip.tokenize([q["text"] for q in self.diag_queries], truncate=True)
        
        q_sks = []
        for q in self.diag_queries:
            p = os.path.join(sketches_dir, q["sketch"])
            with Image.open(p) as img:
                q_sks.append(self.sketch_preprocess(img.convert("RGB")))
        self.query_sketch_tensors = torch.stack(q_sks)
        print(f"[DiagnosticEvaluator] Ready: {len(self.diag_queries)} held-out queries against {len(self.val_gallery_names)} gallery images.")

    def evaluate(self, model: STNet) -> Tuple[Dict[str, float], float]:
        model.eval()
        t0 = time.time()
        K = len(self.val_gallery_names)
        Q = len(self.diag_queries)
        batch_size = 64
        
        with torch.no_grad():
            gallery_tokens = []
            for i in range(0, K, batch_size):
                sub_imgs = self.gallery_img_tensors[i : i + batch_size].to(self.device)
                gallery_tokens.append(model.extract_image_tokens(sub_imgs))
            H_gallery = torch.cat(gallery_tokens, dim=0) # (K, 197, 768)
            
            h_T = model.encode_text(self.query_text_tokens.to(self.device))
            z_t = F.normalize(model.loss_ct_fn.txt_proj(h_T), p=2, dim=-1) # (Q, 512)
            
            h_S = model.encode_sketch(self.query_sketch_tensors.to(self.device)) # (Q, 768)
            
            ranks = []
            for i in range(Q):
                sq = h_S[i : i + 1].repeat(K, 1) # (K, 768)
                h_I_avg, _, _ = model.cross_attention(H_gallery, sq)
                z_i = F.normalize(model.loss_ct_fn.img_proj(h_I_avg), p=2, dim=-1)
                sims = (z_t[i : i + 1] @ z_i.T).squeeze(0).cpu().numpy()
                sorted_idx = np.argsort(-sims)
                rank = int(np.where(sorted_idx == self.gt_indices[i])[0][0]) + 1
                ranks.append(rank)
                
        metrics = compute_retrieval_metrics(ranks)
        eval_time = time.time() - t0
        return metrics, eval_time

def train_stnet(
    config_path: str = "CSTBIR/configs/stnet_train.yaml",
    lambda_od: float = 1.0,
    seed: int = 42,
    run_name: Optional[str] = None,
    epochs_override: Optional[int] = None,
    patience: int = 4,
    min_epochs: int = 5,
    diag_samples: int = 200
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
    train_json_path = cfg["data"].get("train_json_path", json_path)
    vg_boxes_path = cfg["data"].get("vg_boxes_path", "CSTBIR/data/vg_boxes.json")
    train_ds = CSTBIRDataset(
        json_path=train_json_path,
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
    epochs = epochs_override if epochs_override is not None else cfg["training"]["epochs"]
    steps_per_epoch = cfg["training"].get("steps_per_epoch", min(len(train_ds) // batch_size, 1000))
    total_steps = steps_per_epoch * epochs
    
    # 3. Optimizer & Scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    print(f"Optimizer: Adam | Base LR: {learning_rate} | Batch Size: {batch_size} | Epochs: {epochs} | Steps/Epoch: {steps_per_epoch}")
    print(f"Training split queries: {len(train_ds)} | Test-1K queries: {len(val_ds)}")
    
    # 4. Diagnostic Evaluator on Held-Out Test-1K
    diag_evaluator = DiagnosticEvaluator(
        json_path=json_path,
        images_dir=cfg["data"]["images_dir"],
        sketches_dir=cfg["data"]["sketches_dir"],
        num_queries=diag_samples,
        device=device
    )
    
    # History logs for all loss terms & diagnostic metrics
    history = {
        'loss_total': [],
        'loss_ct': [],
        'loss_cls_t': [],
        'loss_cls_i': [],
        'loss_od': [],
        'loss_od_weighted': [],
        'loss_sr': [],
        'diag_r10': [],
        'diag_r20': [],
        'diag_mdr': []
    }
    
    best_ckpt_path = os.path.join(save_dir, "stnet_best.pt")
    best_diag_r10 = -1.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = {k: 0.0 for k in ['loss_total', 'loss_ct', 'loss_cls_t', 'loss_cls_i', 'loss_od', 'loss_od_weighted', 'loss_sr']}
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
            
        print(f"\n[Epoch {epoch+1}/{epochs} Training Summary (lambda_od={lambda_od})]")
        print(f"  L_total:     {history['loss_total'][-1]:.4f}")
        print(f"  L_CT:        {history['loss_ct'][-1]:.4f}")
        print(f"  L_CLS^T:     {history['loss_cls_t'][-1]:.4f}")
        print(f"  L_CLS^I:     {history['loss_cls_i'][-1]:.4f}")
        print(f"  L_OD (raw):  {history['loss_od'][-1]:.4f}")
        print(f"  L_SR:        {history['loss_sr'][-1]:.4f}")
        
        # Run held-out diagnostic evaluation
        diag_metrics, diag_time = diag_evaluator.evaluate(model)
        diag_r10 = diag_metrics['R@10']
        diag_r20 = diag_metrics['R@20']
        diag_mdr = diag_metrics['MdR']
        
        history['diag_r10'].append(diag_r10)
        history['diag_r20'].append(diag_r20)
        history['diag_mdr'].append(diag_mdr)
        
        print(f"  >>> Held-Out Diagnostic (200-sample): R@10 = {diag_r10:5.2f}% (238-img Baseline: 4.50%) | R@20 = {diag_r20:5.2f}% | MdR = {diag_mdr:5.1f} ({diag_time:.1f}s)")
        
        # Save epoch checkpoint
        ckpt_path = os.path.join(save_dir, f"stnet_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'lambda_od': lambda_od,
            'seed': seed,
            'diag_metrics': diag_metrics
        }, ckpt_path)
        
        # Save best model based on diagnostic R@10
        if diag_r10 > best_diag_r10:
            best_diag_r10 = diag_r10
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'lambda_od': lambda_od,
                'seed': seed,
                'diag_metrics': diag_metrics
            }, best_ckpt_path)
            print(f"  * New best diagnostic R@10 ({diag_r10:.2f}%)! Saved to {best_ckpt_path}")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience} (Best R@10: {best_diag_r10:.2f}% at Epoch {best_epoch})")
            if patience_counter >= patience and (epoch + 1) >= min_epochs:
                print(f"\n[Early Stopping Triggered] Diagnostic R@10 has not improved for {patience} epochs.")
                break
            
        # Save history json
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as hf:
            json.dump(history, hf, indent=2)
            
    print(f"\nTraining completed. Best diagnostic R@10: {best_diag_r10:.2f}% at Epoch {best_epoch}.")
    return history, best_ckpt_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="CSTBIR/configs/stnet_train.yaml")
    parser.add_argument("--lambda_od", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min_epochs", type=int, default=5)
    parser.add_argument("--diag_samples", type=int, default=200)
    args = parser.parse_args()
    train_stnet(
        config_path=args.config,
        lambda_od=args.lambda_od,
        seed=args.seed,
        run_name=args.run_name,
        epochs_override=args.epochs,
        patience=args.patience,
        min_epochs=args.min_epochs,
        diag_samples=args.diag_samples
    )

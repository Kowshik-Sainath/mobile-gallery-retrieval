"""
PART P-6: Loss-Weighting Disentanglement Ablation Experiment
============================================================
Ablates lambda_OD in {1.0, 0.1, 0.0} under strictly identical random seeds
and identical batch sampling sequences to determine if L_OD magnitude dominance
hurts retrieval performance on Test-1K.

Configs:
  (a) lambda_OD = 1.0 (unweighted sum -- reproduces current run)
  (b) lambda_OD = 0.1 (reweighted detection loss)
  (c) lambda_OD = 0.0 (L_OD gradient fully disabled)
"""

import os
import sys
import json
import time
from typing import Dict, Any

import torch

sys.path.insert(0, os.path.abspath("."))
from CSTBIR.models.stnet import STNet
from CSTBIR.train import train_stnet
from CSTBIR.run_full_evaluation import evaluate_split

def run_ablation():
    print("=" * 80)
    print("PART P-6: LAMBDA_OD LOSS-WEIGHTING DISENTANGLEMENT ABLATION")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    configs = [
        {"name": "lambda_1.0", "lambda_od": 1.0, "desc": "lambda_OD = 1.0 (Unweighted Baseline)"},
        {"name": "lambda_0.1", "lambda_od": 0.1, "desc": "lambda_OD = 0.1 (Down-Weighted 10x)"},
        {"name": "lambda_0.0", "lambda_od": 0.0, "desc": "lambda_OD = 0.0 (Detection Loss Disabled)"},
    ]
    
    results = {}
    fixed_seed = 42
    
    for cfg in configs:
        name = cfg["name"]
        lam = cfg["lambda_od"]
        desc = cfg["desc"]
        
        print("\n" + "#" * 80)
        print(f"STARTING CONFIG: {desc}")
        print(f"Config Name: {name} | Seed: {fixed_seed} | lambda_od: {lam}")
        print("#" * 80)
        
        t0 = time.time()
        # 1. Train under fixed seed
        history, ckpt_path = train_stnet(
            config_path="CSTBIR/configs/stnet_train.yaml",
            lambda_od=lam,
            seed=fixed_seed,
            run_name=name
        )
        train_time = time.time() - t0
        print(f"\n[Training for {name} finished in {train_time:.1f}s]")
        
        # 2. Evaluate on Test-1K (val split)
        print(f"\n[Evaluating {name} on Test-1K...]")
        t_eval = time.time()
        
        eval_model = STNet(
            num_classes=258,
            device=device,
            pretrained_sketch=True,
            sketch_encoder_ckpt="CSTBIR/checkpoints/sketch_encoder_quickdraw_adapted.pt"
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        eval_model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        
        metrics_1k = evaluate_split(eval_model, split_name="val")
        eval_time = time.time() - t_eval
        print(f"[Evaluation on Test-1K finished in {eval_time:.1f}s]")
        
        # Free memory
        del eval_model
        del ckpt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        results[name] = {
            "lambda_od": lam,
            "description": desc,
            "seed": fixed_seed,
            "checkpoint": ckpt_path,
            "train_time_sec": train_time,
            "eval_time_sec": eval_time,
            "history": history,
            "metrics_1k": metrics_1k
        }
        
        # Save intermediate results
        with open("CSTBIR/p6_ablation_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            
    # -------------------------------------------------------------
    # 3. Print Comprehensive Side-by-Side Comparison
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print("PART P-6: ABLATION RESULTS SUMMARY (TEST-1K)")
    print("=" * 90)
    print(f"{'Metric':<10} | {'Published':>10} | {'Zero-Shot':>10} | {'Bugged Bbox':>12} | {'lam_OD=1.0':>10} | {'lam_OD=0.1':>10} | {'lam_OD=0.0':>10}")
    print("-" * 90)
    
    paper_1k = {'R@10': 73.7, 'R@20': 80.6, 'R@50': 89.4, 'R@100': 93.5, 'MdR': 3.0}
    zeroshot_1k = {'R@10': 1.0, 'R@20': 2.0, 'R@50': 4.6, 'R@100': 10.1, 'MdR': 509.0}
    bugged_1k = {'R@10': 1.3, 'R@20': 2.7, 'R@50': 6.7, 'R@100': 12.7, 'MdR': 444.5}
    
    m_1_0 = results["lambda_1.0"]["metrics_1k"]
    m_0_1 = results["lambda_0.1"]["metrics_1k"]
    m_0_0 = results["lambda_0.0"]["metrics_1k"]
    
    for m in ['R@10', 'R@20', 'R@50', 'R@100', 'MdR']:
        p_str = f"{paper_1k[m]:.1f}"
        zs_str = f"{zeroshot_1k[m]:.1f}"
        bg_str = f"{bugged_1k[m]:.1f}"
        v10 = f"{m_1_0[m]:.2f}"
        v01 = f"{m_0_1[m]:.2f}"
        v00 = f"{m_0_0[m]:.2f}"
        print(f"{m:<10} | {p_str:>10} | {zs_str:>10} | {bg_str:>12} | {v10:>10} | {v01:>10} | {v00:>10}")
        
    print("=" * 90)
    
    print("\n--- Multi-Loss Epoch 3 Values ---")
    print(f"{'Config':<20} | {'L_total':>10} | {'L_CT':>10} | {'L_OD (raw)':>12} | {'L_OD (wtd)':>12} | {'L_SR':>10}")
    print("-" * 80)
    for name in ["lambda_1.0", "lambda_0.1", "lambda_0.0"]:
        h = results[name]["history"]
        lt = h['loss_total'][-1]
        lct = h['loss_ct'][-1]
        lod = h['loss_od'][-1]
        lodw = h['loss_od_weighted'][-1]
        lsr = h['loss_sr'][-1]
        print(f"{name:<20} | {lt:10.4f} | {lct:10.4f} | {lod:12.4f} | {lodw:12.4f} | {lsr:10.4f}")
    print("=" * 80)

if __name__ == "__main__":
    run_ablation()

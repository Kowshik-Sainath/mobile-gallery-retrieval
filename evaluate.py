"""
Evaluation Protocol Script for FS-COCO T+SBIR System.
Reports Recall@1, Recall@5, Recall@10, and NDCG@10 across modal combinations (Text-Only, Sketch-Only, Composite).
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.fscoco_dataset import FSCOCODataset
from torch.utils.data import DataLoader
from src.models.composite_model import TSBIRCompositeModel
from src.utils.metrics import compute_recall_at_k, compute_ndcg_at_k

def evaluate_retrieval(model, test_loader, device="cuda"):
    model.eval()
    
    sketch_embeds_list = []
    text_embeds_list = []
    composite_embeds_list = []
    photo_embeds_list = []

    print("Extracting test set embeddings...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting Features"):
            sketch = batch['sketch'].to(device)
            captions = batch['caption']
            photo = batch['photo'].to(device)

            e_sketch = model.encode_sketch(sketch)
            e_text = model.encode_text(captions)
            e_photo = model.encode_photo(photo)

            raw_composite = torch.cat([e_sketch, e_text], dim=-1)
            e_composite = F.normalize(model.composite_fusion(raw_composite), dim=-1)

            sketch_embeds_list.append(e_sketch.cpu().numpy())
            text_embeds_list.append(e_text.cpu().numpy())
            composite_embeds_list.append(e_composite.cpu().numpy())
            photo_embeds_list.append(e_photo.cpu().numpy())

    sketch_embeds = np.concatenate(sketch_embeds_list, axis=0)
    text_embeds = np.concatenate(text_embeds_list, axis=0)
    composite_embeds = np.concatenate(composite_embeds_list, axis=0)
    photo_embeds = np.concatenate(photo_embeds_list, axis=0)

    # Cosine Similarity Matrices
    sim_text = np.dot(text_embeds, photo_embeds.T)
    sim_sketch = np.dot(sketch_embeds, photo_embeds.T)
    sim_composite = np.dot(composite_embeds, photo_embeds.T)

    results = {}
    
    # 1. Text-only evaluation
    r_text = compute_recall_at_k(sim_text)
    ndcg_text = compute_ndcg_at_k(sim_text)
    results['Text-Only'] = {**r_text, 'NDCG@10': ndcg_text}

    # 2. Sketch-only evaluation
    r_sketch = compute_recall_at_k(sim_sketch)
    ndcg_sketch = compute_ndcg_at_k(sim_sketch)
    results['Sketch-Only'] = {**r_sketch, 'NDCG@10': ndcg_sketch}

    # 3. Composite (Sketch + Text) Late Fusion evaluation
    sim_composite_late = 0.5 * sim_sketch + 0.5 * sim_text
    r_comp_late = compute_recall_at_k(sim_composite_late)
    ndcg_comp_late = compute_ndcg_at_k(sim_composite_late)
    results['Composite (Late Fusion)'] = {**r_comp_late, 'NDCG@10': ndcg_comp_late}

    # 4. Composite (STNet Fused Projection) evaluation
    r_comp = compute_recall_at_k(sim_composite)
    ndcg_comp = compute_ndcg_at_k(sim_composite)
    results['Composite (STNet Projection)'] = {**r_comp, 'NDCG@10': ndcg_comp}

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate T+SBIR Model on FS-COCO Test Split")
    parser.add_argument("--data_dir", type=str, default="fscoco", help="Path to FS-COCO root directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Evaluation on device: {device}")

    # Load Test Data
    test_dataset = FSCOCODataset(args.data_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded FS-COCO Test Split: {len(test_dataset)} test samples.")

    # Load Model
    model = TSBIRCompositeModel(device=device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint weights from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint.get('full_state_dict', checkpoint), strict=False)
    
    model.to(device)

    # Evaluate
    results = evaluate_retrieval(model, test_loader, device=device)

    # Print Summary Table
    print("\n" + "="*65)
    print(f"{'Evaluation Strategy':<25} | {'R@1':<7} | {'R@5':<7} | {'R@10':<7} | {'NDCG@10':<7}")
    print("-" * 65)
    for mode, metrics in results.items():
        print(f"{mode:<25} | {metrics['R@1']:<7.2f} | {metrics['R@5']:<7.2f} | {metrics['R@10']:<7.2f} | {metrics['NDCG@10']:<7.2f}")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()

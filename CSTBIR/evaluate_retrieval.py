import sys
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath("."))
from CSTBIR.models.stnet import STNet

def compute_retrieval_metrics(ranks: List[int]) -> Dict[str, float]:
    """
    Computes Recall@10, Recall@20, Recall@50, Recall@100, and Median Rank (MdR)
    matching the paper's Table 3 evaluation protocol.
    Ranks are 1-indexed.
    """
    ranks_arr = np.array(ranks)
    total_queries = len(ranks_arr)
    
    r10 = float(np.mean(ranks_arr <= 10) * 100.0)
    r20 = float(np.mean(ranks_arr <= 20) * 100.0)
    r50 = float(np.mean(ranks_arr <= 50) * 100.0)
    r100 = float(np.mean(ranks_arr <= 100) * 100.0)
    mdr = float(np.median(ranks_arr))
    
    return {
        'R@10': r10,
        'R@20': r20,
        'R@50': r50,
        'R@100': r100,
        'MdR': mdr,
        'total_queries': total_queries
    }

def evaluate_gallery_ranking(
    model: STNet,
    query_texts: torch.Tensor,       # (Q, 77)
    query_sketches: torch.Tensor,    # (Q, 768)
    gallery_images: torch.Tensor,    # (K, 3, 224, 224)
    ground_truth_indices: List[int], # (Q,) index in [0, K-1] of the true matching image
    batch_size: int = 64
) -> Dict[str, float]:
    """
    Evaluates full gallery retrieval by computing similarity between each query and
    all K gallery images, sorting in descending order, and computing recall / MdR.
    """
    model.eval()
    device = model.device
    Q = query_texts.shape[0]
    K = gallery_images.shape[0]
    assert len(ground_truth_indices) == Q
    
    # 1. Compute similarity matrix
    with torch.no_grad():
        sim_matrix = model.compute_query_image_similarity(query_texts, query_sketches, gallery_images) # (Q, K)
        
    ranks = []
    for q in range(Q):
        gt_idx = ground_truth_indices[q]
        sims_q = sim_matrix[q].cpu().numpy()
        
        # Sort in descending order of similarity
        # argsort gives ascending order, so flip
        sorted_indices = np.argsort(-sims_q)
        
        # Find 1-indexed rank of gt_idx
        gt_rank = int(np.where(sorted_indices == gt_idx)[0][0]) + 1
        ranks.append(gt_rank)
        
    metrics = compute_retrieval_metrics(ranks)
    return metrics

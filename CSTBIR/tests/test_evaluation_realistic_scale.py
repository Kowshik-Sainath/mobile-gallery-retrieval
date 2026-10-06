import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath("."))
from CSTBIR.models.stnet import STNet
from CSTBIR.evaluate_retrieval import compute_retrieval_metrics, evaluate_gallery_ranking

def test_evaluation_realistic_scale():
    print("=" * 75)
    print("PART P-3 (Task 2): Gallery Retrieval Evaluation at Realistic Scale (K=100)")
    print("=" * 75)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # -----------------------------------------------------------------
    # Test 1: Realistic Scale Gallery Ranking (K=100) with Known Hand-Assigned Ranks
    # -----------------------------------------------------------------
    print("\n--- Test 1: Hand-Assigned Target Ranks in K=100 Gallery ---")
    print("In a 10-image gallery (the old test), all items have rank <= 10, so R@10 was trivially 100%.")
    print("Here we scale the gallery to K=100 items and test 10 queries with exact target ranks:")
    
    K = 100
    Q = 10
    
    # Carefully designed ground-truth ranks spanning key boundary conditions:
    # Query 0: rank 1   (<= 10 -> passes R@10, R@20, R@50, R@100)
    # Query 1: rank 5   (<= 10 -> passes R@10, R@20, R@50, R@100)
    # Query 2: rank 10  (<= 10 -> boundary! passes R@10, R@20, R@50, R@100)
    # Query 3: rank 11  (> 10, <= 20 -> FAILS R@10! boundary! passes R@20, R@50, R@100)
    # Query 4: rank 20  (> 10, <= 20 -> FAILS R@10, boundary! passes R@20, R@50, R@100)
    # Query 5: rank 21  (> 20, <= 50 -> FAILS R@10, FAILS R@20, boundary! passes R@50, R@100)
    # Query 6: rank 50  (> 20, <= 50 -> FAILS R@10, FAILS R@20, boundary! passes R@50, R@100)
    # Query 7: rank 51  (> 50, <= 100 -> FAILS R@10, FAILS R@20, FAILS R@50, boundary! passes R@100)
    # Query 8: rank 85  (> 50, <= 100 -> FAILS R@10, FAILS R@20, FAILS R@50, passes R@100)
    # Query 9: rank 100 (> 50, <= 100 -> boundary! passes R@100)
    target_ranks = [1, 5, 10, 11, 20, 21, 50, 51, 85, 100]
    
    # Hand-computed expected values:
    # R@10:  queries 0, 1, 2 (ranks 1, 5, 10) -> 3/10 = 30.0%
    # R@20:  queries 0..4 (ranks 1, 5, 10, 11, 20) -> 5/10 = 50.0%
    # R@50:  queries 0..6 (ranks 1, 5, 10, 11, 20, 21, 50) -> 7/10 = 70.0%
    # R@100: all 10 queries -> 10/10 = 100.0%
    # MdR:   median of [1, 5, 10, 11, 20, 21, 50, 51, 85, 100] = (20 + 21)/2 = 20.5
    expected_r10 = 30.0
    expected_r20 = 50.0
    expected_r50 = 70.0
    expected_r100 = 100.0
    expected_mdr = 20.5
    
    print(f"Hand-Computed Expectations for ranks {target_ranks}:")
    print(f"  Expected R@10:  {expected_r10:.1f}%")
    print(f"  Expected R@20:  {expected_r20:.1f}%")
    print(f"  Expected R@50:  {expected_r50:.1f}%")
    print(f"  Expected R@100: {expected_r100:.1f}%")
    print(f"  Expected MdR:   {expected_mdr:.1f}")
    
    # Build synthetic similarity matrix (Q, K) that produces EXACTLY these ranks
    # Choose ground truth index for each query (e.g. gt_idx = (q * 7 + 3) % K)
    gt_indices = [(q * 7 + 3) % K for q in range(Q)]
    synthetic_sims = torch.zeros(Q, K, device=device)
    
    for q in range(Q):
        desired_rank = target_ranks[q] # 1-indexed
        gt_idx = gt_indices[q]
        
        # Fill row with random or decreasing values, then arrange so gt_idx is at desired_rank
        # Give all items similarity scores from 0.0 to 1.0
        scores = torch.linspace(0.1, 0.9, K, device=device)
        # In descending sort, score at rank 1 is highest (index 0 of sorted), rank K is lowest (index K-1)
        # So we place scores such that exactly (desired_rank - 1) images have strictly higher score than gt_idx
        # Let gt_idx have score S_gt.
        # We give (desired_rank - 1) other images scores > S_gt, and the rest < S_gt.
        s_gt = 0.50
        synthetic_sims[q, gt_idx] = s_gt
        
        other_indices = [i for i in range(K) if i != gt_idx]
        np.random.seed(42 + q)
        np.random.shuffle(other_indices)
        
        higher_indices = other_indices[:desired_rank - 1]
        lower_indices = other_indices[desired_rank - 1:]
        
        # Assign higher scores in (0.51, 0.99)
        if len(higher_indices) > 0:
            high_vals = torch.linspace(0.55, 0.95, len(higher_indices), device=device)
            for idx, val in zip(higher_indices, high_vals):
                synthetic_sims[q, idx] = val
                
        # Assign lower scores in (0.01, 0.49)
        if len(lower_indices) > 0:
            low_vals = torch.linspace(0.05, 0.45, len(lower_indices), device=device)
            for idx, val in zip(lower_indices, low_vals):
                synthetic_sims[q, idx] = val
                
    # Now evaluate ranking through evaluate_gallery_ranking logic
    # Create a mock model whose compute_query_image_similarity returns our synthetic_sims
    class MockRetrievalModel(nn.Module):
        def __init__(self, sim_matrix):
            super().__init__()
            self.sim_matrix = sim_matrix
            self.device = sim_matrix.device
        def compute_query_image_similarity(self, q_t, q_s, g_i):
            return self.sim_matrix
            
    mock_model = MockRetrievalModel(synthetic_sims)
    
    # Run evaluation
    dummy_q_t = torch.zeros(Q, 77, dtype=torch.long, device=device)
    dummy_q_s = torch.zeros(Q, 768, device=device)
    dummy_g_i = torch.zeros(K, 3, 224, 224, device=device)
    
    metrics = evaluate_gallery_ranking(
        model=mock_model,
        query_texts=dummy_q_t,
        query_sketches=dummy_q_s,
        gallery_images=dummy_g_i,
        ground_truth_indices=gt_indices
    )
    
    print("\nCalculated Metrics at Scale K=100:")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v}")
        
    assert metrics['R@10'] == expected_r10, f"R@10 mismatch: got {metrics['R@10']}, expected {expected_r10}"
    assert metrics['R@20'] == expected_r20, f"R@20 mismatch: got {metrics['R@20']}, expected {expected_r20}"
    assert metrics['R@50'] == expected_r50, f"R@50 mismatch: got {metrics['R@50']}, expected {expected_r50}"
    assert metrics['R@100'] == expected_r100, f"R@100 mismatch: got {metrics['R@100']}, expected {expected_r100}"
    assert metrics['MdR'] == expected_mdr, f"MdR mismatch: got {metrics['MdR']}, expected {expected_mdr}"
    print("\nPASS: All metrics at K=100 gallery scale EXACTLY matched hand-calculated expectations.")
    print("PASS: Verified that queries with rank > 10 genuinely FAIL R@10 (R@10 = 30.0% != 100%).")
    
    # -----------------------------------------------------------------
    # Test 2: Inverted Sorting Sensitivity Check (Failure Detection)
    # -----------------------------------------------------------------
    print("\n--- Test 2: Inverted Sorting Sensitivity Check ---")
    # If the ranking logic had an inversion bug (e.g. argsort ascending instead of descending),
    # the ranks would invert (rank r becomes K - r + 1), and R@10 would crash to 0%
    inverted_ranks = []
    for q in range(Q):
        gt_idx = gt_indices[q]
        sims_q = synthetic_sims[q].cpu().numpy()
        # Buggy ascending sort
        buggy_sorted = np.argsort(sims_q) # ascending order (lowest similarity first)
        buggy_rank = int(np.where(buggy_sorted == gt_idx)[0][0]) + 1
        inverted_ranks.append(buggy_rank)
        
    buggy_metrics = compute_retrieval_metrics(inverted_ranks)
    print(f"Buggy Ascending Sort R@10: {buggy_metrics['R@10']:.1f}% (Expected failure, < 10%)")
    assert buggy_metrics['R@10'] < expected_r10, "Evaluation failed to detect inverted sorting order!"
    print("PASS: Confirmed that ranking test detects sorting order bugs.")
    
    # -----------------------------------------------------------------
    # Test 3: Real STNet compute_query_image_similarity at K=100 Gallery Scale
    # -----------------------------------------------------------------
    print("\n--- Test 3: Real STNet compute_query_image_similarity Execution (K=100 Gallery) ---")
    stnet = STNet(num_classes=258, device=device, pretrained_sketch=False).to(device)
    stnet.eval()
    
    # Test tensor dimensions: Q=5 queries vs K=100 gallery images
    real_Q = 5
    real_K = 100
    
    real_query_texts = torch.randint(1, 1000, (real_Q, 77), device=device)
    real_query_sketches = torch.randn(real_Q, 768, device=device)
    real_gallery_images = torch.randn(real_K, 3, 224, 224, device=device)
    real_gt_indices = [3, 27, 45, 78, 92]
    
    with torch.no_grad():
        sim_mat = stnet.compute_query_image_similarity(
            text_tokens=real_query_texts,
            sketch_embeds=real_query_sketches,
            gallery_images=real_gallery_images
        )
        
    print(f"Similarity matrix output shape: {sim_mat.shape} (Expected: ({real_Q}, {real_K}))")
    assert sim_mat.shape == (real_Q, real_K), f"Expected ({real_Q}, {real_K}), got {sim_mat.shape}"
    assert not torch.isnan(sim_mat).any(), "NaN found in similarity matrix!"
    assert not torch.isinf(sim_mat).any(), "Inf found in similarity matrix!"
    
    # Run full ranking evaluation on this real similarity matrix
    real_metrics = evaluate_gallery_ranking(
        model=stnet,
        query_texts=real_query_texts,
        query_sketches=real_query_sketches,
        gallery_images=real_gallery_images,
        ground_truth_indices=real_gt_indices
    )
    
    print("Real STNet Gallery Ranking Metrics:")
    for k, v in real_metrics.items():
        print(f"  {k:15s}: {v}")
        
    assert real_metrics['total_queries'] == real_Q
    assert 0.0 <= real_metrics['R@10'] <= 100.0
    assert 1.0 <= real_metrics['MdR'] <= real_K
    print("\n[PART P-3 TASK 2: PASS] Gallery retrieval evaluation verified at K=100 scale.")

if __name__ == "__main__":
    test_evaluation_realistic_scale()

import sys
import os
import torch

sys.path.insert(0, os.path.abspath("."))
from CSTBIR.models.stnet import STNet
from CSTBIR.evaluate_retrieval import compute_retrieval_metrics, evaluate_gallery_ranking

def test_evaluation_protocol():
    print("=" * 70)
    print("STEP 4: Retrieval Evaluation Protocol Verification (Table 3 Metrics)")
    print("=" * 70)
    
    # -----------------------------------------------------------------
    # Test 1: Math Verification of Recall@K and Median Rank
    # -----------------------------------------------------------------
    print("\n--- Test 1: Exact Math Verification on Known Ranks ---")
    # 10 queries with known ranks: [1, 3, 5, 8, 12, 18, 30, 45, 75, 120]
    # R@10: ranks <= 10: 4 queries (1, 3, 5, 8) -> 4/10 = 40.0%
    # R@20: ranks <= 20: 6 queries (+ 12, 18) -> 6/10 = 60.0%
    # R@50: ranks <= 50: 8 queries (+ 30, 45) -> 8/10 = 80.0%
    # R@100: ranks <= 100: 9 queries (+ 75) -> 9/10 = 90.0%
    # MdR: median of [1, 3, 5, 8, 12, 18, 30, 45, 75, 120] -> (12 + 18)/2 = 15.0
    known_ranks = [1, 3, 5, 8, 12, 18, 30, 45, 75, 120]
    metrics = compute_retrieval_metrics(known_ranks)
    
    print(f"R@10:  {metrics['R@10']:.1f}% (Expected: 40.0%)")
    print(f"R@20:  {metrics['R@20']:.1f}% (Expected: 60.0%)")
    print(f"R@50:  {metrics['R@50']:.1f}% (Expected: 80.0%)")
    print(f"R@100: {metrics['R@100']:.1f}% (Expected: 90.0%)")
    print(f"MdR:   {metrics['MdR']:.1f} (Expected: 15.0)")
    
    assert metrics['R@10'] == 40.0
    assert metrics['R@20'] == 60.0
    assert metrics['R@50'] == 80.0
    assert metrics['R@100'] == 90.0
    assert metrics['MdR'] == 15.0
    print("Exact retrieval metrics math verified.")
    
    # -----------------------------------------------------------------
    # Test 2: Full Gallery Ranking Pipeline with STNet
    # -----------------------------------------------------------------
    print("\n--- Test 2: Testing Full Gallery Ranking with STNet ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = STNet(device=device, pretrained_sketch=False)
    
    Q = 4
    K = 10
    query_texts = torch.randint(1, 1000, (Q, 77), device=device)
    query_sketches = torch.randn(Q, 768, device=device)
    gallery_images = torch.randn(K, 3, 224, 224, device=device)
    # Ground truth targets: query 0 matches image 2, query 1 matches image 5, query 2 matches image 0, query 3 matches image 7
    gt_targets = [2, 5, 0, 7]
    
    eval_metrics = evaluate_gallery_ranking(
        model=model,
        query_texts=query_texts,
        query_sketches=query_sketches,
        gallery_images=gallery_images,
        ground_truth_indices=gt_targets
    )
    
    print(f"Gallery evaluation metrics on {Q} queries vs {K} gallery images:")
    for k, v in eval_metrics.items():
        print(f"  {k}: {v}")
        
    assert eval_metrics['total_queries'] == Q
    assert 0.0 <= eval_metrics['R@10'] <= 100.0
    print("\n[STEP 4: PASS] Gallery retrieval evaluation protocol verified.")

if __name__ == "__main__":
    test_evaluation_protocol()

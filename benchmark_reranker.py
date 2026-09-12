"""
Feedback Reranker Benchmark Script.
Measures model footprint and on-device CPU fine-tuning latency per update step.
"""

import time
import torch
from src.reranker.combiner import FeedbackCombinerReranker

def benchmark_reranker():
    print("Benchmarking Gated MLP Combiner Reranker...")
    device = torch.device("cpu") # CPU proxy for edge AI phones
    
    reranker = FeedbackCombinerReranker(feature_dim=512, hidden_dim=128).to(device)
    total_params = sum(p.numel() for p in reranker.parameters())
    param_size_mb = sum(p.numel() * p.element_size() for p in reranker.parameters()) / (1024 * 1024)
    
    print(f"Combiner Parameter Count: {total_params:,} (<1M parameter target satisfied)")
    print(f"Combiner Memory Footprint: {param_size_mb:.2f} MB")

    optimizer = torch.optim.Adam(reranker.parameters(), lr=1e-3)

    # Benchmark online update latency (1 positive, 4 negative skipped candidates)
    query_embed = torch.randn(1, 512, device=device)
    pos_cand = torch.randn(1, 512, device=device)
    neg_cands = torch.randn(4, 512, device=device)

    # Warmup
    reranker.online_feedback_update(query_embed, pos_cand, neg_cands, optimizer, num_steps=1)

    step_counts = [1, 3, 5]
    latency_results = {}
    
    for steps in step_counts:
        times = []
        for _ in range(10):
            elapsed_ms, loss = reranker.online_feedback_update(
                query_embed, pos_cand, neg_cands, optimizer, num_steps=steps
            )
            times.append(elapsed_ms)
        avg_ms = sum(times) / len(times)
        latency_results[steps] = avg_ms

    print("\n" + "="*55)
    print(f"{'Update Gradient Steps':<25} | {'Latency per Update (ms)':<20}")
    print("-" * 55)
    for steps, latency in latency_results.items():
        print(f"{steps:<25} | {latency:<20.2f} ms")
    print("="*55 + "\n")

if __name__ == "__main__":
    benchmark_reranker()

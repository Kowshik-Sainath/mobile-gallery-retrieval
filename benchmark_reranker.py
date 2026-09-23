"""
Feedback Reranker Benchmark Script for CLIP4Cir FeedbackComposedRetriever.
Measures model parameter count, memory footprint, CPU inference latency,
and on-device CPU fine-tuning latency per update step.
"""

import time
import torch
from src.reranker.combiner import FeedbackComposedRetriever


def benchmark_reranker():
    print("=" * 60)
    print("Benchmarking CLIP4Cir FeedbackComposedRetriever on CPU...")
    print("=" * 60)
    device = torch.device("cpu")  # CPU proxy for edge smartphone inference

    combiner = FeedbackComposedRetriever(feature_dim=512, hidden_dim=256).to(device)
    total_params = sum(p.numel() for p in combiner.parameters())
    param_size_mb = sum(p.numel() * p.element_size() for p in combiner.parameters()) / (1024 * 1024)

    print(f"Combiner Parameter Count: {total_params:,} (<1M budget: {'PASS' if total_params < 1_000_000 else 'FAIL'})")
    print(f"Combiner Memory Footprint: {param_size_mb:.2f} MB")

    # Benchmark forward composition latency on CPU
    sh_cand = torch.randn(1, 512, device=device)
    ref_q   = torch.randn(1, 512, device=device)

    # Warmup
    for _ in range(10):
        _ = combiner(sh_cand, ref_q)

    fwd_times = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = combiner(sh_cand, ref_q)
        fwd_times.append((time.perf_counter() - t0) * 1000.0)
    avg_fwd_ms = sum(fwd_times) / len(fwd_times)
    print(f"CPU Forward Query Composition Latency: {avg_fwd_ms:.3f} ms")

    # Benchmark online local SGD update latency on CPU (1 positive, 4 negative skipped candidates)
    pos_target = torch.randn(1, 512, device=device)
    neg_targets = torch.randn(4, 512, device=device)
    optimizer = torch.optim.Adam(combiner.parameters(), lr=1e-3)

    # Warmup
    combiner.online_feedback_update(sh_cand, ref_q, pos_target, neg_targets, optimizer, num_steps=1)

    step_counts = [1, 3, 5]
    latency_results = {}

    for steps in step_counts:
        times = []
        for _ in range(10):
            elapsed_ms, loss = combiner.online_feedback_update(
                sh_cand, ref_q, pos_target, neg_targets, optimizer, num_steps=steps
            )
            times.append(elapsed_ms)
        avg_ms = sum(times) / len(times)
        latency_results[steps] = avg_ms

    print("\n" + "=" * 55)
    print(f"{'Update Gradient Steps':<25} | {'Latency per Update (ms)':<20}")
    print("-" * 55)
    for steps, latency in latency_results.items():
        print(f"{steps:<25} | {latency:<20.2f} ms")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    benchmark_reranker()

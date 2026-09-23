"""
CLIP4Cir-style Query Composition Combiner Module for Feedback-Driven Edge Retrieval.
Ported and adapted from ABaldrati/CLIP4Cir (CVPR 2022).

Instead of a pointwise candidate scorer, this module composes:
  - shown_candidate_embed: e_shown (B, D) — e.g. the image currently seen / rejected / tapped
  - refined_query_embed:   e_query (B, D) — e.g. the text/sketch corrective modification or composite query
into a newly composed query vector in the gallery embedding space:
  combined = L2Norm((1 - lambda) * e_shown + lambda * e_query + v)
where:
  lambda in (0, 1)^D is a dynamic feature-wise gate (predicted via MLP + Sigmoid)
  v in R^D is a directional residual offset (predicted via MLP)
  Total parameter count: <1M parameters (<4MB FP32), strictly mobile edge friendly.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedbackComposedRetriever(nn.Module):
    """
    CLIP4Cir Gated Query Composition Combiner (<1M params).
    Composes (shown_candidate, refined_query) -> combined_query for re-searching the gallery.
    """
    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim  = hidden_dim

        # Gate MLP: predicts per-dimension combination weight lambda in (0, 1)^D
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid(),
        )

        # Residual offset MLP: predicts directional correction v in R^D
        self.res_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(
        self,
        shown_candidate: torch.Tensor,   # (B, D)
        refined_query: torch.Tensor,     # (B, D)
    ) -> torch.Tensor:
        """
        Combines shown candidate and refined query into a new query embedding.
        Returns:
            combined_query: (B, D), L2-normalized.
        """
        # Ensure 2D tensors
        if shown_candidate.dim() == 1:
            shown_candidate = shown_candidate.unsqueeze(0)
        if refined_query.dim() == 1:
            refined_query = refined_query.unsqueeze(0)

        # Concat along channel dimension
        fused_input = torch.cat([shown_candidate, refined_query], dim=-1)   # (B, 2*D)

        gate = self.gate_mlp(fused_input)   # (B, D) in (0, 1)
        res  = self.res_mlp(fused_input)    # (B, D)

        # Convex interpolation + residual offset
        combined = (1.0 - gate) * shown_candidate + gate * refined_query + res
        return F.normalize(combined, dim=-1)

    def online_feedback_update(
        self,
        shown_candidate: torch.Tensor,
        refined_query: torch.Tensor,
        positive_target: torch.Tensor,
        negative_targets: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        num_steps: int = 3,
        temperature: float = 0.07,
    ):
        """
        Performs incremental on-device local SGD fine-tuning using user feedback.
        Args:
            shown_candidate: (1, D) embedding of candidate photo shown to user
            refined_query:   (1, D) embedding of the refined query
            positive_target: (1, D) embedding of target photo tapped by user
            negative_targets: (K, D) embeddings of photos rejected/skipped
            optimizer: torch optimizer instance (e.g. Adam with lr=1e-3)
            num_steps: number of gradient updates
            temperature: InfoNCE temperature
        Returns:
            elapsed_ms: total update latency in milliseconds
            last_loss: final step loss value
        """
        self.train()
        start_time = time.perf_counter()

        for _ in range(num_steps):
            optimizer.zero_grad()
            # Predict composed query: (1, D)
            combined = self(shown_candidate, refined_query)

            # Bank of candidates: positive followed by negatives -> (1 + K, D)
            all_targets = torch.cat([positive_target, negative_targets], dim=0)

            # Cosine logits: (1, 1 + K)
            logits = torch.matmul(combined, all_targets.T) / temperature
            labels = torch.zeros(1, dtype=torch.long, device=logits.device)   # positive is index 0

            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.eval()
        return elapsed_ms, loss.item()


# Backward compatibility alias
FeedbackCombinerReranker = FeedbackComposedRetriever

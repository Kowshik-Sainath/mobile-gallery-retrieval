"""
CLIP4Cir-style Gated MLP Combiner Reranker Module.
Ported from ABaldrati/CLIP4Cir (CVPR 2022).
Fuses query embedding, candidate feature, and user interaction feedback signal for on-device reranking.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedbackCombinerReranker(nn.Module):
    """
    Gated MLP Combiner Module (<1M parameters) for Feedback-Driven Edge Reranking.
    """
    def __init__(self, feature_dim=512, hidden_dim=128):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Projection heads for query and candidate + feedback history
        self.fc_query = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.fc_candidate = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )

        # Gated fusion layer
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
        # Output scoring layer
        self.score_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, query_embed, candidate_embed, feedback_history=None):
        """
        Args:
            query_embed: (B, D)
            candidate_embed: (B, D)
            feedback_history: (B, D) optional user feedback vector (e.g. positive tap - negative skip)
        Returns:
            fused_score: (B, 1) refined similarity score
        """
        q_proj = self.fc_query(query_embed)
        c_proj = self.fc_candidate(candidate_embed)

        if feedback_history is not None:
            q_proj = q_proj + feedback_history

        # Gate mechanism
        g = self.gate(torch.cat([q_proj, c_proj], dim=-1))
        fused = g * q_proj + (1.0 - g) * c_proj
        fused_norm = F.normalize(fused, dim=-1)

        scores = self.score_head(fused_norm)
        return scores

    def online_feedback_update(self, query_embed, positive_candidate, negative_candidates, optimizer, num_steps=3):
        """
        Performs incremental on-device fine-tuning step using user tap (positive) and skip (negative) feedback.
        Returns latency in milliseconds.
        """
        self.train()
        start_time = time.time()
        
        # Construct small batch: 1 positive candidate + N negative candidates
        candidates = torch.cat([positive_candidate, negative_candidates], dim=0) # (1+N, D)
        queries = query_embed.expand(candidates.shape[0], -1) # (1+N, D)
        
        labels = torch.zeros(candidates.shape[0], device=query_embed.device)
        labels[0] = 1.0 # Positive candidate target score = 1.0, negatives = 0.0

        for _ in range(num_steps):
            optimizer.zero_grad()
            scores = self(queries, candidates).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            loss.backward()
            optimizer.step()

        elapsed_ms = (time.time() - start_time) * 1000.0
        self.eval()
        return elapsed_ms, loss.item()

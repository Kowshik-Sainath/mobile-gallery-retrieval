"""
MoCo (Momentum Contrast) Queue for small-batch InfoNCE contrastive learning.

Why MoCo?
---------
Standard InfoNCE with B=16 gives only 15 negatives per sample, producing a loss
space log(16) ≈ 2.77 where random guessing already achieves ~2.34. The model
learns to separate only the 15 in-batch negatives, then catastrophically overfits.

MoCo decouples the number of negatives from mini-batch size by maintaining a FIFO
queue of K=4096 momentum-encoded photo embeddings as a persistent negative bank.
This pushes the random-guessing baseline to log(4096+1) ≈ 8.3, forcing the model
to learn genuinely discriminative representations.

Architecture:
  - Queue: FIFO buffer of K normalized photo embeddings, shape (D, K)
  - Momentum encoder: EMA copy of the base photo encoder (no LoRA), updated each step:
      θ_k ← m·θ_k + (1-m)·θ_q    (momentum=0.999)
  - Loss: InfoNCE where positives are current-batch (composite_query, photo_key)
          and negatives are the entire queue.

Reference: He et al., "Momentum Contrast for Unsupervised Visual Representation
           Learning", CVPR 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCoQueue(nn.Module):
    """
    Maintains a FIFO queue of momentum-encoded photo embeddings.

    Usage (inside composite model forward):
        # 1. Encode photo with momentum encoder (no grad)
        with torch.no_grad():
            photo_key = momentum_encoder(photo)          # (B, D)
            photo_key = F.normalize(photo_key, dim=-1)

        # 2. Compute InfoNCE loss using queue as negatives
        loss = moco_queue.infonce_loss(composite_query, photo_key)

        # 3. Enqueue current batch (AFTER loss computation)
        moco_queue.dequeue_and_enqueue(photo_key)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        queue_size: int = 4096,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.queue_size = queue_size
        self.temperature = temperature

        # Queue stored as (D, K) for efficient matmul: query (B, D) @ queue (D, K) → (B, K)
        # Initialised with random unit vectors so the queue is immediately valid
        self.register_buffer(
            'queue',
            F.normalize(torch.randn(embed_dim, queue_size), dim=0)
        )
        # Pointer to the next write position
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """
        Enqueue a batch of normalised photo key embeddings and dequeue
        the oldest entries (circular FIFO).

        Args:
            keys: (B, D) — normalised momentum-encoded photo embeddings.
        """
        B = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Handle wrap-around gracefully
        end = ptr + B
        if end <= self.queue_size:
            self.queue[:, ptr:end] = keys.T
        else:
            # Split: fill tail then wrap to head
            tail = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:tail].T
            self.queue[:, :B - tail] = keys[tail:].T

        self.queue_ptr[0] = (ptr + B) % self.queue_size

    def infonce_loss(
        self,
        queries: torch.Tensor,
        positive_keys: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE loss using the current queue as the negative bank.

        For each sample i:
          logit_pos  = (query_i · key_i) / T
          logit_neg_j = (query_i · queue_j) / T  for all j in [0, K)

        The positive logit is placed at index 0; cross-entropy with label=0
        maximises the positive similarity relative to all K queue negatives.

        Args:
            queries:       (B, D) — composite query embeddings (normalised).
            positive_keys: (B, D) — momentum-encoded photo embeddings (normalised).

        Returns:
            scalar loss
        """
        B = queries.shape[0]

        # Positive logit: (B, 1)
        l_pos = torch.bmm(
            queries.unsqueeze(1),            # (B, 1, D)
            positive_keys.unsqueeze(2)       # (B, D, 1)
        ).squeeze(-1)                         # (B, 1)

        # Negative logits from queue: (B, K)
        l_neg = torch.mm(queries, self.queue.clone().detach())   # (B, K)

        # Concatenate: positive first, then K negatives → (B, 1+K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature

        # Labels: the positive is always at index 0
        labels = torch.zeros(B, dtype=torch.long, device=queries.device)

        return F.cross_entropy(logits, labels)

    @property
    def num_negatives(self) -> int:
        """Number of negatives currently available in the queue."""
        return self.queue_size

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, queue_size={self.queue_size}, "
            f"temperature={self.temperature}"
        )

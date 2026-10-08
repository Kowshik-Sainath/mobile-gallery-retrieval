import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveRetrievalLoss(nn.Module):
    """
    Cross-Modal Contrastive Retrieval Loss (L_CT) for STNet (AAAI 2024).
    
    Paper Specification (Section: Contrastive Training (L_CT), Page 4):
        "We adopt a batch-wise contrastive learning strategy akin to CLIP (Radford et al. 2021)
        to facilitate image retrieval. Given a batch of N paired (query, image) samples from
        the train set, we aim to maximize the cosine similarity of the image and query embeddings
        of the N real pairs in the batch while minimizing the cosine similarity of the embeddings
        of the N(N - 1) incorrect pairings... Particularly, we use the InfoNCE loss between
        h^T_{CLS} and h^I_{AVG} to obtain the contrastive loss (L_CT) as done in CLIP."
        
    Implementation Details:
        - Takes query text embedding h^T_{CLS} and sketch-attended image embedding h^I_{AVG}.
        - If embedding dimensions differ (e.g. 512 text vs 768 image), applies a linear projection
          (matching CLIP's visual projection W_proj: 768 -> 512).
        - Computes symmetric InfoNCE with learned logit scale parameter (clamped to max 100).
    """
    def __init__(
        self,
        text_dim: int = 512,
        img_dim: int = 768,
        proj_dim: int = 512,
        init_temperature: float = 0.07,
        learnable_temp: bool = True,
        clamp_min: Optional[float] = 1.0
    ):
        super().__init__()
        self.clamp_min = clamp_min
        # If dimensions differ, project image features to match text retrieval dimension
        if img_dim != proj_dim:
            self.img_proj = nn.Linear(img_dim, proj_dim, bias=False)
        else:
            self.img_proj = nn.Identity()
            
        if text_dim != proj_dim:
            self.txt_proj = nn.Linear(text_dim, proj_dim, bias=False)
        else:
            self.txt_proj = nn.Identity()
            
        # Learned temperature parameter (stored as log_scale as in CLIP)
        init_scale = math.log(1.0 / init_temperature)
        if learnable_temp:
            self.logit_scale = nn.Parameter(torch.tensor([init_scale]))
        else:
            self.register_buffer("logit_scale", torch.tensor([init_scale]))
            
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        h_T_CLS: torch.Tensor,  # (B, text_dim) query text embedding
        h_I_AVG: torch.Tensor   # (B, img_dim) sketch-attended image embedding
    ):
        """
        Returns:
            loss_ct: scalar symmetric InfoNCE loss
            logits_per_text:  (B, B) cosine similarity logits scaled by temperature
            logits_per_image: (B, B)
        """
        B = h_T_CLS.shape[0]
        device = h_T_CLS.device
        
        # 1. Project to common retrieval space (512-dim)
        z_t = self.txt_proj(h_T_CLS)
        z_i = self.img_proj(h_I_AVG)
        
        # 2. L2 normalize
        z_t = F.normalize(z_t, p=2, dim=-1)
        z_i = F.normalize(z_i, p=2, dim=-1)
        
        # 3. Scale logits by learned temperature
        if self.clamp_min is not None and self.clamp_min > 0:
            logit_scale = torch.clamp(self.logit_scale.exp(), min=self.clamp_min, max=100.0)
        else:
            logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        
        # Cosine similarity matrix: (B, B)
        logits_per_text = logit_scale * (z_t @ z_i.T)
        logits_per_image = logits_per_text.T
        
        # 4. Diagonal targets: (B,) [0, 1, ..., B-1]
        ground_truth = torch.arange(B, device=device, dtype=torch.long)
        
        loss_t2i = self.criterion(logits_per_text, ground_truth)
        loss_i2t = self.criterion(logits_per_image, ground_truth)
        loss_ct = (loss_t2i + loss_i2t) / 2.0
        
        return loss_ct, logits_per_text, logits_per_image

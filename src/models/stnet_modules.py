"""
STNet Modules: Sketch-Guided Attention Pooling & Sketch Reconstruction Decoder.
Ported from vl2g/CSTBIR (STNet, AAAI 2024).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SketchGuidedAttentionPooling(nn.Module):
    """
    Sketch-Guided Attention Pooling Mechanism.
    Uses sketch embedding vector as query over image patch feature tokens via scaled dot-product attention.
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = 1.0 / math.sqrt(embed_dim)

    def forward(self, sketch_embed, patch_embeds):
        """
        Args:
            sketch_embed: Tensor of shape (B, D)
            patch_embeds: Tensor of shape (B, N, D)
        Returns:
            attended_photo_embed: Tensor of shape (B, D)
        """
        if patch_embeds.dim() == 2:
            # If patch_embeds is (B, D), treat as single token sequence
            patch_embeds = patch_embeds.unsqueeze(1)

        B, N, D = patch_embeds.shape
        
        # Q: (B, 1, D)
        q = self.q_proj(sketch_embed).unsqueeze(1)
        # K, V: (B, N, D)
        k = self.k_proj(patch_embeds)
        v = self.v_proj(patch_embeds)

        # Attention map: (B, 1, N)
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale
        attn_probs = F.softmax(attn_weights, dim=-1)

        # Attended output: (B, 1, D) -> (B, D)
        attended = torch.bmm(attn_probs, v).squeeze(1)
        output = self.out_proj(attended)
        
        return output, attn_probs


class SketchReconstructionDecoder(nn.Module):
    """
    Auxiliary Sketch Reconstruction Decoder.
    Reconstructs 224x224 raster sketch image from vision patch tokens to enforce sketch-specific structural representations.
    Training-time only, discarded during ONNX/TFLite export.
    """
    def __init__(self, embed_dim=512, out_channels=3):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Linear projection to initial spatial feature grid 7x7
        self.fc = nn.Linear(embed_dim, 256 * 7 * 7)
        
        # Deconvolutional / Transposed Conv Upsampling blocks (7x7 -> 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 56x56
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),    # 112x112
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1), # 224x224
            nn.Sigmoid()
        )

    def forward(self, feature_vector):
        """
        Args:
            feature_vector: Tensor of shape (B, D)
        Returns:
            reconstructed_sketch: Tensor of shape (B, out_channels, 224, 224)
        """
        B = feature_vector.shape[0]
        x = self.fc(feature_vector).view(B, 256, 7, 7)
        reconstruction = self.decoder(x)
        return reconstruction

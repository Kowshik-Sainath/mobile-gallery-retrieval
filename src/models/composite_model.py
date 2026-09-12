"""
Composite Text + Sketch Based Image Retrieval Model.
Integrates MobileCLIP backbone, PEFT LoRA adapter, STNet Attention Pooling, and Auxiliary Sketch Reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import load_mobileclip_backbone, apply_sketch_lora
from .stnet_modules import SketchGuidedAttentionPooling, SketchReconstructionDecoder

class TSBIRCompositeModel(nn.Module):
    """
    Composite Model for Text + Sketch Based Image Retrieval (T+SBIR).
    """
    def __init__(self, backbone_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cuda", temperature=0.07):
        super().__init__()
        self.device = device
        self.temperature = temperature
        
        # Load backbone and apply LoRA adapter
        self.backbone, self.tokenizer, self.preprocess = load_mobileclip_backbone(
            model_name=backbone_name, 
            pretrained=pretrained, 
            device=device
        )
        self.backbone = apply_sketch_lora(self.backbone)

        # Get embedding dimension (512 for ViT-B-32)
        embed_dim = self.backbone.visual.output_dim if hasattr(self.backbone.visual, 'output_dim') else 512
        self.embed_dim = embed_dim

        # Fusion projection layer for sketch + text composite query
        self.composite_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

        # STNet Attention Pooling & Reconstruction Decoder
        self.attention_pooling = SketchGuidedAttentionPooling(embed_dim=embed_dim)
        self.sketch_decoder = SketchReconstructionDecoder(embed_dim=embed_dim)

    def encode_sketch(self, sketch_tensor):
        """Encodes raster sketch image using LoRA-adapted vision tower."""
        sketch_features = self.backbone.encode_image(sketch_tensor)
        return F.normalize(sketch_features, dim=-1)

    def encode_text(self, text_list):
        """Encodes text query using frozen text tower."""
        tokens = self.tokenizer(text_list).to(self.device)
        text_features = self.backbone.encode_text(tokens)
        return F.normalize(text_features, dim=-1)

    def encode_photo(self, photo_tensor):
        """Encodes photo image using frozen vision tower."""
        photo_features = self.backbone.encode_image(photo_tensor)
        return F.normalize(photo_features, dim=-1)

    def forward(self, sketch_tensor, text_list, photo_tensor, target_sketch_tensor=None):
        """
        Forward pass computing sketch embedding, text embedding, composite query, attended photo embedding, and multi-loss.
        """
        # 1. Compute embeddings
        e_sketch = self.encode_sketch(sketch_tensor)
        e_text = self.encode_text(text_list)
        e_photo = self.encode_photo(photo_tensor)

        # 2. Fuse Sketch + Text into Composite Query Embedding
        raw_composite = torch.cat([e_sketch, e_text], dim=-1)
        composite_query = F.normalize(self.composite_fusion(raw_composite), dim=-1)

        # 3. STNet Sketch-Guided Attention Pooling over photo features
        attended_photo, attn_probs = self.attention_pooling(e_sketch, e_photo)
        attended_photo = F.normalize(attended_photo, dim=-1)

        # 4. InfoNCE Contrastive Loss
        sim_matrix = torch.matmul(composite_query, attended_photo.T) / self.temperature
        batch_size = sim_matrix.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        loss_infonce = F.cross_entropy(sim_matrix, labels)

        # 5. Auxiliary Sketch Reconstruction Loss (Training time only)
        loss_rec = torch.tensor(0.0, device=self.device)
        recon_sketch = None
        if target_sketch_tensor is not None and self.training:
            recon_sketch = self.sketch_decoder(e_sketch)
            loss_rec = F.mse_loss(recon_sketch, target_sketch_tensor)

        # Combined Loss
        total_loss = loss_infonce + 0.5 * loss_rec

        return {
            'loss': total_loss,
            'loss_infonce': loss_infonce,
            'loss_rec': loss_rec,
            'composite_query': composite_query,
            'attended_photo': attended_photo,
            'recon_sketch': recon_sketch
        }

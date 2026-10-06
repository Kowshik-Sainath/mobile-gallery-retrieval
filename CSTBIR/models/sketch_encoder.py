import torch
import torch.nn as nn
import timm

class SketchEncoder(nn.Module):
    """
    Sketch Encoder for STNet (CSTBIR AAAI 2024).
    
    Paper Specification (Section: Query Encoding):
        "Given the query sketch image S, we use a pretrained ViT encoder which is fed
        the input F_S = [CLS, s_1, s_2, ..., s_m]... As the ViT encoder is pretrained on
        the ImageNet-21K dataset (Ridnik et al. 2021), we first train it on the sketch data
        for the classification task to adapt it for the sketch domain. This trained encoder
        is then used to embed the sketch input. Overall, this results into sketch embedding h^S_{CLS}."
        
    Architecture:
        - Backbone: Vision Transformer (ViT-Base, patch size 16x16, 224x224 input)
          pretrained on ImageNet-21K ('vit_base_patch16_224_miil').
        - Feature dimension: 768.
        - Adaptation head: Linear(768, num_classes) where num_classes = 258.
    """
    def __init__(
        self,
        num_classes: int = 258,
        pretrained: bool = True,
        model_name: str = 'vit_base_patch16_224_miil',
        drop_rate: float = 0.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = 768
        
        # Load ViT backbone
        try:
            self.vit = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0, # Remove classifier head to get pooled representation
                drop_rate=drop_rate
            )
            self.embed_dim = self.vit.num_features
        except Exception as e:
            # Fallback to standard vit_base_patch16_224 if specific miil weights cannot be fetched
            print(f"[SketchEncoder] Warning: Failed to load {model_name} ({e}), falling back to vit_base_patch16_224")
            self.vit = timm.create_model(
                'vit_base_patch16_224',
                pretrained=pretrained,
                num_classes=0,
                drop_rate=drop_rate
            )
            self.embed_dim = self.vit.num_features
            
        # 258-class adaptation head for QuickDraw classification
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract sketch CLS embedding h^S_{CLS}.
        Args:
            x: (B, 3, 224, 224) sketch image tensor
        Returns:
            h_cls: (B, 768) pooled CLS token embedding
        """
        return self.vit(x)
        
    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Args:
            x: (B, 3, 224, 224) sketch image tensor
            return_features: if True, returns (logits, h_cls)
        Returns:
            logits: (B, 258) classification logits (or tuple if return_features=True)
        """
        h_cls = self.forward_features(x)
        logits = self.classifier(h_cls)
        if return_features:
            return logits, h_cls
        return logits

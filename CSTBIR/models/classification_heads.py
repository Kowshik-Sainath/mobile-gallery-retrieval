import torch
import torch.nn as nn

class ObjectClassificationHeads(nn.Module):
    """
    Object Classification Heads for STNet (AAAI 2024).
    
    Paper Specification (Section: Object Classification (L_CLS^T and L_CLS^I), Page 4):
        "Given that the CSTBIR problem focuses on object-specific queries, we propose
        separately predicting the object name from the text sentence and image inputs.
        To this end, we train the text and image encoders for the multi-class classification
        objective to predict the object's class from the C object categories available in
        the train set... We refer to the classification losses computed using text encodings
        and the image encodings as L_CLS^T and L_CLS^I, respectively."
        
    Architecture:
        - text_cls_head: Linear(text_dim, num_classes) where text_dim = 512 (CLIP text embed)
        - img_cls_head:  Linear(img_dim, num_classes) where img_dim = 768 (or 512)
        - num_classes = 258 (intersecting VG & QuickDraw object categories)
    """
    def __init__(
        self,
        text_dim: int = 512,
        img_dim: int = 768,
        num_classes: int = 258,
        use_mlp: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        
        if use_mlp:
            self.text_head = nn.Sequential(
                nn.Linear(text_dim, text_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(text_dim, num_classes)
            )
            self.img_head = nn.Sequential(
                nn.Linear(img_dim, img_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(img_dim, num_classes)
            )
        else:
            self.text_head = nn.Linear(text_dim, num_classes)
            self.img_head = nn.Linear(img_dim, num_classes)

    def forward_text(self, h_T_CLS: torch.Tensor) -> torch.Tensor:
        """Predicts object class logits from text embedding h^T_{CLS}."""
        return self.text_head(h_T_CLS)

    def forward_image(self, h_I_AVG: torch.Tensor) -> torch.Tensor:
        """Predicts object class logits from sketch-attended image embedding h^I_{AVG}."""
        return self.img_head(h_I_AVG)

    def forward(self, h_T_CLS: torch.Tensor, h_I_AVG: torch.Tensor):
        """
        Returns:
            logits_txt: (B, 258)
            logits_img: (B, 258)
        """
        logits_txt = self.forward_text(h_T_CLS)
        logits_img = self.forward_image(h_I_AVG)
        return logits_txt, logits_img


class ObjectClassificationLoss(nn.Module):
    """
    Combined Text and Image Object Classification Loss:
        L_CLS = L_CLS^T + L_CLS^I
    where both use standard Multi-Class Cross-Entropy across C=258 categories.
    """
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        logits_txt: torch.Tensor,   # (B, C)
        logits_img: torch.Tensor,   # (B, C)
        targets: torch.Tensor       # (B,) category labels in [0, C-1]
    ):
        loss_txt = self.criterion(logits_txt, targets)
        loss_img = self.criterion(logits_img, targets)
        total_loss = loss_txt + loss_img
        return total_loss, loss_txt, loss_img

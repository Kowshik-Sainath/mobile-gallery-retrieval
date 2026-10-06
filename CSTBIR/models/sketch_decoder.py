import torch
import torch.nn as nn
import torch.nn.functional as F

class SketchReconstructionDecoder(nn.Module):
    """
    Sketch Reconstruction Decoder for STNet (AAAI 2024).
    
    Paper Specification (Section: Sketch Reconstruction (L_SR), Page 4-5):
        "Similar to the object detection training objective, which facilitates the localization
        of the relevant objects, we introduce the task of sketch reconstruction using the
        image features H^I as illustrated in Figure 3. We employ eight blocks of
        Convolution-BatchNorm-ReLU as done in (Isola et al. 2017) to upsample the information
        to a reconstructed sketch tensor of size 1 x 224 x 224. Further to train the
        sketch-reconstruction module, we utilize a combination of Binary Cross Entropy loss
        and the DICE loss (Sudre et al. 2017) as L_SR = alpha * L_BCE + beta * L_DICE."
        
    Architecture:
        - Input: H^I spatial tokens (14x14 grid, 768 dim).
        - 8 blocks of Conv/ConvTranspose - BatchNorm - ReLU upsampling 14x14 -> 224x224 (1x224x224).
    """
    def __init__(self, in_dim: int = 768):
        super().__init__()
        self.in_dim = in_dim
        
        # 8 Blocks of Convolution - BatchNorm - ReLU:
        # Block 1: 14x14 -> 28x28 (upsample)
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # Block 2: 28x28 (refine)
        self.block2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        # Block 3: 28x28 -> 56x56 (upsample)
        self.block3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # Block 4: 56x56 (refine)
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # Block 5: 56x56 -> 112x112 (upsample)
        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Block 6: 112x112 (refine)
        self.block6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        # Block 7: 112x112 -> 224x224 (upsample)
        self.block7 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Block 8: Final projection to 1x224x224 sketch
        self.block8 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, H_I: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H_I: (B, m, D) sketch-attended spatial tokens.
                 If m = 197, drop CLS token -> (B, 196, D) -> (B, D, 14, 14).
        Returns:
            reconstructed_sketch: (B, 1, 224, 224) pixel probabilities in [0, 1]
        """
        B, m, D = H_I.shape
        if m == 197:
            spatial_tokens = H_I[:, 1:, :] # drop CLS token -> (B, 196, D)
        else:
            spatial_tokens = H_I
            
        x = spatial_tokens.transpose(1, 2).view(B, D, 14, 14) # (B, 768, 14, 14)
        
        x = self.block1(x) # 28x28
        x = self.block2(x) # 28x28
        x = self.block3(x) # 56x56
        x = self.block4(x) # 56x56
        x = self.block5(x) # 112x112
        x = self.block6(x) # 112x112
        x = self.block7(x) # 224x224
        x = self.block8(x) # 1x224x224
        return x


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary sketch stroke segmentation (Sudre et al. 2017).
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 1, H, W) in [0, 1]
            target: (B, 1, H, W) in [0, 1]
        """
        pred_flat = pred.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)
        
        intersection = (pred_flat * target_flat).sum(dim=-1)
        cardinality = pred_flat.sum(dim=-1) + target_flat.sum(dim=-1)
        
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return torch.mean(1.0 - dice)


class SketchReconstructionLoss(nn.Module):
    """
    Combined BCE + DICE loss for sketch reconstruction:
        L_SR = alpha * L_BCE + beta * L_DICE
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_clamped = torch.clamp(pred, min=1e-6, max=1.0 - 1e-6)
        target_clamped = torch.clamp(target, min=0.0, max=1.0)
        loss_bce = self.bce(pred_clamped, target_clamped)
        loss_dice = self.dice(pred_clamped, target_clamped)
        return self.alpha * loss_bce + self.beta * loss_dice

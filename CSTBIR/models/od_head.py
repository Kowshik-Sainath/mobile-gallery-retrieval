import torch
import torch.nn as nn
import torch.nn.functional as F

class SketchGuidedObjectDetectionHead(nn.Module):
    """
    Sketch-Guided Object Detection Head (L_OD) for STNet (AAAI 2024).
    
    Paper Specification (Section: Sketch-Guided Object Detection (L_OD), Page 4):
        "Specifically, given the sketch-attended embeddings H^I from the image encoder,
        we utilize the embeddings corresponding to the 16x16 spatial grids. Following the
        implementation from YOLO (Redmon et al. 2016), we transform the output embeddings
        of the ViT network to predict an output of shape S x S x (5B + C), where S x S
        represents the image grid size, each predicting B bounding boxes, and C class
        probabilities. We use S = 7, B = 2, and we have C = 258 classes in our train set,
        so we predict a 7 x 7 x 268 output tensor. Finally, we use intersection over
        union (IoU) to calculate the multipart object detection (L_OD) loss as done in
        (Redmon et al. 2016)."
    """
    def __init__(
        self,
        in_dim: int = 768,
        grid_size: int = 7,
        num_boxes: int = 2,
        num_classes: int = 258
    ):
        super().__init__()
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        self.out_channels = 5 * num_boxes + num_classes  # 5*2 + 258 = 268
        
        # Spatial downsampler from 14x14 ViT patch tokens to 7x7 YOLO grid
        self.downsample = nn.Sequential(
            nn.Conv2d(in_dim, 512, kernel_size=3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, self.out_channels, kernel_size=1)             # 7x7x268
        )
        
    def forward(self, H_I: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H_I: (B, m, D) sketch-attended spatial tokens from cross-attention.
                 If m = 197 (1 CLS + 196 patches), CLS token is sliced off to get 14x14.
        Returns:
            out: (B, S, S, 5B + C) = (B, 7, 7, 268)
        """
        B, m, D = H_I.shape
        if m == 197:
            spatial_tokens = H_I[:, 1:, :]  # drop CLS token -> (B, 196, D)
        else:
            spatial_tokens = H_I
            
        # Reshape to 2D feature map: (B, D, 14, 14)
        feat_2d = spatial_tokens.transpose(1, 2).view(B, D, 14, 14)
        
        # Forward through downsampling head: (B, 268, 7, 7)
        out = self.downsample(feat_2d)
        
        # Permute to standard YOLO shape: (B, 7, 7, 268)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Computes IoU between two sets of boxes.
    Boxes are in (x_center, y_center, w, h) format normalized to [0, 1].
    """
    b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
    b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
    b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
    b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    b1_area = torch.clamp(b1_x2 - b1_x1, min=0) * torch.clamp(b1_y2 - b1_y1, min=0)
    b2_area = torch.clamp(b2_x2 - b2_x1, min=0) * torch.clamp(b2_y2 - b2_y1, min=0)

    union = b1_area + b2_area - inter_area + 1e-6
    return inter_area / union


class YoloDetectionLoss(nn.Module):
    """
    YOLOv1 multipart detection loss as specified in Redmon et al. 2016 and STNet.
    
    Formula:
        L_OD = lambda_coord * sum( (x - x_hat)^2 + (y - y_hat)^2 )
             + lambda_coord * sum( (sqrt(w) - sqrt(w_hat))^2 + (sqrt(h) - sqrt(h_hat))^2 )
             + sum_obj (conf - conf_hat)^2
             + lambda_noobj * sum_noobj (conf - conf_hat)^2
             + sum_obj sum_c (p(c) - p_hat(c))^2
    """
    def __init__(
        self,
        grid_size: int = 7,
        num_boxes: int = 2,
        num_classes: int = 258,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5
    ):
        super().__init__()
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, S, S, 5B + C) = (B, 7, 7, 268) model predictions
            target: (B, S, S, 5B + C) = (B, 7, 7, 268) ground truth grid from real annotations
        Returns:
            loss_od: scalar detection loss
        """
        B_batch = pred.shape[0]
        
        # Target masks: object presence is target[..., 4] == 1
        # target structure: box1=(x, y, w, h, conf), box2=(x, y, w, h, conf), classes=(p_0..p_257)
        obj_mask = (target[..., 4] > 0).unsqueeze(-1)       # (B, S, S, 1)
        noobj_mask = (target[..., 4] == 0).unsqueeze(-1)    # (B, S, S, 1)
        
        # 1. Predictions extraction
        # Box 1: (x, y, w, h)
        pred_box1 = pred[..., 0:4]
        conf1 = pred[..., 4:5]
        # Box 2: (x, y, w, h)
        pred_box2 = pred[..., 5:9]
        conf2 = pred[..., 9:10]
        # Class probabilities
        pred_cls = pred[..., 10:]

        target_box = target[..., 0:4]  # ground-truth box
        target_conf = target[..., 4:5] # 1 if object, 0 otherwise
        target_cls = target[..., 10:]  # one-hot class vector

        # Calculate IoU of both predicted boxes with ground truth to determine responsibility
        iou1 = compute_iou(pred_box1, target_box).unsqueeze(-1)  # (B, S, S, 1)
        iou2 = compute_iou(pred_box2, target_box).unsqueeze(-1)  # (B, S, S, 1)

        # best_box_mask: 1 if box 1 has higher IoU, 0 if box 2 has higher IoU
        best_box1 = (iou1 >= iou2).float() * obj_mask.float()
        best_box2 = (iou2 > iou1).float() * obj_mask.float()

        # Coordinate Loss (x, y)
        loss_xy = self.lambda_coord * (
            torch.sum(best_box1 * ((pred_box1[..., :2] - target_box[..., :2]) ** 2)) +
            torch.sum(best_box2 * ((pred_box2[..., :2] - target_box[..., :2]) ** 2))
        )

        # Coordinate Loss (w, h) with sqrt per YOLOv1
        pred_w1 = torch.sign(pred_box1[..., 2:4]) * torch.sqrt(torch.abs(pred_box1[..., 2:4]) + 1e-6)
        pred_w2 = torch.sign(pred_box2[..., 2:4]) * torch.sqrt(torch.abs(pred_box2[..., 2:4]) + 1e-6)
        target_w = torch.sqrt(torch.clamp(target_box[..., 2:4], min=1e-6))

        loss_wh = self.lambda_coord * (
            torch.sum(best_box1 * ((pred_w1 - target_w) ** 2)) +
            torch.sum(best_box2 * ((pred_w2 - target_w) ** 2))
        )

        # Object Confidence Loss (responsible predictor predicts target_conf=1.0)
        loss_conf_obj = (
            torch.sum(best_box1 * ((conf1 - target_conf) ** 2)) +
            torch.sum(best_box2 * ((conf2 - target_conf) ** 2))
        )

        # No-Object Confidence Loss (all boxes in cells without objects, plus non-responsible box in cell with object)
        noobj_box1 = noobj_mask.float() + (1.0 - best_box1) * obj_mask.float()
        noobj_box2 = noobj_mask.float() + (1.0 - best_box2) * obj_mask.float()
        
        loss_conf_noobj = self.lambda_noobj * (
            torch.sum(noobj_box1 * (conf1 ** 2)) +
            torch.sum(noobj_box2 * (conf2 ** 2))
        )

        # Classification Loss
        loss_class = torch.sum(obj_mask.float() * ((pred_cls - target_cls) ** 2))

        total_loss = (loss_xy + loss_wh + loss_conf_obj + loss_conf_noobj + loss_class) / B_batch
        return total_loss


def build_yolo_target(
    boxes: torch.Tensor,       # (B, 4) in normalized [x_min, y_min, x_max, y_max] format in [0, 1]
    class_labels: torch.Tensor, # (B,) category index in [0, C-1]
    grid_size: int = 7,
    num_classes: int = 258,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Constructs ground-truth YOLO target grid from real Visual Genome annotations.
    Strictly independent of model activations or similarity scores.
    
    Returns:
        target: (B, S, S, 5B + C) tensor with real ground-truth bounding box assignments.
    """
    B = boxes.shape[0]
    S = grid_size
    target = torch.zeros(B, S, S, 2 * 5 + num_classes, device=device)
    
    for b in range(B):
        x_min, y_min, x_max, y_max = boxes[b].tolist()
        cls_idx = class_labels[b].item()
        
        # Center coordinates and dimensions
        w = max(x_max - x_min, 1e-4)
        h = max(y_max - y_min, 1e-4)
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        
        # Determine responsible grid cell (i = row, j = col)
        col = int(min(max(x_center * S, 0), S - 1))
        row = int(min(max(y_center * S, 0), S - 1))
        
        # Box coordinates normalized relative to the cell
        x_cell = x_center * S - col
        y_cell = y_center * S - row
        
        # Assign to Box 1 target: (x_cell, y_cell, w, h, conf=1.0)
        target[b, row, col, 0] = x_cell
        target[b, row, col, 1] = y_cell
        target[b, row, col, 2] = w
        target[b, row, col, 3] = h
        target[b, row, col, 4] = 1.0  # Object present!
        
        # Also assign to Box 2 target
        target[b, row, col, 5] = x_cell
        target[b, row, col, 6] = y_cell
        target[b, row, col, 7] = w
        target[b, row, col, 8] = h
        target[b, row, col, 9] = 1.0
        
        # One-hot class target
        if 0 <= cls_idx < num_classes:
            target[b, row, col, 10 + cls_idx] = 1.0
            
    return target

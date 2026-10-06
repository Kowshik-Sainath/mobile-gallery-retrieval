import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from CSTBIR.clip import clip
from CSTBIR.models.sketch_encoder import SketchEncoder
from CSTBIR.models.cross_attention import SketchGuidedImageAttention
from CSTBIR.models.od_head import SketchGuidedObjectDetectionHead, YoloDetectionLoss, build_yolo_target
from CSTBIR.models.sketch_decoder import SketchReconstructionDecoder, SketchReconstructionLoss
from CSTBIR.models.classification_heads import ObjectClassificationHeads, ObjectClassificationLoss
from CSTBIR.models.contrastive_loss import ContrastiveRetrievalLoss

class STNet(nn.Module):
    """
    STNet: Multimodal Transformer Architecture for Composite Sketch+Text Image Retrieval.
    (AAAI 2024 - Gatti et al.)
    
    Architecture (Figure 3 & Section 4):
        1. Query Text Encoder: Pretrained CLIP Transformer text encoder -> h^T_{CLS} (512-dim)
        2. Query Sketch Encoder: Pretrained ViT (ImageNet-21K) adapted on QuickDraw -> h^S_{CLS} (768-dim)
        3. Image Encoder: Pretrained CLIP-ViT -> spatial tokens H~^I (197x768)
        4. Cross-Modal Attention: alpha_IS = Softmax(H~^I x h^S_{CLS}) -> h^I_{AVG} & H^I
        
    Auxiliary Heads & Training Objectives:
        - L_CT: Contrastive retrieval loss (InfoNCE between h^T_{CLS} and h^I_{AVG})
        - L_CLS^T, L_CLS^I: Multi-class object classification (258 classes) on text and image
        - L_OD: Sketch-guided YOLOv1 object detection (7x7x268 grid)
        - L_SR: Sketch reconstruction decoder (8-block Conv-BN-ReLU -> 1x224x224)
        
    Overall Loss (unweighted sum per paper formula):
        L_total = L_CT + L_CLS^T + L_CLS^I + L_OD + L_SR
    """
    def __init__(
        self,
        clip_model_name: str = "ViT-B/16",
        num_classes: int = 258,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        pretrained_sketch: bool = False
    ):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        
        # 1. Base CLIP Model (Text & Image Encoders)
        print(f"[STNet] Loading CLIP model: {clip_model_name}...")
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
        self.clip_model = self.clip_model.float()
        self.text_dim = self.clip_model.transformer.width       # 512
        self.img_dim = self.clip_model.visual.transformer.width # 768
        
        # 2. Sketch Encoder (ViT pretrained on ImageNet-21K, adapted to QuickDraw 258 classes)
        print(f"[STNet] Initializing SketchEncoder (258 classes, embed_dim={self.img_dim})...")
        self.sketch_encoder = SketchEncoder(num_classes=num_classes, pretrained=pretrained_sketch).to(device)
        
        # 3. Cross-Modal Attention: alpha_IS = Softmax(H~^I x h^S_{CLS})
        self.cross_attention = SketchGuidedImageAttention(embed_dim=self.img_dim, scale=False).to(device)
        
        # 4. Object Detection Head (YOLOv1 7x7x268)
        self.od_head = SketchGuidedObjectDetectionHead(
            in_dim=self.img_dim, grid_size=7, num_boxes=2, num_classes=num_classes
        ).to(device)
        
        # 5. Sketch Reconstruction Decoder (8 blocks Conv-BN-ReLU)
        self.sketch_decoder = SketchReconstructionDecoder(in_dim=self.img_dim).to(device)
        
        # 6. Object Classification Heads (258 classes)
        self.cls_heads = ObjectClassificationHeads(
            text_dim=self.text_dim, img_dim=self.img_dim, num_classes=num_classes
        ).to(device)
        
        # Loss Modules
        self.loss_ct_fn = ContrastiveRetrievalLoss(text_dim=self.text_dim, img_dim=self.img_dim, proj_dim=self.text_dim).to(device)
        self.loss_cls_fn = ObjectClassificationLoss().to(device)
        self.loss_od_fn = YoloDetectionLoss(grid_size=7, num_boxes=2, num_classes=num_classes).to(device)
        self.loss_sr_fn = SketchReconstructionLoss(alpha=1.0, beta=1.0).to(device)

    def extract_image_tokens(self, image: torch.Tensor) -> torch.Tensor:
        """Extracts spatial + CLS tokens H~^I from CLIP image transformer: (B, 197, 768)."""
        visual = self.clip_model.visual
        dtype = visual.conv1.weight.dtype
        x = visual.conv1(image.type(dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls_tok = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_tok, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)
        tokens = visual.ln_post(x).float()
        return tokens

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Extracts text CLS embedding h^T_{CLS}: (B, 512)."""
        return self.clip_model.encode_text(text_tokens).float()

    def encode_sketch(self, sketch: torch.Tensor) -> torch.Tensor:
        """Extracts sketch CLS embedding h^S_{CLS}: (B, 768)."""
        return self.sketch_encoder.forward_features(sketch)

    def forward(
        self,
        text_tokens: torch.Tensor,             # (B, 77)
        images: torch.Tensor,                  # (B, 3, 224, 224)
        sketches: Optional[torch.Tensor] = None, # (B, 3, 224, 224) or precomputed embeds
        sketch_embeds: Optional[torch.Tensor] = None, # (B, 768)
        gt_boxes: Optional[torch.Tensor] = None,       # (B, 4) [x_min, y_min, x_max, y_max]
        gt_labels: Optional[torch.Tensor] = None,      # (B,) category index in [0, C-1]
        target_sketch_imgs: Optional[torch.Tensor] = None # (B, 1, 224, 224) for L_SR
    ) -> Dict[str, torch.Tensor]:
        B = text_tokens.shape[0]
        
        # 1. Encodings
        h_T_CLS = self.encode_text(text_tokens)  # (B, 512)
        
        if sketch_embeds is not None:
            h_S_CLS = sketch_embeds
        elif sketches is not None:
            h_S_CLS = self.encode_sketch(sketches) # (B, 768)
        else:
            raise ValueError("Either sketches or sketch_embeds must be provided!")
            
        H_tilde_I = self.extract_image_tokens(images) # (B, 197, 768)
        
        # 2. Cross-Modal Attention: alpha_IS = Softmax(H~^I x h^S_{CLS})
        h_I_AVG, H_I, alpha_IS = self.cross_attention(H_tilde_I, h_S_CLS)
        
        # 3. Auxiliary Heads
        # Classification
        logits_txt, logits_img = self.cls_heads(h_T_CLS, h_I_AVG)
        # Detection Head
        pred_od_grid = self.od_head(H_I)
        # Reconstruction Decoder
        pred_sketches = self.sketch_decoder(H_I)
        
        # 4. Compute All 5 Losses
        # Loss 1: L_CT (Contrastive)
        loss_ct, logits_per_text, logits_per_image = self.loss_ct_fn(h_T_CLS, h_I_AVG)
        
        # Loss 2 & 3: L_CLS^T and L_CLS^I (Classification)
        if gt_labels is not None:
            loss_cls, loss_cls_t, loss_cls_i = self.loss_cls_fn(logits_txt, logits_img, gt_labels)
        else:
            loss_cls = torch.tensor(0.0, device=self.device)
            loss_cls_t = torch.tensor(0.0, device=self.device)
            loss_cls_i = torch.tensor(0.0, device=self.device)
            
        # Loss 4: L_OD (Object Detection)
        if gt_boxes is not None and gt_labels is not None:
            gt_od_grid = build_yolo_target(gt_boxes, gt_labels, grid_size=7, num_classes=self.num_classes, device=self.device)
            loss_od = self.loss_od_fn(pred_od_grid, gt_od_grid)
        else:
            loss_od = torch.tensor(0.0, device=self.device)
            
        # Loss 5: L_SR (Sketch Reconstruction)
        if target_sketch_imgs is not None:
            loss_sr = self.loss_sr_fn(pred_sketches, target_sketch_imgs)
        else:
            loss_sr = torch.tensor(0.0, device=self.device)
            
        # Total Loss: Unweighted sum per paper
        total_loss = loss_ct + loss_cls_t + loss_cls_i + loss_od + loss_sr
        
        return {
            'loss_total': total_loss,
            'loss_ct': loss_ct,
            'loss_cls_t': loss_cls_t,
            'loss_cls_i': loss_cls_i,
            'loss_cls': loss_cls,
            'loss_od': loss_od,
            'loss_sr': loss_sr,
            'h_T_CLS': h_T_CLS,
            'h_I_AVG': h_I_AVG,
            'alpha_IS': alpha_IS,
            'logits_per_text': logits_per_text,
            'logits_per_image': logits_per_image,
            'pred_sketches': pred_sketches,
            'pred_od_grid': pred_od_grid
        }

    @torch.no_grad()
    def compute_query_image_similarity(
        self,
        text_tokens: torch.Tensor,   # (Q, 77) query texts
        sketch_embeds: torch.Tensor, # (Q, 768) query sketch embeddings
        gallery_images: torch.Tensor # (K, 3, 224, 224) gallery images
    ) -> torch.Tensor:
        """
        Computes the Q x K cosine similarity matrix for gallery ranking evaluation.
        For each query (text_q, sketch_q) and gallery image k:
            similarity = CosineSim( z_t(text_q), z_i( attend(image_k, sketch_q) ) )
        """
        Q = text_tokens.shape[0]
        K = gallery_images.shape[0]
        
        # 1. Encode text queries: (Q, 512)
        h_T = self.encode_text(text_tokens)
        z_t = F.normalize(self.loss_ct_fn.txt_proj(h_T), p=2, dim=-1)
        
        # 2. Extract image tokens for all gallery images: (K, 197, 768)
        H_tilde_gallery = self.extract_image_tokens(gallery_images)
        
        # 3. For each query, attend to gallery images conditioned on sketch_q
        # To avoid OOM when Q and K are large, compute in chunks of queries
        sim_matrix = torch.zeros(Q, K, device=self.device)
        
        for q in range(Q):
            sq = sketch_embeds[q:q+1].repeat(K, 1) # (K, 768)
            h_I_avg, _, _ = self.cross_attention(H_tilde_gallery, sq) # (K, 768)
            z_i = F.normalize(self.loss_ct_fn.img_proj(h_I_avg), p=2, dim=-1) # (K, 512)
            
            # Dot product with z_t[q]: (1, 512) @ (512, K) -> (1, K)
            sim_matrix[q] = (z_t[q:q+1] @ z_i.T).squeeze(0)
            
        return sim_matrix

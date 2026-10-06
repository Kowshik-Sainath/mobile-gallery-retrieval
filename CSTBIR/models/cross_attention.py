import torch
import torch.nn as nn
import torch.nn.functional as F

class SketchGuidedImageAttention(nn.Module):
    """
    Sketch-Guided Cross-Modal Attention for STNet (AAAI 2024).
    
    Paper Specification (Section: Image Encoding, Page 4):
        "Specifically, as shown in Figure 3, we use the pooled output of the sketch encoder
        h^S_{CLS} to calculate dot-product attention over the output embeddings of the
        image encoder H~^I. The obtained values represent attention scores alpha_IS over
        the spatial regions of the image as well as the CLS token. We obtain weighted
        values of image embeddings H^I, which are then average pooled to get the final
        image embedding h^I_{AVG}. Mathematically:
            alpha_IS = Softmax(H~^I x h^S_{CLS})
            H^I = alpha_IS (elementwise) H~^I
            h^I_{AVG} = (1/m) * sum_{i=1}^m H^I_i
        where h^I_{AVG} represents the global average pooled embedding of the image encoder."
        
    Design Note:
        The paper states the raw dot product: alpha_IS = Softmax(H~^I x h^S_{CLS}).
        We provide an optional 'scale' flag (default=False for exact paper fidelity,
        or True for standard 1/sqrt(D) scaled dot-product attention).
    """
    def __init__(self, embed_dim: int = 768, scale: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale_factor = (embed_dim ** -0.5) if scale else 1.0

    def forward(
        self,
        H_tilde_I: torch.Tensor,  # (B, m, D) image patch + CLS embeddings
        h_S_CLS: torch.Tensor     # (B, D) sketch CLS embedding
    ):
        """
        Args:
            H_tilde_I: (B, m, D) tensor from image transformer
            h_S_CLS:   (B, D) tensor from sketch encoder
        Returns:
            h_I_AVG:  (B, D) sketch-attended pooled image embedding
            H_I:      (B, m, D) sketch-weighted spatial feature map
            alpha_IS: (B, m) attention probability distribution over tokens
        """
        B, m, D = H_tilde_I.shape
        assert h_S_CLS.shape == (B, D), f"Dimension mismatch: H_tilde_I has D={D}, h_S_CLS has shape {h_S_CLS.shape}"

        # 1. Compute dot-product attention scores: (B, m)
        # H_tilde_I: (B, m, D), h_S_CLS.unsqueeze(-1): (B, D, 1) -> (B, m, 1)
        raw_scores = torch.bmm(H_tilde_I, h_S_CLS.unsqueeze(-1)).squeeze(-1) * self.scale_factor  # (B, m)

        # 2. Softmax over spatial tokens (and CLS token): (B, m)
        alpha_IS = F.softmax(raw_scores, dim=-1)  # (B, m), sums to 1.0 per sample

        # 3. Element-wise product: H^I = alpha_IS (elementwise) H~^I
        # alpha_IS.unsqueeze(-1): (B, m, 1), H_tilde_I: (B, m, D) -> (B, m, D)
        H_I = alpha_IS.unsqueeze(-1) * H_tilde_I  # (B, m, D)

        # 4. Global average pooling over m tokens: (B, D)
        h_I_AVG = H_I.mean(dim=1)  # (B, D)

        return h_I_AVG, H_I, alpha_IS

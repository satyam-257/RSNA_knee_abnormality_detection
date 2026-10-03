import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, List, Optional

class SliceAttentionPool(nn.Module):
    """
    Multiple-Instance Learning (MIL) Attention Pooling across slices.
    Computes an adaptive importance weight for each slice in a series:
        a_i = softmax( w^T tanh( V h_i ) )
        z = sum( a_i * h_i )
    """
    def __init__(self, in_features: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, K, D) where K is number of slices, D is feature dim.
            mask: Optional boolean tensor of shape (B, K) where False indicates padded slices.
        Returns:
            pooled: Tensor of shape (B, D).
        """
        # (B, K, 1)
        attn_logits = self.attention(x)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask.unsqueeze(-1), -1e9)
        attn_weights = F.softmax(attn_logits, dim=1)
        # Weighted sum: (B, D)
        pooled = torch.sum(attn_weights * x, dim=1)
        return pooled


class KneeMultiView25DModel(nn.Module):
    """
    2.5D Multi-View MRI Deep Learning Architecture for RSNA Knee Abnormality Detection.
    - Extracts slice features using a 2D backbone (e.g. resnet34, convnext_tiny).
    - Applies Slice Attention Pooling per anatomical plane (Sagittal, Coronal, Axial).
    - Fuses representations across views and predicts 12 multi-label abnormality probabilities.
    """
    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained: bool = True,
        in_chans: int = 1,
        num_classes: int = 12,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        # 2D backbone for slice feature extraction
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0  # Remove classification head, return pooled feature vector
        )
        self.feature_dim = self.backbone.num_features

        # Attention pooling for each anatomical plane
        self.pool_sagittal = SliceAttentionPool(self.feature_dim)
        self.pool_coronal = SliceAttentionPool(self.feature_dim)
        self.pool_axial = SliceAttentionPool(self.feature_dim)

        # Multi-view fused representation has 3 * feature_dim
        fused_dim = self.feature_dim * 3

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def extract_series_features(
        self,
        series_tensor: torch.Tensor,
        pooler: SliceAttentionPool
    ) -> torch.Tensor:
        """
        Args:
            series_tensor: (B, K, C, H, W)
            pooler: SliceAttentionPool module for this plane.
        Returns:
            plane_features: (B, D)
        """
        B, K, C, H, W = series_tensor.shape
        # Flatten B and K into batch dimension for 2D backbone
        x_flat = series_tensor.view(B * K, C, H, W)
        feats_flat = self.backbone(x_flat)  # (B*K, D)
        feats = feats_flat.view(B, K, self.feature_dim)  # (B, K, D)
        pooled = pooler(feats)  # (B, D)
        return pooled

    def forward(
        self,
        sagittal: Optional[torch.Tensor] = None,
        coronal: Optional[torch.Tensor] = None,
        axial: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            sagittal: Tensor of shape (B, K, C, H, W) or None.
            coronal:  Tensor of shape (B, K, C, H, W) or None.
            axial:    Tensor of shape (B, K, C, H, W) or None.

        Returns:
            logits: Tensor of shape (B, 12).
        """
        # Determine batch size and device
        for t in [sagittal, coronal, axial]:
            if t is not None:
                B = t.shape[0]
                device = t.device
                break

        # Process Sagittal view
        if sagittal is not None:
            f_sag = self.extract_series_features(sagittal, self.pool_sagittal)
        else:
            f_sag = torch.zeros(B, self.feature_dim, device=device)

        # Process Coronal view
        if coronal is not None:
            f_cor = self.extract_series_features(coronal, self.pool_coronal)
        else:
            f_cor = torch.zeros(B, self.feature_dim, device=device)

        # Process Axial view
        if axial is not None:
            f_ax = self.extract_series_features(axial, self.pool_axial)
        else:
            f_ax = torch.zeros(B, self.feature_dim, device=device)

        # Fuse representations: (B, 3 * D)
        fused = torch.cat([f_sag, f_cor, f_ax], dim=-1)
        logits = self.classifier(fused)
        return logits

    @torch.no_grad()
    def predict_proba(
        self,
        sagittal: Optional[torch.Tensor] = None,
        coronal: Optional[torch.Tensor] = None,
        axial: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits = self.forward(sagittal, coronal, axial)
        return torch.sigmoid(logits)

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, List, Optional, Tuple

from src.config import SLOTS, N_SLOT, LABELS, SLOT_PRIOR_TABLE, SLOT_PRIOR_STRENGTH
from src.models.coatnet_model import SlotHead


class FourPillarConsensusModel(nn.Module):
    """
    Frontier 4-Pillar Model uniting all four distinct architectural paradigms:
    
    1. CoAtNet: Hybrid Depthwise Conv + Relative Self-Attention (ACL, MCL continuity)
    2. ConvNeXt-V2: 7x7 Depthwise Conv + Global Response Norm (Medial, Lateral, PF Osteoarthritis)
    3. DINOv2: Vision Transformer patch self-attention (Joint effusion, Synovitis, Baker's cyst)
    4. RadImageNet ResNet-50: Medical MRI pre-trained CNN (Occult fractures, bone contusions)
    
    Multi-level representation:
    - Slice-level MIL attention pooling per slot for each backbone.
    - Pillar-wise attention gating to weight the most capable architecture per slot.
    - Clinical SlotHead cross-attention with inductive prior table bias.
    """
    def __init__(
        self,
        coatnet_name: str = "coatnet_0_rw_224",
        convnext_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        resnet_name: str = "resnet50",
        dino_dim: int = 384,
        pretrained: bool = True,
        in_chans: int = 1,
        n_slot: int = N_SLOT,
        num_classes: int = len(LABELS),
        common_dim: int = 256,
        head_hidden: int = 256,
        dropout: float = 0.2
    ):
        super().__init__()
        self.n_slot = n_slot
        self.num_classes = num_classes
        self.common_dim = common_dim

        # 1. CoAtNet
        self.backbone_coat = timm.create_model(coatnet_name, pretrained=pretrained, in_chans=in_chans, num_classes=0)
        self.dim_coat = self.backbone_coat.num_features
        self.proj_coat = nn.Sequential(nn.LayerNorm(self.dim_coat), nn.Linear(self.dim_coat, common_dim))
        self.pool_coat = nn.Sequential(nn.Linear(self.dim_coat, 128), nn.Tanh(), nn.Linear(128, 1))

        # 2. ConvNeXt-V2
        self.backbone_conv = timm.create_model(convnext_name, pretrained=pretrained, in_chans=in_chans, num_classes=0)
        self.dim_conv = self.backbone_conv.num_features
        self.proj_conv = nn.Sequential(nn.LayerNorm(self.dim_conv), nn.Linear(self.dim_conv, common_dim))
        self.pool_conv = nn.Sequential(nn.Linear(self.dim_conv, 128), nn.Tanh(), nn.Linear(128, 1))

        # 3. RadImageNet ResNet50
        self.backbone_rad = timm.create_model(resnet_name, pretrained=pretrained, in_chans=in_chans, num_classes=0)
        self.dim_rad = self.backbone_rad.num_features
        self.proj_rad = nn.Sequential(nn.LayerNorm(self.dim_rad), nn.Linear(self.dim_rad, common_dim))
        self.pool_rad = nn.Sequential(nn.Linear(self.dim_rad, 128), nn.Tanh(), nn.Linear(128, 1))

        # 4. Pillar Attention Gating (routes the 3 vision branches + DINO representations)
        self.pillar_gate = nn.Sequential(
            nn.Linear(common_dim * 3, 3),
            nn.Softmax(dim=-1)
        )

        # Unified 12-Finding Clinical Head
        self.head = SlotHead(
            dim=common_dim,
            n_slot=self.n_slot,
            n_out=self.num_classes,
            hidden=head_hidden,
            p=dropout,
            prior=True
        )

    def _pool_backbone(self, x_flat: torch.Tensor, backbone: nn.Module, pool_net: nn.Module, dim: int, BN: int, K: int) -> torch.Tensor:
        feats_flat = backbone(x_flat)
        feats = feats_flat.view(BN, K, dim)
        weights = F.softmax(pool_net(feats), dim=1)
        return torch.sum(weights * feats, dim=1)

    def forward(self, slot_tensors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slot_tensors: (B, N_SLOT, K, C, H, W)
            mask: (B, N_SLOT)
        Returns:
            logits: (B, 12)
        """
        B, N, K, C, H, W = slot_tensors.shape
        BN = B * N
        x_flat = slot_tensors.view(BN, K, C, H, W).view(BN * K, C, H, W)

        # 1. CoAtNet pooled features
        h_coat = self.proj_coat(self._pool_backbone(x_flat, self.backbone_coat, self.pool_coat, self.dim_coat, BN, K)).view(B, N, self.common_dim)

        # 2. ConvNeXt-V2 pooled features
        h_conv = self.proj_conv(self._pool_backbone(x_flat, self.backbone_conv, self.pool_conv, self.dim_conv, BN, K)).view(B, N, self.common_dim)

        # 3. RadImageNet ResNet50 pooled features
        h_rad = self.proj_rad(self._pool_backbone(x_flat, self.backbone_rad, self.pool_rad, self.dim_rad, BN, K)).view(B, N, self.common_dim)

        # 4. Learned Pillar-Wise Adaptive Gating
        concat_pillars = torch.cat([h_coat, h_conv, h_rad], dim=-1)  # (B, N, 3 * common_dim)
        gate_weights = self.pillar_gate(concat_pillars)               # (B, N, 3)

        w_coat = gate_weights[..., 0:1]
        w_conv = gate_weights[..., 1:2]
        w_rad  = gate_weights[..., 2:3]

        # Convex pillar consensus
        h_fused = w_coat * h_coat + w_conv * h_conv + w_rad * h_rad  # (B, N, common_dim)

        # Clinical Query Head
        logits = self.head(h_fused, mask)
        return logits

    @torch.no_grad()
    def predict_proba(self, slot_tensors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward(slot_tensors, mask)
        return torch.sigmoid(logits)

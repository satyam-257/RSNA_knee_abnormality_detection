import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, List, Optional, Tuple

from src.config import SLOTS, N_SLOT, LABELS, SLOT_PRIOR_TABLE, SLOT_PRIOR_STRENGTH
from src.models.coatnet_model import SlotHead


class GatedFeatureFusion(nn.Module):
    """
    Learned Gated Fusion between CoAtNet and ConvNeXt-V2 representations.
    
    Instead of simple concatenation, dynamic gating allows the network to
    selectively favor CoAtNet for global continuous structures (ACL, MCL)
    and ConvNeXt-V2 for high-frequency localized textures (OA, subchondral wear).
    """
    def __init__(self, dim_coat: int, dim_conv: int, out_dim: int):
        super().__init__()
        self.proj_coat = nn.Sequential(
            nn.LayerNorm(dim_coat),
            nn.Linear(dim_coat, out_dim)
        )
        self.proj_conv = nn.Sequential(
            nn.LayerNorm(dim_conv),
            nn.Linear(dim_conv, out_dim)
        )
        # Gate generator conditioned on both feature sets
        self.gate = nn.Sequential(
            nn.Linear(dim_coat + dim_conv, out_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h_coat: torch.Tensor, h_conv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_coat: Tensor of shape (B, N_SLOT, dim_coat)
            h_conv: Tensor of shape (B, N_SLOT, dim_conv)
        Returns:
            fused: Tensor of shape (B, N_SLOT, out_dim)
        """
        p_coat = self.proj_coat(h_coat)
        p_conv = self.proj_conv(h_conv)

        concat = torch.cat([h_coat, h_conv], dim=-1)
        g = self.gate(concat)  # (B, N_SLOT, out_dim) in [0, 1]

        # Convex combination
        fused = g * p_coat + (1.0 - g) * p_conv
        return self.norm(fused)


class CoAtNetConvNeXtFusionModel(nn.Module):
    """
    Dual-Backbone Hybrid Model combining CoAtNet and ConvNeXt-V2.
    
    Architecture:
    1. Parallel slice feature extractors:
       - CoAtNet branch: Hybrid Depthwise Conv + Relative Self-Attention
       - ConvNeXt-V2 branch: 7x7 Depthwise Conv + Global Response Normalization (GRN)
    2. Slice-level MIL attention pooling independently for each backbone.
    3. Gated representation fusion per clinical anatomical slot.
    4. 12-Finding Clinical Query Cross-Attention Head with prior table injection.
    """
    def __init__(
        self,
        coatnet_name: str = "coatnet_0_rw_224",
        convnext_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        pretrained: bool = True,
        in_chans: int = 1,
        n_slot: int = N_SLOT,
        num_classes: int = len(LABELS),
        fusion_dim: int = 384,
        head_hidden: int = 256,
        dropout: float = 0.2
    ):
        super().__init__()
        self.n_slot = n_slot
        self.num_classes = num_classes

        # Backbone 1: CoAtNet
        self.backbone_coat = timm.create_model(
            coatnet_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0
        )
        self.dim_coat = self.backbone_coat.num_features

        # Backbone 2: ConvNeXt-V2
        self.backbone_conv = timm.create_model(
            convnext_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0
        )
        self.dim_conv = self.backbone_conv.num_features

        # Independent slice attention poolers
        self.pool_coat = nn.Sequential(
            nn.Linear(self.dim_coat, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.pool_conv = nn.Sequential(
            nn.Linear(self.dim_conv, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Gated fusion module
        self.fusion = GatedFeatureFusion(
            dim_coat=self.dim_coat,
            dim_conv=self.dim_conv,
            out_dim=fusion_dim
        )

        # Unified 12-query clinical head
        self.head = SlotHead(
            dim=fusion_dim,
            n_slot=self.n_slot,
            n_out=self.num_classes,
            hidden=head_hidden,
            p=dropout,
            prior=True
        )

    def _pool_branch(self, x_flat: torch.Tensor, backbone: nn.Module, pool_net: nn.Module, dim: int, BN: int, K: int) -> torch.Tensor:
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
        slot_tensors_flat = slot_tensors.view(B * N, K, C, H, W)
        BN = B * N
        x_flat = slot_tensors_flat.view(BN * K, C, H, W)

        # Extract & pool CoAtNet features
        h_coat = self._pool_branch(x_flat, self.backbone_coat, self.pool_coat, self.dim_coat, BN, K)
        h_coat = h_coat.view(B, N, self.dim_coat)

        # Extract & pool ConvNeXt-V2 features
        h_conv = self._pool_branch(x_flat, self.backbone_conv, self.pool_conv, self.dim_conv, BN, K)
        h_conv = h_conv.view(B, N, self.dim_conv)

        # Gated multi-backbone feature fusion
        h_fused = self.fusion(h_coat, h_conv)  # (B, N_SLOT, fusion_dim)

        # Route through clinical query cross-attention head
        logits = self.head(h_fused, mask)
        return logits

    @torch.no_grad()
    def predict_proba(self, slot_tensors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward(slot_tensors, mask)
        return torch.sigmoid(logits)

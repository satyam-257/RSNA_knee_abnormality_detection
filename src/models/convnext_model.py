import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, List, Optional, Tuple

from src.config import SLOTS, N_SLOT, LABELS, SLOT_PRIOR_TABLE, SLOT_PRIOR_STRENGTH
from src.models.coatnet_model import SlotHead


class ConvNeXtV2SlotModel(nn.Module):
    """
    Pure Convolutional 2.5D Architecture based on ConvNeXt-V2.
    
    Why ConvNeXt-V2 for Knee MRI:
    - 7x7 Depthwise convolutions provide large receptive fields with pure CNN inductive bias.
    - Global Response Normalization (GRN) prevents channel collapse on subtle contrast gradients.
    - Excels at high-frequency texture detection: cartilage thinning (Medial/Lateral/PF OA),
      subchondral sclerosis, and bone contusions.
    """
    def __init__(
        self,
        backbone_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        pretrained: bool = True,
        in_chans: int = 1,
        n_slot: int = N_SLOT,
        num_classes: int = len(LABELS),
        hidden_dim: int = 256,
        dropout: float = 0.2
    ):
        super().__init__()
        self.n_slot = n_slot
        self.num_classes = num_classes

        # 2D ConvNeXt-V2 backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0
        )
        self.dim = self.backbone.num_features

        # Gated slice attention pooling per slot
        self.slice_attention = nn.Sequential(
            nn.Linear(self.dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Query-based clinical slot head
        self.head = SlotHead(
            dim=self.dim,
            n_slot=self.n_slot,
            n_out=self.num_classes,
            hidden=hidden_dim,
            p=dropout,
            prior=True
        )

    def pool_slices(self, slot_slices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slot_slices: (B * N_SLOT, K, C, H, W)
        Returns:
            pooled: (B * N_SLOT, Dim)
        """
        BN, K, C, H, W = slot_slices.shape
        x_flat = slot_slices.view(BN * K, C, H, W)
        feats_flat = self.backbone(x_flat)  # (BN*K, Dim)
        feats = feats_flat.view(BN, K, self.dim)  # (BN, K, Dim)

        # Gated attention weights across slices
        weights = F.softmax(self.slice_attention(feats), dim=1)  # (BN, K, 1)
        pooled = torch.sum(weights * feats, dim=1)  # (BN, Dim)
        return pooled

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

        # Pool slice features for all slots
        slot_feats = self.pool_slices(slot_tensors_flat)  # (B*N, Dim)
        slot_feats = slot_feats.view(B, N, self.dim)      # (B, N_SLOT, Dim)

        # Clinical slot routing
        logits = self.head(slot_feats, mask)
        return logits

    @torch.no_grad()
    def predict_proba(self, slot_tensors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward(slot_tensors, mask)
        return torch.sigmoid(logits)

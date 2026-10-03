import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, List, Optional, Tuple

from src.config import SLOTS, N_SLOT, LABELS, SLOT_PRIOR_TABLE, SLOT_PRIOR_STRENGTH

class SlotHead(nn.Module):
    """
    Query-based Cross-Attention Head with Clinical Inductive Prior Bias.
    As designed in Speedy Raptors CoAtNet (LB 0.943+).
    
    Learns an abnormality-specific query for each of the 12 targets that attends
    across the 6 anatomical slots, biased by the clinical prior table.
    """
    def __init__(
        self,
        dim: int,
        n_slot: int = N_SLOT,
        n_out: int = len(LABELS),
        hidden: int = 256,
        p: float = 0.2,
        prior: bool = True
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU()
        )
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        self.prior = prior

        if prior:
            p_ = torch.zeros(n_out, n_slot)
            for t, slots in SLOT_PRIOR_TABLE.items():
                if t in LABELS:
                    t_idx = LABELS.index(t)
                    p_[t_idx, list(slots)] = SLOT_PRIOR_STRENGTH
            self.register_buffer("slot_prior", p_)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, N_SLOT, Dim) containing pooled features for each slot.
            mask: Tensor of shape (B, N_SLOT) where 1 indicates present slot, 0 indicates missing slot.

        Returns:
            logits: Tensor of shape (B, N_OUT).
        """
        # (B, N_SLOT, Hidden)
        h = self.proj(x) + self.slot_emb

        # Cross attention: (B, N_OUT, N_SLOT)
        att = torch.einsum("bsh,oh->bos", h, self.query) / (self.hidden ** 0.5)

        if self.prior:
            att = att + self.slot_prior.unsqueeze(0)

        # Mask missing slots with -1e4 before softmax
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -10000.0).softmax(-1)

        # Context vector per abnormality: (B, N_OUT, Hidden)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))

        # Project to scalar logit per abnormality: (B, N_OUT)
        logits = (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias
        return logits


class SpeedyRaptorCoAtNetModel(nn.Module):
    """
    Speedy Raptors CoAtNet Hybrid Architecture.
    - Pretrained CoAtNet (or ConvNeXt/ResNet) backbone for slice representation.
    - Multiple-Instance Learning (MIL) slice pooling per slot.
    - 6-Slot Clinical Head with inductive prior bias.
    """
    def __init__(
        self,
        backbone_name: str = "coatnet_0_rw_224",
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

        # 2D backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0
        )
        self.dim = self.backbone.num_features

        # Slice attention pooling per slot
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

        # Attention weights across slices
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
        slot_feats = slot_feats.view(B, N, self.dim)  # (B, N_SLOT, Dim)

        # Query cross-attention head with clinical prior
        logits = self.head(slot_feats, mask)
        return logits

    @torch.no_grad()
    def predict_proba(self, slot_tensors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.forward(slot_tensors, mask)
        return torch.sigmoid(logits)

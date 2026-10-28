"""Trainable fusion head for whole-vehicle, viewpoint, and named-part embeddings."""

from __future__ import annotations

import math

import torch
from torch import nn

PART_TYPES = ("headlight", "taillight", "grille", "door", "window", "spoiler", "wheel", "mirror")
VIEWPOINT_TYPES = ("front", "front_three_quarter", "side", "rear_three_quarter", "rear", "unknown")


class PartAwareFusion(nn.Module):
    """Fuse features from visible parts without assuming every part is present."""

    def __init__(
        self,
        feature_size: int,
        hidden_size: int,
        make_classes: int,
        model_classes: int,
        generation_classes: int,
    ) -> None:
        super().__init__()
        self.global_projection = nn.Linear(feature_size, hidden_size)
        self.part_projection = nn.Linear(feature_size, hidden_size)
        self.part_embedding = nn.Embedding(len(PART_TYPES), hidden_size)
        self.viewpoint_embedding = nn.Embedding(len(VIEWPOINT_TYPES), hidden_size)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.make_head = nn.Linear(hidden_size, make_classes)
        self.model_head = nn.Linear(hidden_size, model_classes)
        self.generation_head = nn.Linear(hidden_size, generation_classes)

    def forward(
        self,
        global_features: torch.Tensor,
        part_features: torch.Tensor,
        part_type_ids: torch.Tensor,
        visible_part_mask: torch.Tensor,
        viewpoint_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return hierarchical logits and interpretable weights for visible parts.

        Shapes are global [batch, feature], parts [batch, parts, feature], IDs/mask
        [batch, parts], and viewpoint [batch]. A true mask value means that part is
        visible and annotated for the image.
        """
        global_hidden = self.global_projection(global_features)
        part_hidden = self.part_projection(part_features) + self.part_embedding(part_type_ids)
        query = global_hidden + self.viewpoint_embedding(viewpoint_ids)
        scores = (part_hidden * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(part_hidden.shape[-1])
        scores = scores.masked_fill(~visible_part_mask.bool(), -1e4)
        weights = scores.softmax(dim=1)
        weights = weights * visible_part_mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        part_summary = (part_hidden * weights.unsqueeze(-1)).sum(dim=1)
        fused = self.fusion(torch.cat((global_hidden, part_summary), dim=-1))
        return {
            "make_logits": self.make_head(fused),
            "model_logits": self.model_head(fused),
            "generation_logits": self.generation_head(fused),
            "part_weights": weights,
        }

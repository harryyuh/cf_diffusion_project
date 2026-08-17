"""Learned causal tokens for frozen text-to-image cross-attention."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn


class CausalTokenMapper(nn.Module):
    """Map binary causal attributes to learned cross-attention tokens.

    Input labels are expected as a tensor with one column per attribute, using
    either {-1, +1} or {0, 1}. The output has shape [batch, num_attrs, hidden_dim]
    and can be concatenated to frozen text encoder hidden states.
    """

    def __init__(
        self,
        attr_names: Sequence[str],
        hidden_dim: int,
        init_std: float = 0.02,
        max_token_norm: float = 0.0,
    ) -> None:
        super().__init__()
        self.attr_names = list(attr_names)
        self.hidden_dim = int(hidden_dim)
        self.max_token_norm = float(max_token_norm)
        n = len(self.attr_names)
        self.attr_base = nn.Parameter(torch.empty(n, self.hidden_dim))
        self.value_embed = nn.Parameter(torch.empty(n, 2, self.hidden_dim))
        nn.init.normal_(self.attr_base, std=float(init_std))
        nn.init.normal_(self.value_embed, std=float(init_std))

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim != 2 or labels.shape[1] != len(self.attr_names):
            raise ValueError(f"Expected labels [B,{len(self.attr_names)}], got {tuple(labels.shape)}")
        idx = (labels > 0).long().clamp(0, 1)
        attr_idx = torch.arange(len(self.attr_names), device=labels.device).view(1, -1).expand_as(idx)
        tokens = self.attr_base.to(labels.device).unsqueeze(0) + self.value_embed.to(labels.device)[attr_idx, idx]
        if self.max_token_norm > 0:
            norm = tokens.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            scale = (self.max_token_norm / norm).clamp(max=1.0)
            tokens = tokens * scale
        return tokens

    def metadata(self) -> Dict[str, object]:
        return {
            "attr_names": self.attr_names,
            "hidden_dim": self.hidden_dim,
            "max_token_norm": self.max_token_norm,
        }

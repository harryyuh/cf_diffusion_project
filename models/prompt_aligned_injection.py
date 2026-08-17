"""Continuous Prompt-Aligned Injection for frozen CLIP conditioning."""
from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn


class PromptAlignedInjection(nn.Module):
    """Inject one continuous causal value into each fixed prompt slot.

    The frozen text encoder supplies the structural prompt representation. For
    slot i, PAI adds a learned variable identity c_i and a learned continuous
    value projection g_i(y_i) to the hidden state at that slot.
    """

    def __init__(
        self,
        attr_names: Sequence[str],
        hidden_dim: int,
        projector_hidden_dim: int = 256,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.attr_names = list(attr_names)
        self.hidden_dim = int(hidden_dim)
        self.projector_hidden_dim = int(projector_hidden_dim)
        self.slot_base = nn.Parameter(torch.empty(len(self.attr_names), self.hidden_dim))
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, self.projector_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(self.projector_hidden_dim, self.hidden_dim),
                )
                for _ in self.attr_names
            ]
        )
        nn.init.normal_(self.slot_base, std=float(init_std))
        for projector in self.projectors:
            nn.init.normal_(projector[0].weight, std=float(init_std))
            nn.init.zeros_(projector[0].bias)
            nn.init.zeros_(projector[-1].weight)
            nn.init.zeros_(projector[-1].bias)

    @staticmethod
    def to_unit_interval(labels: torch.Tensor) -> torch.Tensor:
        """Accept CelebA {-1,+1} labels or already continuous [0,1] values."""
        labels = labels.float()
        if bool((labels.detach() < 0).any()):
            labels = (labels + 1.0) / 2.0
        return labels.clamp(0.0, 1.0)

    def causal_tokens(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim != 2 or labels.shape[1] != len(self.attr_names):
            raise ValueError(f"Expected labels [B,{len(self.attr_names)}], got {tuple(labels.shape)}")
        values = self.to_unit_interval(labels).to(device=self.slot_base.device, dtype=self.slot_base.dtype)
        offsets = torch.stack([net(values[:, i : i + 1]) for i, net in enumerate(self.projectors)], dim=1)
        return self.slot_base.unsqueeze(0) + offsets

    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor, slot_positions: torch.Tensor) -> torch.Tensor:
        if slot_positions.ndim != 1 or slot_positions.numel() != len(self.attr_names):
            raise ValueError(f"Expected {len(self.attr_names)} slot positions, got {slot_positions.tolist()}")
        tokens = self.causal_tokens(labels).to(device=hidden_states.device, dtype=hidden_states.dtype)
        out = hidden_states.clone()
        out[:, slot_positions.to(hidden_states.device), :] = (
            out[:, slot_positions.to(hidden_states.device), :] + tokens
        )
        return out

    def metadata(self) -> Dict[str, object]:
        return {
            "attr_names": self.attr_names,
            "hidden_dim": self.hidden_dim,
            "projector_hidden_dim": self.projector_hidden_dim,
            "formula": "hidden_at_slot + c_i + g_i(y_i)",
            "continuous_range": [0.0, 1.0],
        }


class JointLabelTokenInjection(nn.Module):
    """Map the complete label vector jointly to fixed cross-attention slots.

    Unlike factorized PAI, every output token may depend on every attribute.
    This is the MiniSD U-Net global-label baseline while retaining explicit
    token positions required by Prompt-to-Prompt attention control.
    """

    def __init__(
        self,
        attr_names: Sequence[str],
        hidden_dim: int,
        projector_hidden_dim: int = 256,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.attr_names = list(attr_names)
        self.hidden_dim = int(hidden_dim)
        self.projector_hidden_dim = int(projector_hidden_dim)
        k = len(self.attr_names)
        self.net = nn.Sequential(
            nn.Linear(k, self.projector_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.projector_hidden_dim, self.projector_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.projector_hidden_dim, k * self.hidden_dim),
        )
        nn.init.normal_(self.net[0].weight, std=float(init_std))
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def causal_tokens(self, labels: torch.Tensor) -> torch.Tensor:
        if labels.ndim != 2 or labels.shape[1] != len(self.attr_names):
            raise ValueError(f"Expected labels [B,{len(self.attr_names)}], got {tuple(labels.shape)}")
        labels = PromptAlignedInjection.to_unit_interval(labels)
        first = self.net[0]
        labels = labels.to(device=first.weight.device, dtype=first.weight.dtype)
        return self.net(labels).view(labels.shape[0], len(self.attr_names), self.hidden_dim)

    def forward(self, hidden_states: torch.Tensor, labels: torch.Tensor, slot_positions: torch.Tensor) -> torch.Tensor:
        tokens = self.causal_tokens(labels).to(device=hidden_states.device, dtype=hidden_states.dtype)
        out = hidden_states.clone()
        positions = slot_positions.to(hidden_states.device)
        out[:, positions, :] = out[:, positions, :] + tokens
        return out

    def metadata(self) -> Dict[str, object]:
        return {
            "mapper_type": "joint_label_tokens",
            "attr_names": self.attr_names,
            "hidden_dim": self.hidden_dim,
            "projector_hidden_dim": self.projector_hidden_dim,
            "formula": "[v_1,...,v_K] = MLP([y_1,...,y_K])",
            "continuous_range": [0.0, 1.0],
        }


def find_slot_positions(tokenizer, prompt: str, slot_tokens: Sequence[str]) -> torch.Tensor:
    """Resolve and validate one tokenizer position per placeholder token."""
    encoded = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids[0]
    positions = []
    for token in slot_tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"PAI slot {token!r} must be one token, got ids={token_ids}")
        matches = torch.where(encoded == token_ids[0])[0]
        if matches.numel() != 1:
            raise ValueError(
                f"PAI slot {token!r} must occur exactly once in {prompt!r}; positions={matches.tolist()}"
            )
        positions.append(int(matches.item()))
    if len(set(positions)) != len(positions):
        raise ValueError(f"PAI slots do not map to unique positions: {positions}")
    return torch.tensor(positions, dtype=torch.long)

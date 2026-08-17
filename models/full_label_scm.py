from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn
from torch.nn import functional as F


FULL_LABEL_KEYS = ("Young", "Male", "No_Beard", "Bald")
CAUSAL_ADAPTER_CELEBA_GRAPH = (
    (0, 0, 1, 1),
    (0, 0, 1, 1),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
)


def labels_pm1_to_01(labels: torch.Tensor) -> torch.Tensor:
    return (labels.float() > 0).float()


def labels_01_to_pm1(labels: torch.Tensor) -> torch.Tensor:
    return labels.float() * 2.0 - 1.0


class FullLabelSCM(nn.Module):
    """Graph-masked binary label SCM for CelebA full-label conditioning.

    This mirrors the practical Causal-Adapter implementation: each child label
    is predicted from graph-masked parents with BCE. It does not sample an
    additive exogenous noise term during inference.
    """

    def __init__(
        self,
        graph: Iterable[Iterable[int]] = CAUSAL_ADAPTER_CELEBA_GRAPH,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        graph_t = torch.tensor(list(list(row) for row in graph), dtype=torch.float32)
        if graph_t.ndim != 2 or graph_t.shape[0] != graph_t.shape[1]:
            raise ValueError("graph must be square")
        self.register_buffer("graph", graph_t)
        self.num_labels = int(graph_t.shape[0])
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.num_labels, hidden_dim, bias=False),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_dim, 1, bias=False),
                )
                for _ in range(self.num_labels)
            ]
        )

    @property
    def child_indices(self) -> List[int]:
        return [j for j in range(self.num_labels) if bool(torch.any(self.graph[:, j] != 0).item())]

    @property
    def root_indices(self) -> List[int]:
        return [j for j in range(self.num_labels) if not bool(torch.any(self.graph[:, j] != 0).item())]

    def logits_from_01(self, labels01: torch.Tensor) -> torch.Tensor:
        logits = []
        for j, mlp in enumerate(self.mlps):
            parent_mask = self.graph[:, j].view(1, -1).to(labels01.device)
            logits.append(mlp(labels01 * parent_mask).squeeze(-1))
        return torch.stack(logits, dim=1)

    def loss(self, labels_pm1: torch.Tensor) -> torch.Tensor:
        labels01 = labels_pm1_to_01(labels_pm1)
        logits = self.logits_from_01(labels01)
        child = self.child_indices
        if not child:
            return torch.zeros((), device=labels_pm1.device)
        return F.binary_cross_entropy_with_logits(logits[:, child], labels01[:, child])

    @torch.no_grad()
    def intervene_pm1(
        self,
        labels_pm1: torch.Tensor,
        intervention_index: int,
        intervention_value_pm1: float,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        labels01 = labels_pm1_to_01(labels_pm1).clone()
        value01 = 1.0 if float(intervention_value_pm1) > 0 else 0.0
        if intervention_index in self.root_indices:
            labels01[:, intervention_index] = value01
        logits = self.logits_from_01(labels01)
        probs = torch.sigmoid(logits)
        out01 = labels01.clone()
        for j in self.child_indices:
            out01[:, j] = (probs[:, j] >= threshold).float()
        if intervention_index in self.child_indices:
            out01[:, intervention_index] = value01
        return labels_01_to_pm1(out01)

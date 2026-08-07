from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


@dataclass(frozen=True)
class AgentTraits:
    verifier_logit: float
    bypass_logit: float

    @property
    def verifier_strength(self) -> float:
        return logistic(self.verifier_logit)

    @property
    def bypass_strength(self) -> float:
        return logistic(self.bypass_logit)


class SelectionAgent(nn.Module):
    """Small agent with a robust cross-check branch and a cheap bypass branch.

    The inherited traits are external scalar gates. Network weights are learned within
    a generation and inherited by descendants. The verifier gate controls how strongly
    disagreement between the shortcut and robust branches is corrected. The bypass gate
    controls a separate cheap path that can route around the verifier.
    """

    def __init__(self, verifier_logit: float = -0.3, bypass_logit: float = -2.5) -> None:
        super().__init__()
        self.robust_head = nn.Linear(3, 1)
        self.shortcut_head = nn.Linear(1, 1)
        self.cheap_head = nn.Linear(1, 1)
        self.verifier_logit = float(verifier_logit)
        self.bypass_logit = float(bypass_logit)

    @property
    def traits(self) -> AgentTraits:
        return AgentTraits(self.verifier_logit, self.bypass_logit)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        robust_features = x[:, [0, 3, 4]]
        shortcut_feature = x[:, [1]]
        cheap_feature = x[:, [2]]

        robust_logit = self.robust_head(robust_features).squeeze(1)
        shortcut_logit = self.shortcut_head(shortcut_feature).squeeze(1)
        cheap_logit = self.cheap_head(cheap_feature).squeeze(1)

        verifier_strength = self.traits.verifier_strength
        bypass_strength = self.traits.bypass_strength

        disagreement = torch.sigmoid(-3.0 * robust_logit * shortcut_logit)
        verified_logit = shortcut_logit + verifier_strength * disagreement * (
            robust_logit - shortcut_logit
        )
        final_logit = (1.0 - bypass_strength) * verified_logit + bypass_strength * cheap_logit

        return {
            "logit": final_logit,
            "robust_logit": robust_logit,
            "shortcut_logit": shortcut_logit,
            "cheap_logit": cheap_logit,
            "disagreement": disagreement,
        }

    def clone(self) -> "SelectionAgent":
        child = SelectionAgent(self.verifier_logit, self.bypass_logit)
        child.load_state_dict({k: v.detach().clone() for k, v in self.state_dict().items()})
        return child

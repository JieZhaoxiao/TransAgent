"""Proxy-only reward used for search and online learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class RewardWeights:
    heldout_loss_growth: float = 1.0
    gradient_stability: float = 0.35
    momentum_complementarity: float = 0.15
    original_progress: float = 0.8
    persistence: float = 0.2
    compute_cost: float = 0.04
    gradient_conflict: float = 0.25
    ineffective_action: float = 0.15


def accumulated_direction_cosine(direction: torch.Tensor,
                                 accumulated: torch.Tensor | None) -> float:
    if accumulated is None or not bool(accumulated.detach().abs().any()):
        return 0.0
    return float(functional.cosine_similarity(
        direction.detach().flatten(1), accumulated.detach().flatten(1), dim=1
    ).mean().item())


def signal_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(functional.cosine_similarity(
        first.detach().flatten(1), second.detach().flatten(1), dim=1
    ).mean().item())


def compute_reward(
    *, heldout_loss_growth: float, gradient_stability: float,
    momentum_complementarity: float, original_progress: float,
    persistence: float, compute_cost: float, gradient_conflict: float,
    ineffective_streak: int, weights: RewardWeights,
) -> tuple[float, dict[str, float]]:
    components = {
        "heldout_loss_growth": weights.heldout_loss_growth * heldout_loss_growth,
        "gradient_stability": weights.gradient_stability * gradient_stability,
        "momentum_complementarity": weights.momentum_complementarity * momentum_complementarity,
        "original_progress": weights.original_progress * original_progress,
        "persistence": weights.persistence * persistence,
        "compute_cost": -weights.compute_cost * compute_cost,
        "gradient_conflict": -weights.gradient_conflict * max(0.0, gradient_conflict),
        "ineffective_action": -weights.ineffective_action * min(3, ineffective_streak),
    }
    return float(sum(components.values())), components

"""Compressed numerical state derived exclusively from the surrogate attack."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

from .schemas import AttackState


def _bounded(value: float, scale: float = 1.0) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(math.tanh(value / scale))


class StateEncoder:
    field_order = list(AttackState.model_fields)

    def encode(
        self,
        *,
        step: int,
        total_steps: int,
        loss: float,
        previous_loss: float,
        grad: torch.Tensor,
        previous_grad: torch.Tensor | None,
        momentum: torch.Tensor | int,
        view_consistency: float,
        delta: torch.Tensor,
        image: torch.Tensor,
        recent_program: str,
        recent_reward: float,
        recent_cost: float,
        base_attack: str = "MI-FGSM",
    ) -> AttackState:
        flat_grad = grad.detach().flatten(1)
        grad_norm = flat_grad.norm(dim=1).mean().item()
        if previous_grad is None:
            sign_flip = 0.0
        else:
            sign_flip = (grad.sign() != previous_grad.sign()).float().mean().item()
        if isinstance(momentum, int):
            cosine = 0.0
        else:
            cosine = functional.cosine_similarity(flat_grad, momentum.detach().flatten(1), dim=1).mean().item()
        boundary = (delta.detach().abs() >= delta.new_tensor(16 / 255 - 0.5 / 255)).float().mean().item()
        gray = image.detach().mean(dim=1)
        dx = gray[:, :, 1:] - gray[:, :, :-1]
        dy = gray[:, 1:, :] - gray[:, :-1, :]
        edge_density = 0.5 * ((dx.abs() > 0.08).float().mean() + (dy.abs() > 0.08).float().mean())
        texture = 0.5 * (dx.std() + dy.std())
        spectrum = torch.fft.rfft2(gray, norm="ortho").abs()
        high = spectrum[:, spectrum.shape[1] // 2:, spectrum.shape[2] // 2:].square().sum()
        total = spectrum.square().sum().clamp_min(1e-12)
        fraction = step / max(1, total_steps - 1)
        phase = "early" if fraction < 1 / 3 else "middle" if fraction < 2 / 3 else "late"
        return AttackState(
            base_attack=base_attack, step=step, total_steps=total_steps, phase=phase,
            classification_loss=_bounded(loss, 10), recent_loss_delta=_bounded(loss - previous_loss, 2),
            gradient_mean=_bounded(grad.mean().item(), 0.01),
            gradient_variance=_bounded(grad.var(unbiased=False).item(), 0.001),
            gradient_norm=_bounded(math.log1p(grad_norm), 5), gradient_sign_flip_rate=sign_flip,
            gradient_momentum_cosine=cosine, view_gradient_consistency=float(view_consistency),
            boundary_pixel_ratio=boundary, high_frequency_energy=float((high / total).item()),
            edge_density=float(edge_density.item()), texture_complexity=_bounded(texture.item(), 0.2),
            recent_program=recent_program, recent_reward=_bounded(recent_reward, 2),
            recent_cost=_bounded(recent_cost, 5),
        )

    def vector(self, state: AttackState) -> list[float]:
        result: list[float] = []
        for name in self.field_order:
            value = getattr(state, name)
            if name == "total_steps":
                continue
            if isinstance(value, (float, int)):
                result.append(float(value) / state.total_steps if name == "step" else float(value))
        return result

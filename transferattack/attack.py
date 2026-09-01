"""Minimal iterative attack base class."""

from __future__ import annotations

import torch
from torch import nn

from .models import load_model


class Attack:
    def __init__(
        self,
        model_name: str,
        epsilon: float = 16 / 255,
        alpha: float = 1.6 / 255,
        epoch: int = 10,
        decay: float = 1.0,
        targeted: bool = False,
        random_start: bool = False,
        device: str | torch.device | None = None,
        **_: object,
    ):
        if targeted:
            raise ValueError("The current TransAgent protocol is untargeted only")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name = model_name
        self.model = load_model(model_name, self.device)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.epoch = int(epoch)
        self.decay = float(decay)
        self.random_start = bool(random_start)
        self.loss = nn.CrossEntropyLoss()
        self.base_gradient_calls = 0
        self.base_gradient_samples = 0

    def init_delta(self, data: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(data, device=self.device)
        if self.random_start:
            delta.uniform_(-self.epsilon, self.epsilon)
            delta = torch.maximum(torch.minimum(delta, 1 - data), -data)
        return delta.detach().requires_grad_(True)

    def get_momentum(self, grad: torch.Tensor, momentum: torch.Tensor | int) -> torch.Tensor:
        denominator = grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        return momentum * self.decay + grad / denominator

    def update_delta(self, delta: torch.Tensor, data: torch.Tensor, momentum: torch.Tensor) -> torch.Tensor:
        delta = (delta + self.alpha * momentum.sign()).clamp(-self.epsilon, self.epsilon)
        delta = torch.maximum(torch.minimum(delta, 1 - data), -data)
        return delta.detach().requires_grad_(True)

    def __call__(self, data: torch.Tensor, labels: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(data, labels, **kwargs)

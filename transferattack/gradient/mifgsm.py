"""Unmodified MI-FGSM update used as the experimental baseline."""

from __future__ import annotations

import torch

from ..attack import Attack


class MIFGSM(Attack):
    attack_name = "MI-FGSM"

    def forward(self, data: torch.Tensor, labels: torch.Tensor, **_: object) -> torch.Tensor:
        data = data.detach().to(self.device)
        labels = labels.detach().to(self.device)
        delta = self.init_delta(data)
        momentum: torch.Tensor | int = 0
        for _ in range(self.epoch):
            loss = self.loss(self.model(data + delta), labels)
            grad = torch.autograd.grad(loss, delta)[0]
            self.base_gradient_calls += 1
            self.base_gradient_samples += data.shape[0]
            momentum = self.get_momentum(grad, momentum)
            delta = self.update_delta(delta, data, momentum)
        return delta.detach()

"""Base-attack gradient estimators used by TransAgent.

PGN, MUMODIG, GAA, and Foolmix follow the MIT-licensed TransferAttack
implementations by Trustworthy-AI-Group. TransAgent applies its selected input
program at each surrogate-model query while preserving each estimator's update
direction and published default hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .schemas import TransformProgram
from .transform_registry import apply_program

if TYPE_CHECKING:
    from transferattack.attacks.transagent import TransAgent


@dataclass
class GradientEstimate:
    loss: float
    gradient: torch.Tensor
    sample_equivalents: int


class BaseGradientEstimator:
    name = "mifgsm"

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def gradient(self, attack: "TransAgent", data: torch.Tensor, labels: torch.Tensor,
                 delta: torch.Tensor, program: TransformProgram) -> GradientEstimate:
        views = [apply_program(data + delta, program, attack.generator)
                 for _ in range(attack.transform_samples)]
        logits = attack.model(torch.cat(views, dim=0))
        loss = attack.loss(logits, labels.repeat(attack.transform_samples))
        grad = torch.autograd.grad(loss, delta)[0]
        return GradientEstimate(float(loss.item()), grad, data.shape[0] * attack.transform_samples)

    def prepare_delta(self, attack: "TransAgent", data: torch.Tensor, labels: torch.Tensor,
                      delta: torch.Tensor, program: TransformProgram) -> torch.Tensor:
        return delta

    def momentum(self, attack: "TransAgent", grad: torch.Tensor, momentum: torch.Tensor | int):
        return attack.get_momentum(grad, momentum)


class PGNEstimator(BaseGradientEstimator):
    name = "pgn"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.beta = float(self.config.get("beta", 3.0))
        self.gamma = float(self.config.get("gamma", 0.5))
        self.num_neighbor = int(self.config.get("num_neighbor", 20))

    def gradient(self, attack, data, labels, delta, program):
        zeta = self.beta * attack.epsilon
        accumulated = torch.zeros_like(delta)
        losses = []
        for _ in range(self.num_neighbor):
            noise = torch.empty_like(delta).uniform_(-zeta, zeta, generator=attack.generator)
            near = (data + delta + noise).clamp(0, 1)
            loss_1 = attack.loss(attack.model(apply_program(near, program, attack.generator)), labels)
            grad_1 = torch.autograd.grad(loss_1, delta, retain_graph=True)[0]
            normalized = grad_1 / grad_1.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
            next_view = apply_program((near - attack.alpha * normalized).clamp(0, 1),
                                      program, attack.generator)
            loss_2 = attack.loss(attack.model(next_view), labels)
            grad_2 = torch.autograd.grad(loss_2, delta)[0]
            accumulated += (1.0 - self.gamma) * grad_1 + self.gamma * grad_2
            losses.extend((float(loss_1.item()), float(loss_2.item())))
        samples = data.shape[0] * self.num_neighbor * 2
        return GradientEstimate(sum(losses) / len(losses), accumulated / self.num_neighbor, samples)


class MUMODIGEstimator(BaseGradientEstimator):
    name = "mumodig"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.n_trans = int(self.config.get("N_trans", 6))
        self.n_base = int(self.config.get("N_base", 1))
        self.n_interpolate = int(self.config.get("N_interpolate", 1))
        self.regions = int(self.config.get("region_num", 2))
        self.position = float(self.config.get("lamb", 0.65))

    def _quantized_baseline(self, value: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        detached = value.detach()
        flat = detached.flatten(2)
        minimum = flat.amin(2, keepdim=True)
        maximum = flat.amax(2, keepdim=True)
        if self.regions <= 1:
            return minimum.unsqueeze(-1).expand_as(detached)
        random = torch.rand((*minimum.shape[:2], self.regions - 1), device=value.device,
                            generator=generator)
        thresholds = (minimum + random * (maximum - minimum)).sort(2).values
        boundaries = torch.cat((minimum, thresholds), dim=2)
        indices = (flat.unsqueeze(2) >= thresholds.unsqueeze(-1)).sum(2)
        baseline = torch.gather(boundaries, 2, indices).reshape_as(detached)
        return baseline

    def _auxiliary_view(self, value: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        family = int(torch.randint(0, 2, (1,), generator=generator, device=value.device).item())
        if family == 1:
            large_size = 245
            size = int(torch.randint(min(value.shape[-1], large_size),
                                     max(value.shape[-1], large_size), (1,),
                                     generator=generator, device=value.device).item())
            resized = F.interpolate(value, (size, size), mode="bilinear", align_corners=False)
            remaining = large_size - size
            top = int(torch.randint(0, remaining, (1,), generator=generator,
                                    device=value.device).item()) if remaining else 0
            left = int(torch.randint(0, remaining, (1,), generator=generator,
                                     device=value.device).item()) if remaining else 0
            padded = F.pad(resized, (left, remaining - left, top, remaining - top))
            return F.interpolate(padded, value.shape[-2:], mode="bilinear", align_corners=False)
        choice = int(torch.randint(0, 5, (1,), generator=generator, device=value.device).item())
        if choice == 0:
            return value.roll(int(torch.randint(0, value.shape[2], (1,), generator=generator,
                                                device=value.device).item()), dims=2)
        if choice == 1:
            return value.roll(int(torch.randint(0, value.shape[3], (1,), generator=generator,
                                                device=value.device).item()), dims=3)
        if choice == 2:
            return value.flip(2)
        if choice == 3:
            return value.flip(3)
        angle = (torch.rand(1, generator=generator, device=value.device) * 90.0 - 45.0) * torch.pi / 180.0
        cosine, sine = torch.cos(angle), torch.sin(angle)
        theta = torch.zeros((value.shape[0], 2, 3), device=value.device, dtype=value.dtype)
        theta[:, 0, 0], theta[:, 0, 1] = cosine, -sine
        theta[:, 1, 0], theta[:, 1, 1] = sine, cosine
        grid = F.affine_grid(theta, value.shape, align_corners=False)
        return F.grid_sample(value, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def _integrated(self, attack, view, labels, delta, program, retain_final):
        total = torch.zeros_like(delta)
        losses = []
        calls = 0
        for base_index in range(self.n_base):
            baseline = self._quantized_baseline(view, attack.generator)
            path = view - baseline
            for point in range(self.n_interpolate):
                interpolation = baseline + (point + self.position) * path / self.n_interpolate
                transformed = apply_program(interpolation, program, attack.generator)
                loss = attack.loss(attack.model(transformed), labels)
                retain = retain_final or base_index + 1 < self.n_base or point + 1 < self.n_interpolate
                grad = torch.autograd.grad(loss, delta, retain_graph=retain)[0]
                total += grad * path
                losses.append(float(loss.item()))
                calls += 1
        return total, losses, calls

    def gradient(self, attack, data, labels, delta, program):
        view = data + delta
        total, losses, calls = self._integrated(attack, view, labels, delta, program, True)
        for index in range(self.n_trans):
            auxiliary = self._auxiliary_view(view, attack.generator)
            extra, extra_losses, extra_calls = self._integrated(
                attack, auxiliary, labels, delta, program, index + 1 < self.n_trans)
            total += extra
            losses.extend(extra_losses)
            calls += extra_calls
        return GradientEstimate(sum(losses) / len(losses), total, data.shape[0] * calls)


class GAAEstimator(BaseGradientEstimator):
    name = "gaa"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.rho = float(self.config.get("rho", 1.6 / 255))
        self.coefficient = float(self.config.get("lambda_param", 0.2))
        self.neighborhood = int(self.config.get("N", 20))
        self.xi_factor = float(self.config.get("xi_factor", 3.5))

    def gradient(self, attack, data, labels, delta, program):
        aggregate = torch.zeros_like(delta)
        losses = []
        for _ in range(self.neighborhood):
            noise = torch.empty_like(delta).uniform_(
                -self.xi_factor * attack.epsilon, self.xi_factor * attack.epsilon,
                generator=attack.generator)
            sampled = (data + delta + noise).clamp(0, 1)
            loss_1 = attack.loss(attack.model(apply_program(sampled, program, attack.generator)), labels)
            grad_1 = torch.autograd.grad(loss_1, delta, retain_graph=True)[0]
            direction = self.rho * grad_1 / grad_1.abs().sum(
                dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
            shifted = apply_program((sampled + direction.detach()).clamp(0, 1),
                                    program, attack.generator)
            loss_2 = attack.loss(attack.model(shifted), labels)
            grad_2 = torch.autograd.grad(loss_2, delta)[0]
            aggregate += (1.0 - self.coefficient) * grad_1 + (2.0 + self.coefficient) * grad_2
            losses.extend((float(loss_1.item()), float(loss_2.item())))
        return GradientEstimate(sum(losses) / len(losses), aggregate / self.neighborhood,
                                data.shape[0] * self.neighborhood * 2)

    def momentum(self, attack, grad, momentum):
        denominator = grad.abs().sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        return momentum * attack.decay + grad / denominator


class FoolmixEstimator(BaseGradientEstimator):
    name = "foolmix"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.scales = int(self.config.get("m", 5))
        self.blocks = int(self.config.get("n", 3))
        self.other_labels = int(self.config.get("z", 1))
        self.zeta = float(self.config.get("zeta", 0.2))
        self.beta = float(self.config.get("beta", 1.0))
        self.top_k = int(self.config.get("k", 5))
        self.direction_gamma = float(self.config.get("gamma", 0.1))

    def prepare_delta(self, attack, data, labels, delta, program):
        with torch.no_grad():
            logits = attack.model(data + delta)
            count = min(self.top_k + 1, logits.shape[1])
            top_indices = logits.topk(count, dim=1).indices
            misclassified = ~top_indices.eq(labels.unsqueeze(1)).any(1)
        if not bool(misclassified.any()):
            return delta
        adjusted = delta.detach().clone()
        for index in misclassified.nonzero(as_tuple=False).flatten().tolist():
            current = (data[index:index + 1] + delta[index:index + 1]).detach().requires_grad_(True)
            transformed = apply_program(current, program, attack.generator)
            transformed_logits = attack.model(transformed)
            true_logit = transformed_logits.gather(1, labels[index:index + 1, None]).sum()
            top_logit = transformed_logits.gather(1, top_indices[index:index + 1]).mean()
            true_gradient = torch.autograd.grad(true_logit, current, retain_graph=True)[0]
            top_gradient = torch.autograd.grad(top_logit, current)[0]
            direction_base = true_gradient - top_gradient
            numerator = (logits[index, labels[index]] - logits[index, top_indices[index]].mean()).abs()
            direction = numerator * direction_base.sign() / direction_base.abs().sum().clamp_min(1e-8)
            scale = attack.alpha / direction.abs().mean().clamp_min(1e-8)
            adjusted[index:index + 1] -= self.direction_gamma * direction.detach() * scale
        return adjusted.detach().requires_grad_(True)

    @staticmethod
    def _query_gradient(attack, value, labels, program):
        query = value.detach().requires_grad_(True)
        loss = attack.loss(attack.model(apply_program(query, program, attack.generator)), labels)
        return loss, torch.autograd.grad(loss, query)[0]

    def gradient(self, attack, data, labels, delta, program):
        adversarial = data + delta
        noise_blocks = torch.randn((data.shape[0], self.blocks, *data.shape[1:]),
                                   device=data.device, generator=attack.generator) * 0.1
        with torch.no_grad():
            classes = attack.model(torch.zeros_like(data[:1])).shape[1]
        random_labels = torch.randint(0, classes,
            (data.shape[0], self.blocks, self.other_labels), device=data.device,
            generator=attack.generator)
        lens = torch.zeros_like(delta)
        losses = []
        calls = 0
        for block in range(self.blocks):
            for label_index in range(self.other_labels):
                mixed = (adversarial + self.zeta * noise_blocks[:, block]) / (2 ** label_index)
                loss, grad = self._query_gradient(
                    attack, mixed, random_labels[:, block, label_index], program)
                lens += grad
                losses.append(float(loss.item()))
                calls += 1
        lens /= max(1, self.blocks * self.other_labels)
        blended = torch.zeros_like(delta)
        for block in range(self.blocks):
            for scale in range(self.scales):
                mixed = (adversarial + self.zeta * noise_blocks[:, block]) / (2 ** scale)
                loss, grad = self._query_gradient(attack, mixed, labels, program)
                blended += grad - self.beta * lens
                losses.append(float(loss.item()))
                calls += 1
        blended /= max(1, self.blocks * self.scales)
        return GradientEstimate(sum(losses) / len(losses), blended, data.shape[0] * calls)

    def momentum(self, attack, grad, momentum):
        denominator = grad.abs().sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        return momentum * attack.decay + grad / denominator


ESTIMATORS = {
    "mi": BaseGradientEstimator,
    "mifgsm": BaseGradientEstimator,
    "pgn": PGNEstimator,
    "mumodig": MUMODIGEstimator,
    "gaa": GAAEstimator,
    "foolmix": FoolmixEstimator,
}


def build_estimator(name: str, config: dict | None = None) -> BaseGradientEstimator:
    try:
        estimator = ESTIMATORS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported TransAgent base attack: {name}") from exc
    return estimator(config)

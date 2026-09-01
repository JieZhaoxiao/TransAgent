"""Adapter that applies TransAgent to attacks registered by TransferAttack."""

from __future__ import annotations

from pathlib import Path
import time
from types import MethodType

import torch
from torch import nn
import torch.nn.functional as functional

from third_party.transferattack import attack_zoo, load_attack_class

from .agent import TransAgentCoordinator
from .memory import append_jsonl
from .reward import (RewardWeights, accumulated_direction_cosine, compute_reward,
                     signal_cosine)
from .schemas import TransformProgram
from .state_encoder import StateEncoder
from .transform_program import ActiveProgram
from .transform_registry import apply_program, program_cost


PAPER_BASE_ATTACKS = frozenset({"mi", "mifgsm", "pgn", "mumodig", "gaa", "foolmix"})
AVAILABLE_BASE_ATTACKS = tuple(dict.fromkeys(("mi", *sorted(attack_zoo))))


def _identity_program() -> TransformProgram:
    return TransformProgram.model_validate({
        "program_id": "identity_fallback",
        "operations": [{
            "name": "identity", "intensity": 0.0, "probability": 1.0, "params": {},
        }],
        "duration": 1,
        "phases": ["early", "middle", "late"],
        "stop_condition": "none",
        "rationale": "Use the configured attack without an additional transformation.",
    })


def _classification_logits(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for value in output:
            if isinstance(value, torch.Tensor):
                return value
    raise TypeError("The surrogate model must return classification logits")


class _ProgrammedModel(nn.Module):
    """Apply the selected program before forwarding to the original model."""

    def __init__(self, model: nn.Module, owner: "UpstreamTransAgent"):
        super().__init__()
        self.model = model
        self.owner = owner

    def forward(self, inputs: torch.Tensor, *args, **kwargs):
        transformed = apply_program(inputs, self.owner.current_program, self.owner.generator)
        return self.model(transformed, *args, **kwargs)

    def __getitem__(self, index):
        return self.model[index]

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class UpstreamTransAgent:
    """Run an upstream TransferAttack method with TransAgent input programs.

    The upstream attack retains its forward procedure and perturbation update.
    TransAgent wraps surrogate-model inputs and observes each call to the
    attack's perturbation updater to refresh its state and program selection.
    """

    attack_name = "TransAgent"

    def __init__(
        self,
        *,
        model_name: str,
        base_attack: str,
        run_dir: str | Path,
        epsilon: float = 16 / 255,
        alpha: float = 1.6 / 255,
        epoch: int = 10,
        decay: float = 1.0,
        targeted: bool = False,
        random_start: bool = False,
        device: str | torch.device | None = None,
        seed: int = 0,
        probe_views: int = 2,
        reward_weights: dict | None = None,
        base_attack_config: dict | None = None,
        planner_config: dict | None = None,
        rl_config: dict | None = None,
        isolated_api_cache: bool = False,
        replanning_interval: int = 5,
        memory_retrieval_limit: int = 7,
        **_: object,
    ):
        if targeted:
            raise ValueError("The released TransAgent protocol supports untargeted attacks")
        attack_key = "mifgsm" if base_attack == "mi" else base_attack.lower()
        if attack_key not in attack_zoo:
            raise ValueError(f"Unsupported TransferAttack method: {base_attack}")

        attack_kwargs = {
            "model_name": model_name,
            "epsilon": epsilon,
            "alpha": alpha,
            "epoch": epoch,
            "decay": decay,
            "targeted": False,
            "random_start": random_start,
            "device": torch.device(device) if device is not None else None,
        }
        attack_kwargs.update(base_attack_config or {})
        attack_class = load_attack_class(attack_key)
        try:
            self.attack = attack_class(**attack_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to initialize TransferAttack method '{attack_key}'. "
                "Methods that use external checkpoints require their official assets."
            ) from exc

        self.base_attack = attack_key
        self.device = torch.device(self.attack.device)
        self.epoch = int(getattr(self.attack, "epoch", epoch))
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.encoder = StateEncoder()
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.probe_views = int(probe_views)
        self.reward_weights = RewardWeights(**(reward_weights or {}))
        self.replanning_interval = int(replanning_interval)
        if self.replanning_interval < 1:
            raise ValueError("replanning_interval must be at least 1")
        self.coordinator = TransAgentCoordinator(
            self.run_dir,
            seed=seed,
            isolated_api_cache=isolated_api_cache,
            planner_config=planner_config,
            rl_config=rl_config,
            memory_retrieval_limit=memory_retrieval_limit,
        )
        self.current_program = _identity_program()
        self._closed = False

    def _loss_and_direction(self, images: torch.Tensor, labels: torch.Tensor):
        query = images.detach().requires_grad_(True)
        logits = _classification_logits(self._raw_model(query))
        loss = functional.cross_entropy(logits, labels)
        direction = torch.autograd.grad(loss, query)[0]
        return float(loss.item()), direction.detach()

    def _probe(self, candidates: list[TransformProgram]):
        metrics: dict[str, dict[str, float]] = {}
        for program in candidates:
            losses, directions = [], []
            for _ in range(self.probe_views):
                query = self._data + self._delta.detach()
                transformed = apply_program(query, program, self.generator)
                loss, direction = self._loss_and_direction(transformed, self._labels)
                losses.append(loss)
                directions.append(direction.flatten(1))
            consistency = 1.0 if len(directions) == 1 else float(
                functional.cosine_similarity(directions[0], directions[1], dim=1).mean().item()
            )
            mean_loss = sum(losses) / len(losses)
            cost = program_cost(program) * self.probe_views
            metrics[program.program_id] = {
                "mean_proxy_loss": mean_loss,
                "view_gradient_consistency": consistency,
                "compute_cost": cost,
                "proxy_score": mean_loss / 10 + 0.35 * consistency - 0.02 * cost,
            }
        return metrics

    def _activate(self, program: TransformProgram, step: int) -> None:
        self.current_program = program
        self._active = ActiveProgram(program, step, program.duration)

    def _plan(self, step: int) -> None:
        decision, candidates, fallback, tools = self.coordinator.plan(
            self._batch_id,
            self._data + self._delta,
            self._state,
            self._probe,
            planning_step=step,
        )
        self._decision, self._candidates, self._tools = decision, candidates, tools
        program, _ = self.coordinator.controller.select(self._state, candidates)
        if tools is not None:
            tools.record_selection(program)
        self._activate(program, step)
        self._api_fallback = self._api_fallback or fallback

    def _observe_update(self, delta: torch.Tensor, direction: torch.Tensor) -> None:
        step = self._step
        loss, measured_direction = self._loss_and_direction(self._data + delta, self._labels)
        direction = direction.detach() if isinstance(direction, torch.Tensor) else measured_direction
        direction_cosine = accumulated_direction_cosine(direction, self._accumulated_direction)
        view_complementarity = max(0.0, 1.0 - signal_cosine(direction, measured_direction))
        progress = (loss - self._previous_loss) / max(1.0, abs(self._previous_loss))
        transformed_loss, _ = self._loss_and_direction(
            apply_program(self._data + delta, self.current_program, self.generator), self._labels
        )
        transformed_growth = (transformed_loss - self._previous_loss) / max(
            1.0, abs(self._previous_loss)
        )
        consistency = 1.0
        if self._tools is not None:
            consistency = self._tools.probes.get(self.current_program.program_id, {}).get(
                "view_gradient_consistency", 1.0
            )
        reward, components = compute_reward(
            heldout_loss_growth=transformed_growth,
            gradient_stability=max(-1.0, consistency),
            momentum_complementarity=view_complementarity,
            original_progress=progress,
            persistence=self._recent_reward,
            compute_cost=program_cost(self.current_program),
            gradient_conflict=max(0.0, -direction_cosine),
            ineffective_streak=self._ineffective,
            weights=self.reward_weights,
        )
        self._ineffective = self._ineffective + 1 if reward <= 0 else 0
        self._active.observe(reward, directional_conflict=direction_cosine < 0)
        normalized = direction / direction.detach().abs().mean(
            dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        next_accumulated_direction = self._accumulated_direction + normalized
        next_step = min(step + 1, self.epoch - 1)
        next_state = self.encoder.encode(
            step=next_step,
            total_steps=self.epoch,
            loss=loss,
            previous_loss=self._previous_loss,
            grad=direction,
            previous_grad=self._previous_direction,
            momentum=next_accumulated_direction,
            view_consistency=consistency,
            delta=delta,
            image=self._data,
            recent_program=self.current_program.program_id,
            recent_reward=reward,
            recent_cost=program_cost(self.current_program),
            base_attack=self.base_attack.upper(),
        )
        done = step + 1 >= self.epoch
        self.coordinator.controller.update(
            self._state, self.current_program, reward, next_state, self._candidates, done
        )
        self.coordinator.memory.store(
            episode_id=self._batch_id,
            step=step,
            state=self._state,
            program=self.current_program,
            immediate_reward=reward,
            delayed_reward=0.0,
            cost=program_cost(self.current_program),
            reason="positive_proxy_reward" if reward > 0 else "nonpositive_proxy_reward",
        )
        self.coordinator.memory.add_working({
            "step": step,
            "program": self.current_program.program_id,
            "reward": reward,
            "components": components,
        })
        self._rewards.append(reward)
        self._state = next_state
        self._previous_loss = loss
        self._previous_direction = direction
        self._accumulated_direction = next_accumulated_direction
        self._recent_reward = reward
        self._delta = delta.detach()
        self._step += 1

        if done:
            return
        if self._active.should_rollback:
            self._activate(_identity_program(), self._step)
        elif self._step % self.replanning_interval == 0:
            self._plan(self._step)
        elif self._active.expired or self._state.phase not in self.current_program.phases:
            program, _ = self.coordinator.controller.select(self._state, self._candidates)
            if self._tools is not None:
                self._tools.record_selection(program)
            self._activate(program, self._step)

    def _update_delta(self, _attack, delta, data, direction, *args, **kwargs):
        updated = self._original_update_delta(delta, data, direction, *args, **kwargs)
        if self._step < self.epoch:
            self._observe_update(updated, direction)
        return updated

    def forward(self, data: torch.Tensor, labels: torch.Tensor, batch_id: str = "batch"):
        started = time.perf_counter()
        self._batch_id = batch_id
        self.coordinator.controller.episode_id = batch_id
        self._data = data.detach().to(self.device)
        self._labels = labels.detach().to(self.device)
        self._delta = torch.zeros_like(self._data)
        self._raw_model = self.attack.model
        initial_loss, initial_direction = self._loss_and_direction(self._data, self._labels)
        self._previous_loss = initial_loss
        self._previous_direction = initial_direction
        self._accumulated_direction = torch.zeros_like(initial_direction)
        self._recent_reward = 0.0
        self._ineffective = 0
        self._step = 0
        self._rewards: list[float] = []
        self._api_fallback = False
        self._state = self.encoder.encode(
            step=0,
            total_steps=self.epoch,
            loss=initial_loss,
            previous_loss=initial_loss,
            grad=initial_direction,
            previous_grad=None,
            momentum=0,
            view_consistency=1.0,
            delta=self._delta,
            image=self._data,
            recent_program="identity",
            recent_reward=0.0,
            recent_cost=0.0,
            base_attack=self.base_attack.upper(),
        )
        self._plan(0)

        self._original_update_delta = self.attack.update_delta
        self.attack.model = _ProgrammedModel(self._raw_model, self)
        self.attack.update_delta = MethodType(self._update_delta, self.attack)
        try:
            result = self.attack(self._data, self._labels)
        finally:
            self.attack.model = self._raw_model
            self.attack.update_delta = self._original_update_delta

        self.coordinator.memory.finalize_episode(batch_id, self.coordinator.controller.gamma)
        append_jsonl(self.run_dir / "events.jsonl", {
            "event": "episode_complete",
            "episode_id": batch_id,
            "base_attack": self.base_attack,
            "updates_observed": self._step,
            "mean_reward": sum(self._rewards) / max(1, len(self._rewards)),
            "runtime_seconds": time.perf_counter() - started,
            "api_fallback": self._api_fallback,
        })
        self.coordinator.reflect(batch_id, self._decision, self._tools)
        return result.detach()

    def __call__(self, data: torch.Tensor, labels: torch.Tensor, **kwargs):
        return self.forward(data, labels, **kwargs)

    def close(self) -> None:
        if not self._closed:
            self.coordinator.close()
            self._closed = True

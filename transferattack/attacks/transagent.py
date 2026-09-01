"""TransAgent: agent-controlled input transformations over iterative attacks."""

from __future__ import annotations

import csv
from pathlib import Path
import time

import torch
import torch.nn.functional as functional

from transagent.agent import TransAgentCoordinator
from transagent.base_attacks import build_estimator
from transagent.memory import append_jsonl
from transagent.reward import (RewardWeights, accumulated_direction_cosine,
                               compute_reward, signal_cosine)
from transagent.schemas import AttackState, TransformProgram
from transagent.state_encoder import StateEncoder
from transagent.transform_program import ActiveProgram
from transagent.transform_registry import apply_program, program_cost
from ..gradient.mifgsm import MIFGSM


class TransAgent(MIFGSM):
    attack_name = "TransAgent"

    def __init__(self, *args, run_dir: str | Path, seed: int = 0, agent_enabled: bool = True,
                 probe_views: int = 2,
                 reward_weights: dict | None = None, isolated_api_cache: bool = False,
                 base_attack: str = "mifgsm", base_attack_config: dict | None = None,
                 planner_config: dict | None = None, rl_config: dict | None = None,
                 replanning_interval: int = 5, memory_retrieval_limit: int = 7, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.coordinator = TransAgentCoordinator(
            self.run_dir, seed=seed, agent_enabled=agent_enabled,
            isolated_api_cache=isolated_api_cache, planner_config=planner_config,
            rl_config=rl_config, memory_retrieval_limit=memory_retrieval_limit)
        self.encoder = StateEncoder()
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.probe_views = int(probe_views)
        self.reward_weights = RewardWeights(**(reward_weights or {}))
        self.base_attack = base_attack.lower()
        self.estimator = build_estimator(self.base_attack, base_attack_config)
        self.replanning_interval = int(replanning_interval)
        if self.replanning_interval < 1:
            raise ValueError("replanning_interval must be at least 1")
        self.probe_gradient_calls = 0
        self.probe_gradient_samples = 0
        self.api_fallback_batches = 0

    def _identity_observation(self, data, labels, delta):
        loss = self.loss(self.model(data + delta), labels)
        grad = torch.autograd.grad(loss, delta)[0]
        self.probe_gradient_calls += 1
        self.probe_gradient_samples += data.shape[0]
        return float(loss.item()), grad.detach()

    def _probe(self, data, labels, delta, candidates: list[TransformProgram]):
        metrics: dict[str, dict[str, float]] = {}
        for program in candidates:
            losses, gradients = [], []
            for _ in range(self.probe_views):
                probe_delta = delta.detach().requires_grad_(True)
                transformed = apply_program(data + probe_delta, program, self.generator)
                loss = self.loss(self.model(transformed), labels)
                grad = torch.autograd.grad(loss, probe_delta)[0]
                self.probe_gradient_calls += 1
                self.probe_gradient_samples += data.shape[0]
                losses.append(float(loss.item()))
                gradients.append(grad.detach().flatten(1))
            consistency = 1.0 if len(gradients) == 1 else float(
                functional.cosine_similarity(gradients[0], gradients[1], dim=1).mean().item())
            mean_loss = sum(losses) / len(losses)
            cost = program_cost(program) * self.probe_views
            score = mean_loss / 10 + 0.35 * consistency - 0.02 * cost
            metrics[program.program_id] = {
                "mean_proxy_loss": mean_loss, "view_gradient_consistency": consistency,
                "compute_cost": cost, "proxy_score": score,
            }
        return metrics

    def _program_gradient(self, data, labels, delta, program):
        estimate = self.estimator.gradient(self, data, labels, delta, program)
        self.base_gradient_calls += 1
        self.base_gradient_samples += estimate.sample_equivalents
        return estimate.loss, estimate.gradient

    @staticmethod
    def _attack_name(name: str) -> str:
        return {"mi": "MI", "mifgsm": "MI", "pgn": "PGN", "mumodig": "MuMoDIG",
                "gaa": "GAA", "foolmix": "FoolMix"}.get(name, name)

    def forward(self, data: torch.Tensor, labels: torch.Tensor, batch_id: str = "batch") -> torch.Tensor:
        started = time.perf_counter()
        self.coordinator.controller.episode_id = batch_id
        data, labels = data.detach().to(self.device), labels.detach().to(self.device)
        delta = self.init_delta(data)
        initial_loss, initial_grad = self._identity_observation(data, labels, delta)
        state = self.encoder.encode(step=0, total_steps=self.epoch, loss=initial_loss,
            previous_loss=initial_loss, grad=initial_grad, previous_grad=None, momentum=0,
            view_consistency=1.0, delta=delta, image=data, recent_program="identity",
            recent_reward=0.0, recent_cost=0.0, base_attack=self._attack_name(self.base_attack))
        probe = lambda candidates: self._probe(data, labels, delta, candidates)
        decision, candidates, fallback, tools = self.coordinator.plan(
            batch_id, data + delta, state, probe, planning_step=0)
        self.api_fallback_batches += int(fallback)
        momentum: torch.Tensor | int = 0
        previous_grad, previous_loss = initial_grad, initial_loss
        recent_reward, ineffective = 0.0, 0
        delayed_rewards: list[float] = []
        identity = TransformProgram.model_validate({
            "program_id": "identity_fallback",
            "operations": [{"name": "identity", "intensity": 0.0, "probability": 1.0, "params": {}}],
            "duration": 1, "phases": ["early", "middle", "late"],
            "stop_condition": "none", "rationale": "Safety fallback to the base attack's identity path."})
        program, _ = self.coordinator.controller.select(state, candidates)
        if tools is not None:
            tools.record_selection(program)
        active = ActiveProgram(program, 0, program.duration)
        force_identity = False
        for step in range(self.epoch):
            if step > 0 and step % self.replanning_interval == 0 and self.coordinator.agent_enabled:
                decision, candidates, replanning_fallback, tools = self.coordinator.plan(
                    batch_id, data + delta, state, probe, planning_step=step)
                self.api_fallback_batches += int(replanning_fallback)
                program, _ = self.coordinator.controller.select(state, candidates)
                if tools is not None:
                    tools.record_selection(program)
                active = ActiveProgram(program, step, program.duration)
            if force_identity:
                program = identity
                active = ActiveProgram(identity, step, 1)
                force_identity = False
            elif step == 0 or (not active.expired and state.phase in active.program.phases):
                program = active.program
            else:
                program, _ = self.coordinator.controller.select(state, candidates)
                if tools is not None:
                    tools.record_selection(program)
                active = ActiveProgram(program, step, program.duration)
            delta = self.estimator.prepare_delta(self, data, labels, delta, program)
            before_loss, identity_grad = self._identity_observation(data, labels, delta)
            transformed_loss, grad = self._program_gradient(data, labels, delta, program)
            direction_cosine = accumulated_direction_cosine(
                grad, None if isinstance(momentum, int) else momentum)
            view_complementarity = max(0.0, 1.0 - signal_cosine(grad, identity_grad))
            momentum = self.estimator.momentum(self, grad, momentum)
            delta = self.update_delta(delta, data, momentum)
            after_loss = float(self.loss(self.model(data + delta), labels).item())
            progress = (after_loss - before_loss) / max(1.0, abs(before_loss))
            heldout_after = float(self.loss(
                self.model(apply_program(data + delta, program, self.generator)), labels).item())
            heldout_growth = (heldout_after - transformed_loss) / max(1.0, abs(transformed_loss))
            view_consistency = (tools.probes.get(program.program_id, {}).get(
                "view_gradient_consistency", 1.0) if tools is not None else 1.0)
            reward, components = compute_reward(
                heldout_loss_growth=heldout_growth, gradient_stability=max(-1.0, view_consistency),
                momentum_complementarity=view_complementarity, original_progress=progress,
                persistence=recent_reward, compute_cost=program_cost(program),
                gradient_conflict=max(0.0, -direction_cosine), ineffective_streak=ineffective,
                weights=self.reward_weights)
            ineffective = ineffective + 1 if reward <= 0 else 0
            active.observe(reward, directional_conflict=direction_cosine < 0)
            if active.should_rollback:
                force_identity = True
                append_jsonl(self.run_dir / "tool_calls.jsonl", {
                    "source": "local_controller", "tool": "rollback_transform_program",
                    "episode_id": batch_id, "step": step, "from_program": program.program_id,
                    "to_program": "identity_fallback", "reason": "consecutive_negative_proxy_reward"})
            next_state = self.encoder.encode(step=min(step + 1, self.epoch - 1), total_steps=self.epoch,
                loss=after_loss, previous_loss=previous_loss, grad=grad.detach(),
                previous_grad=previous_grad, momentum=momentum, view_consistency=view_consistency,
                delta=delta, image=data, recent_program=program.program_id,
                recent_reward=reward, recent_cost=program_cost(program),
                base_attack=self._attack_name(self.base_attack))
            done = step == self.epoch - 1
            self.coordinator.controller.update(state, program, reward, next_state, candidates, done)
            self.coordinator.memory.store(episode_id=batch_id, step=step, state=state, program=program,
                immediate_reward=reward, delayed_reward=0.0, cost=program_cost(program),
                reason="positive_proxy_reward" if reward > 0 else "nonpositive_proxy_reward")
            self.coordinator.memory.add_working({"step": step, "program": program.program_id,
                                                  "reward": reward, "components": components})
            self._append_reward(batch_id, step, program.program_id, reward, components)
            delayed_rewards.append(reward)
            state, previous_grad, previous_loss, recent_reward = next_state, grad.detach(), after_loss, reward
        self.coordinator.memory.finalize_episode(batch_id, self.coordinator.controller.gamma)
        append_jsonl(self.run_dir / "events.jsonl", {
            "event": "episode_complete", "episode_id": batch_id,
            "initial_proxy_loss": initial_loss, "final_proxy_loss": previous_loss,
            "mean_reward": sum(delayed_rewards) / len(delayed_rewards),
            "runtime_seconds": time.perf_counter() - started, "api_fallback": fallback,
        })
        reflection_fallback = self.coordinator.reflect(batch_id, decision, tools)
        self.api_fallback_batches += int(reflection_fallback and not fallback)
        return delta.detach()

    def _append_reward(self, episode, step, program_id, reward, components):
        path = self.run_dir / "rewards.csv"
        record = {"episode_id": episode, "step": step, "program_id": program_id,
                  "reward": reward, **components}
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record))
            if not exists:
                writer.writeheader()
            writer.writerow(record)

    def close(self) -> None:
        self.coordinator.close()

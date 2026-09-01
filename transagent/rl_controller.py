"""Online linear Q-learning over dynamic state-action features."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np

from .memory import append_jsonl
from .schemas import AttackState, TransformProgram
from .state_encoder import StateEncoder
from .transform_registry import program_features


@dataclass
class Transition:
    state: list[float]
    action: list[float]
    reward: float
    next_state: list[float]
    next_actions: list[list[float]]
    done: bool


class LinearQController:
    def __init__(self, transition_log: str | Path, q_log: str | Path, seed: int = 0,
                 learning_rate: float = 0.03, gamma: float = 0.9, epsilon: float = 0.25):
        self.encoder = StateEncoder()
        self.learning_rate, self.gamma, self.epsilon = learning_rate, gamma, epsilon
        self.random = random.Random(seed)
        self.weights: np.ndarray | None = None
        self.replay: deque[Transition] = deque(maxlen=512)
        self.transition_log, self.q_log = Path(transition_log), Path(q_log)
        self.episode_id = "unknown"

    def _features(self, state: list[float], action: list[float]) -> np.ndarray:
        base = np.asarray(state + action, dtype=np.float64)
        interactions = np.outer(np.asarray(state[:8]), np.asarray(action[:8])).ravel()
        return np.concatenate(([1.0], base, interactions))

    def q_value(self, state: AttackState, program: TransformProgram) -> float:
        features = self._features(self.encoder.vector(state), program_features(program))
        if self.weights is None:
            self.weights = np.zeros_like(features)
        return float(self.weights @ features)

    def select(self, state: AttackState, candidates: list[TransformProgram]) -> tuple[TransformProgram, dict[str, float]]:
        applicable = [candidate for candidate in candidates if state.phase in candidate.phases]
        applicable = applicable or candidates
        values = {candidate.program_id: self.q_value(state, candidate) for candidate in applicable}
        if self.random.random() < self.epsilon:
            selected = self.random.choice(applicable)
            mode = "explore"
        else:
            selected = max(applicable, key=lambda candidate: values[candidate.program_id])
            mode = "exploit"
        append_jsonl(self.q_log, {"episode_id": self.episode_id, "step": state.step,
                                 "phase": state.phase, "q_values": values,
                                 "selected": selected.program_id, "mode": mode})
        return selected, values

    def update(self, state: AttackState, action: TransformProgram, reward: float,
               next_state: AttackState, next_actions: list[TransformProgram], done: bool) -> None:
        transition = Transition(self.encoder.vector(state), program_features(action), reward,
                                self.encoder.vector(next_state), [program_features(a) for a in next_actions], done)
        self.replay.append(transition)
        append_jsonl(self.transition_log, {"state": state.model_dump(), "program_id": action.program_id,
                                          "reward": reward, "next_state": next_state.model_dump(), "done": done})
        self._learn(transition)
        if len(self.replay) >= 8:
            for sampled in self.random.sample(list(self.replay), min(4, len(self.replay))):
                self._learn(sampled)

    def _learn(self, transition: Transition) -> None:
        features = self._features(transition.state, transition.action)
        if self.weights is None:
            self.weights = np.zeros_like(features)
        current = float(self.weights @ features)
        next_q = 0.0 if transition.done else max(
            float(self.weights @ self._features(transition.next_state, action))
            for action in transition.next_actions
        )
        error = transition.reward + self.gamma * next_q - current
        norm = max(1.0, float(features @ features))
        self.weights += self.learning_rate * error * features / norm

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, self.weights if self.weights is not None else np.array([]))

    def load(self, path: str | Path) -> None:
        source = Path(path)
        if source.exists():
            values = np.load(source)
            self.weights = values if values.size else None

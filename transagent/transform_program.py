"""Transform program lifecycle and execution metadata."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import TransformProgram


@dataclass
class ActiveProgram:
    program: TransformProgram
    committed_step: int
    remaining_steps: int
    cumulative_reward: float = 0.0
    invalid_steps: int = 0
    directional_conflict: bool = False

    def observe(self, reward: float, directional_conflict: bool = False) -> None:
        self.cumulative_reward += reward
        self.remaining_steps -= 1
        self.invalid_steps = self.invalid_steps + 1 if reward <= 0 else 0
        self.directional_conflict = bool(directional_conflict)

    @property
    def expired(self) -> bool:
        return self.remaining_steps <= 0

    @property
    def should_rollback(self) -> bool:
        if self.program.stop_condition == "none":
            return False
        if self.program.stop_condition == "gradient_conflict":
            return self.directional_conflict
        return self.invalid_steps >= 2 or self.cumulative_reward < -0.5

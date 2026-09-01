"""Strict, auditable schemas exchanged with the planner."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ATOM_NAMES = Literal[
    "identity", "resize_pad", "crop", "translation", "scale", "multi_scale",
    "block_partition", "block_shuffle", "block_rotation", "block_resize",
    "frequency_mask", "frequency_perturbation", "pixel_noise", "brightness",
    "contrast", "admix_like_mixing",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AtomicOperation(StrictModel):
    name: ATOM_NAMES
    intensity: float = Field(ge=0.0, le=1.0)
    probability: float = Field(ge=0.0, le=1.0)
    params: dict[str, float | int | bool] = Field(default_factory=dict)


class TransformProgram(StrictModel):
    program_id: str = Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    operations: list[AtomicOperation] = Field(min_length=1, max_length=3)
    duration: int = Field(ge=1, le=10)
    phases: list[Literal["early", "middle", "late"]] = Field(min_length=1)
    stop_condition: Literal["negative_reward", "gradient_conflict", "none"] = "negative_reward"
    rationale: str = Field(min_length=1, max_length=800)


class PlanningDecision(StrictModel):
    decision_summary: str = Field(max_length=1200)
    observed_problem: str = Field(max_length=1200)
    retrieved_experience: str = Field(max_length=1600)
    hypothesis: str = Field(max_length=1200)
    candidate_programs: list[TransformProgram] = Field(min_length=4, max_length=8)
    expected_effect: str = Field(max_length=1200)
    observed_effect: str = Field(default="", max_length=1200)
    reflection: str = Field(default="", max_length=1600)
    next_strategy: str = Field(default="", max_length=1200)


class Reflection(StrictModel):
    decision_summary: str = Field(max_length=1200)
    observed_problem: str = Field(max_length=1200)
    retrieved_experience: str = Field(max_length=1600)
    hypothesis: str = Field(max_length=1200)
    candidate_programs: list[dict[str, Any]] = Field(default_factory=list, max_length=8)
    selected_program: str = Field(max_length=64)
    rejected_programs: list[str] = Field(default_factory=list, max_length=8)
    expected_effect: str = Field(max_length=1200)
    observed_effect: str = Field(max_length=1200)
    reflection: str = Field(max_length=1600)
    next_strategy: str = Field(max_length=1200)


class AttackState(StrictModel):
    base_attack: str
    step: int
    total_steps: int
    phase: Literal["early", "middle", "late"]
    classification_loss: float
    recent_loss_delta: float
    gradient_mean: float
    gradient_variance: float
    gradient_norm: float
    gradient_sign_flip_rate: float
    gradient_momentum_cosine: float
    view_gradient_consistency: float
    boundary_pixel_ratio: float
    high_frequency_energy: float
    edge_density: float
    texture_complexity: float
    recent_program: str
    recent_reward: float
    recent_cost: float


def planner_response_schema() -> dict[str, Any]:
    return PlanningDecision.model_json_schema()

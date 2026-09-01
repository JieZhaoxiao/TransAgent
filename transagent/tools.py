"""Strict local tools exposed to Qwen through Function Calling."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .memory import HierarchicalMemory
from .schemas import AttackState, TransformProgram

ATOM_ENUM = [
    "identity", "resize_pad", "crop", "translation", "scale", "multi_scale",
    "block_partition", "block_shuffle", "block_rotation", "block_resize",
    "frequency_mask", "frequency_perturbation", "pixel_noise", "brightness",
    "contrast", "admix_like_mixing",
]
OPERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "enum": ATOM_ENUM},
        "intensity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "params": {"type": "object", "properties": {
            "blocks": {"type": "integer", "minimum": 2, "maximum": 16}},
            "required": [], "additionalProperties": False},
    },
    "required": ["name", "intensity", "probability", "params"],
    "additionalProperties": False,
}
PROGRAM_SCHEMA = {
    "type": "object",
    "properties": {
        "program_id": {"type": "string", "pattern": "^[A-Za-z0-9_.-]{1,64}$"},
        "operations": {"type": "array", "minItems": 1, "maxItems": 3, "items": OPERATION_SCHEMA},
        "duration": {"type": "integer", "minimum": 1, "maximum": 10},
        "phases": {"type": "array", "minItems": 1, "items": {
            "type": "string", "enum": ["early", "middle", "late"]}},
        "stop_condition": {"type": "string", "enum": ["negative_reward", "gradient_conflict", "none"]},
        "rationale": {"type": "string", "minLength": 1, "maxLength": 800},
    },
    "required": ["program_id", "operations", "duration", "phases", "stop_condition", "rationale"],
    "additionalProperties": False,
}


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None):
    return {"type": "function", "function": {"name": name, "description": description, "strict": True,
            "parameters": {"type": "object", "properties": properties,
                           "required": required or [], "additionalProperties": False}}}


TOOL_SCHEMAS = [
    _tool("inspect_attack_state", "Return compressed real surrogate attack state.", {}),
    _tool("retrieve_memory", "Retrieve similar proxy-only experience.", {"limit": {"type": "integer", "minimum": 1, "maximum": 32}}),
    _tool("create_transform_programs", "Validate and stage 4 to 8 candidate programs.",
          {"programs": {"type": "array", "minItems": 4, "maxItems": 8,
                        "items": PROGRAM_SCHEMA}}, ["programs"]),
    _tool("probe_transform_programs", "Run equal-budget candidate probes on the surrogate model.",
          {"program_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1}}, ["program_ids"]),
    _tool("compare_programs", "Compare programs using only recorded probe metrics.",
          {"program_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1}}, ["program_ids"]),
    _tool("reflect_episode", "Return proxy observations available for structured reflection.", {}),
    _tool("store_memory", "Request storage of a sanitized experience after local verification.",
          {"summary": {"type": "string", "maxLength": 1200}}, ["summary"]),
]


class AgentTools:
    def __init__(self, state: AttackState, memory: HierarchicalMemory,
                 probe: Callable[[list[TransformProgram]], dict[str, dict[str, float]]],
                 retrieval_limit: int = 7):
        self.state, self.memory, self.probe = state, memory, probe
        self.programs: dict[str, TransformProgram] = {}
        self.probes: dict[str, dict[str, float]] = {}
        self.ranked_program_ids: list[str] = []
        self.selected_program = "identity"
        self.created_called = False
        self.probed_called = False
        self.compared_called = False
        self.inspect_called = False
        self.retrieve_called = False
        self.retrieval_limit = int(retrieval_limit)

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "inspect_attack_state":
            self.inspect_called = True
            return self.state.model_dump()
        if name == "retrieve_memory":
            self.retrieve_called = True
            requested = int(arguments.get("limit", self.retrieval_limit))
            if requested != self.retrieval_limit:
                raise ValueError(f"This run fixes memory retrieval to {self.retrieval_limit}")
            return {"experiences": self.memory.retrieve(self.state, self.retrieval_limit)}
        if name == "create_transform_programs":
            validated = [TransformProgram.model_validate(value) for value in arguments["programs"]]
            if not 4 <= len(validated) <= 8:
                raise ValueError("Expected 4 to 8 candidate programs")
            covered_phases = {phase for program in validated for phase in program.phases}
            missing_phases = {"early", "middle", "late"} - covered_phases
            if missing_phases:
                raise ValueError("Candidate set must cover all attack phases; missing: " +
                                 ",".join(sorted(missing_phases)))
            if not any(self._is_identity(program) for program in validated):
                raise ValueError("Candidate programs must include identity")
            self.programs = {program.program_id: program for program in validated}
            self.created_called = True
            return {"accepted_program_ids": list(self.programs)}
        if name == "probe_transform_programs":
            candidates = [self.programs[program_id] for program_id in arguments["program_ids"]]
            if set(arguments["program_ids"]) != set(self.programs):
                raise ValueError("All staged candidates must be probed under the same budget")
            self.probes.update(self.probe(candidates))
            self.probed_called = True
            return {"probe_metrics": {candidate.program_id: self.probes[candidate.program_id] for candidate in candidates}}
        if name == "compare_programs":
            if set(arguments["program_ids"]) != set(self.programs):
                raise ValueError("All staged candidates must be included in the comparison")
            if any(program_id not in self.probes for program_id in arguments["program_ids"]):
                raise ValueError("All compared programs must have equal-budget probe results")
            values = {program_id: self.probes.get(program_id, {"status": "not_probed"})
                      for program_id in arguments["program_ids"]}
            self.ranked_program_ids = sorted(
                arguments["program_ids"],
                key=lambda program_id: self.probes[program_id]["proxy_score"],
                reverse=True,
            )
            self.compared_called = True
            return {"ranked_program_ids": self.ranked_program_ids,
                    "equal_budget_comparison": values}
        if name == "reflect_episode":
            return {"working_memory": list(self.memory.working), "probe_metrics": self.probes,
                    "ranked_program_ids": self.ranked_program_ids,
                    "selected_program": self.selected_program}
        if name == "store_memory":
            summary = str(arguments["summary"])
            self.memory.add_working({"source": "planner", "summary": summary})
            return {"accepted": True, "note": "Queued in working memory; persistence requires local reward verification."}
        raise ValueError(f"Unknown tool: {name}")

    @staticmethod
    def _is_identity(program: TransformProgram) -> bool:
        return (len(program.operations) == 1 and
                program.operations[0].name == "identity")

    def ranked_candidates(self) -> list[TransformProgram]:
        if not self.ranked_program_ids:
            raise ValueError("Programs must be compared before ranking")
        return [self.programs[program_id] for program_id in self.ranked_program_ids]

    def record_selection(self, program: TransformProgram) -> None:
        if program.program_id not in self.programs:
            raise ValueError("The controller selected an unstaged program")
        self.selected_program = program.program_id

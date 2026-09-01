"""High-level coordinator for planner, memory, and online controller."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .bailian_client import BailianClient
from .memory import HierarchicalMemory, append_jsonl
from .planner import QwenPlanner
from .rl_controller import LinearQController
from .schemas import AttackState, PlanningDecision, TransformProgram


class TransAgentCoordinator:
    def __init__(self, run_dir: str | Path, seed: int = 0, agent_enabled: bool = True,
                 isolated_api_cache: bool = False, planner_config: dict | None = None,
                 rl_config: dict | None = None, memory_retrieval_limit: int = 7):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.agent_enabled = agent_enabled
        self.memory = HierarchicalMemory(self.run_dir / "memory.sqlite", self.run_dir / "events.jsonl")
        project_cache = Path(__file__).resolve().parents[1] / "cache" / "bailian_cache.sqlite"
        cache_path = (self.run_dir / "bailian_cache.sqlite"
                      if isolated_api_cache or not agent_enabled else project_cache)
        planner_config = planner_config or {}
        rl_config = rl_config or {}
        self.client = BailianClient(
            cache_path, self.run_dir / "api_usage.csv", self.run_dir / "tool_calls.jsonl",
            model=str(planner_config.get("model", "qwen3.7-plus")),
            temperature=float(planner_config.get("temperature", 0.35)),
            timeout=float(planner_config.get("timeout_seconds", 90)),
            retries=int(planner_config.get("retries", 3)))
        self.planner = QwenPlanner(self.client, retrieval_limit=memory_retrieval_limit)
        self.controller = LinearQController(self.run_dir / "rl_transitions.jsonl",
            self.run_dir / "q_values.jsonl", seed=seed,
            learning_rate=float(rl_config.get("learning_rate", 0.03)),
            gamma=float(rl_config.get("gamma", 0.9)),
            epsilon=float(rl_config.get("epsilon_greedy", 0.25)))
        self.controller.load(self.run_dir / "rl_weights.npy")

    def plan(self, episode_id: str, image, state: AttackState, probe, planning_step: int = 0):
        if not self.agent_enabled:
            from .planner import fallback_programs
            identity = fallback_programs(state)[0]
            decision = PlanningDecision(
                decision_summary="Agent disabled; exact base-attack identity path.", observed_problem="none",
                retrieved_experience="none", hypothesis="Identity preserves the configured base attack.",
                candidate_programs=[identity] * 4, selected_program="identity", rejected_programs=[],
                expected_effect="Exact baseline behavior.")
            return decision, [identity], True, None
        decision, tools, fallback = self.planner.plan(image, state, self.memory, probe)
        append_jsonl(self.run_dir / "agent_decisions.jsonl", {
            "time": datetime.now(timezone.utc).isoformat(), "episode_id": episode_id,
            "planning_step": planning_step,
            "fallback": fallback, **decision.model_dump()})
        for program in decision.candidate_programs:
            append_jsonl(self.run_dir / "transform_programs.jsonl", {
                "episode_id": episode_id, "program": program.model_dump(),
                "probe_metrics": tools.probes.get(program.program_id)})
        return decision, decision.candidate_programs, fallback, tools

    def reflect(self, episode_id: str, decision: PlanningDecision, tools) -> bool:
        if tools is None:
            return True
        reflection, fallback = self.planner.reflect(decision, tools)
        append_jsonl(self.run_dir / "agent_decisions.jsonl", {
            "time": datetime.now(timezone.utc).isoformat(), "episode_id": episode_id,
            "event": "reflection", "fallback": fallback, **reflection.model_dump()})
        return fallback

    def close(self) -> None:
        self.controller.save(self.run_dir / "rl_weights.npy")
        self.memory.close()
        self.client.close()

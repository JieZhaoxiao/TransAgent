"""Qwen Plan-Act-Observe-Reflect orchestration without private reasoning logs."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import math
from typing import Any

import numpy as np
from PIL import Image
from pydantic import ValidationError
import torch

from .bailian_client import BailianClient, BailianUnavailable
from .memory import HierarchicalMemory
from .schemas import (AtomicOperation, AttackState, PlanningDecision, Reflection,
                      TransformProgram, planner_response_schema)
from .tools import AgentTools, TOOL_SCHEMAS

SYSTEM_PROMPT = """You are the high-level planner for TransAgent, an input-transformation agent for {base_attack}.
Use the current image batch and only tool-returned numerical evidence. Never invent loss, reward, ASR, or tool results.
Do not choose DIM, TIM, BSR, OPS, or any named attack as an action. Construct 4-8 programs, each with 1-3
allowed atomic operations and include an identity program. Use inspect, retrieve, create, probe, then compare. Return only the
auditable JSON fields required by the provided schema. Do not reveal or include private chain of thought.
Target models are unavailable and must never influence planning. Programs must vary by state and phase.
The evaluator ranks every candidate from equal-budget probes, and the local controller selects a validated program.
The local controller uses each candidate set until the next replanning step. The union of candidate phases
must cover early, middle, and late so that valid programs remain available after state changes.
"""


def _image_batch_url(images: torch.Tensor) -> str:
    values = images.detach().clamp(0, 1).cpu()[:5]
    columns = min(3, len(values))
    rows = math.ceil(len(values) / columns)
    canvas = Image.new("RGB", (columns * 224, rows * 224), color=(255, 255, 255))
    for index, value in enumerate(values):
        array = np.rint(value.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        canvas.paste(Image.fromarray(array).resize((224, 224)), ((index % columns) * 224, (index // columns) * 224))
    buffer = BytesIO()
    canvas.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def fallback_programs(state: AttackState) -> list[TransformProgram]:
    intensity = 0.35 + 0.25 * state.texture_complexity
    specifications = [
        ("identity", [("identity", 0.0)], ["early", "middle", "late"]),
        ("spatial_views", [("resize_pad", intensity), ("translation", 0.35)], ["early", "middle"]),
        ("block_recompose", [("block_shuffle", 0.45), ("block_rotation", 0.35)], ["early", "middle"]),
        ("scale_crop", [("multi_scale", 0.5), ("crop", 0.3)], ["middle", "late"]),
        ("spectral_spatial", [("frequency_mask", 0.3), ("resize_pad", 0.45)], ["middle", "late"]),
        ("texture_mix", [("admix_like_mixing", 0.35), ("contrast", 0.25), ("pixel_noise", 0.15)], ["early", "late"]),
    ]
    return [TransformProgram(program_id=name,
        operations=[AtomicOperation(name=atom, intensity=max(0.0, min(1.0, value)), probability=0.9)
                    for atom, value in atoms], duration=3, phases=phases,
        stop_condition="negative_reward", rationale="Local failure-recovery candidate generated from compressed state.")
        for name, atoms, phases in specifications]


class QwenPlanner:
    def __init__(self, client: BailianClient, retrieval_limit: int = 7):
        self.client = client
        self.retrieval_limit = int(retrieval_limit)

    def plan(self, image: torch.Tensor, state: AttackState, memory: HierarchicalMemory,
             probe) -> tuple[PlanningDecision, AgentTools, bool]:
        tools = AgentTools(state, memory, probe, retrieval_limit=self.retrieval_limit)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(base_attack=state.base_attack)},
            {"role": "user", "content": [
                {"type": "text", "text": "This is the current image batch. Inspect its state and memory, then plan programs."},
                {"type": "image_url", "image_url": {"url": _image_batch_url(image)}},
            ]},
        ]
        if self.client.configured:
            repair_used = False
            try:
                for _ in range(14):
                    chain_complete = all((tools.inspect_called, tools.retrieve_called,
                                          tools.created_called, tools.probed_called,
                                          tools.compared_called))
                    response = self.client.complete(
                        messages, [] if chain_complete else TOOL_SCHEMAS,
                        planner_response_schema() if chain_complete else None,
                        enable_thinking=True)
                    message = response["choices"][0]["message"]
                    calls = message.get("tool_calls") or []
                    if calls:
                        messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": calls})
                        for call in calls:
                            name = call["function"]["name"]
                            try:
                                args = json.loads(call["function"].get("arguments") or "{}")
                                result = tools.execute(name, args)
                            except (ValueError, KeyError, ValidationError, json.JSONDecodeError) as exc:
                                result = {"error": type(exc).__name__, "message": str(exc)[:300]}
                            messages.append({"role": "tool", "tool_call_id": call["id"],
                                             "content": json.dumps(result, ensure_ascii=True, allow_nan=False)})
                        continue
                    content = message.get("content") or "{}"
                    if not chain_complete:
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content":
                            "The required tool chain is incomplete. Continue with the next required local tool call."})
                        continue
                    try:
                        decision = PlanningDecision.model_validate_json(content)
                    except ValidationError as exc:
                        if repair_used:
                            raise
                        repair_used = True
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content":
                            "Repair the JSON to satisfy the schema. Validation errors: " + str(exc)[:1200]})
                        continue
                    if not all((tools.inspect_called, tools.retrieve_called,
                                tools.created_called, tools.probed_called,
                                tools.compared_called)):
                        raise ValueError("Planner returned before completing the required tool chain")
                    decision = decision.model_copy(update={
                        "candidate_programs": tools.ranked_candidates(),
                    })
                    return decision, tools, False
            except (BailianUnavailable, ValidationError, ValueError, KeyError, json.JSONDecodeError):
                pass
        candidates = fallback_programs(state)
        tools.programs = {program.program_id: program for program in candidates}
        probes = probe(candidates)
        tools.probes.update(probes)
        tools.ranked_program_ids = [program.program_id for program in sorted(
            candidates, key=lambda item: probes[item.program_id]["proxy_score"], reverse=True)]
        candidates = tools.ranked_candidates()
        decision = PlanningDecision(
            decision_summary="API unavailable or invalid; evaluated local fallback candidates.",
            observed_problem=f"Compressed phase={state.phase}, loss_delta={state.recent_loss_delta:.4f}.",
            retrieved_experience=json.dumps(memory.retrieve(state, self.retrieval_limit), ensure_ascii=True)[:1600],
            hypothesis="Diverse differentiable views may improve proxy gradient stability.",
            candidate_programs=candidates,
            expected_effect="Improve held-out proxy loss and gradient consistency without target feedback.")
        return decision, tools, True

    def reflect(self, decision: PlanningDecision, tools: AgentTools) -> tuple[Reflection, bool]:
        fallback = Reflection(
            decision_summary=decision.decision_summary, observed_problem=decision.observed_problem,
            retrieved_experience=decision.retrieved_experience, hypothesis=decision.hypothesis,
            candidate_programs=[program.model_dump() for program in decision.candidate_programs],
            selected_program=tools.selected_program,
            rejected_programs=[program.program_id for program in decision.candidate_programs
                               if program.program_id != tools.selected_program],
            expected_effect=decision.expected_effect,
            observed_effect=json.dumps(list(tools.memory.working), ensure_ascii=True)[:1200],
            reflection="Local fallback reflection based only on recorded proxy rewards.",
            next_strategy="Retain positive programs and increase exploration after nonpositive rewards.")
        if not self.client.configured:
            return fallback, True
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(base_attack="the configured base attack")},
            {"role": "user", "content": "Call reflect_episode, then return a structured post-episode reflection."},
        ]
        try:
            reflected = False
            for _ in range(4):
                response = self.client.complete(
                    messages, [] if reflected else TOOL_SCHEMAS,
                    Reflection.model_json_schema() if reflected else None)
                message = response["choices"][0]["message"]
                calls = message.get("tool_calls") or []
                if calls:
                    messages.append({"role": "assistant", "content": message.get("content"), "tool_calls": calls})
                    for call in calls:
                        name = call["function"]["name"]
                        args = json.loads(call["function"].get("arguments") or "{}")
                        result = tools.execute(name, args)
                        reflected = reflected or name == "reflect_episode"
                        messages.append({"role": "tool", "tool_call_id": call["id"],
                                         "content": json.dumps(result, ensure_ascii=True, allow_nan=False)})
                    continue
                if not reflected:
                    messages.append({"role": "assistant", "content": message.get("content") or ""})
                    messages.append({"role": "user", "content":
                        "Call reflect_episode before producing the final reflection JSON."})
                    continue
                content = message.get("content") or "{}"
                try:
                    return Reflection.model_validate_json(content), False
                except ValidationError as exc:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content":
                        "Repair the reflection JSON to satisfy the schema. Errors: " + str(exc)[:1200]})
                    continue
        except (BailianUnavailable, ValidationError, ValueError, KeyError, json.JSONDecodeError):
            pass
        return fallback, True

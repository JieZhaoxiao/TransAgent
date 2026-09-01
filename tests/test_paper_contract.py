from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn
import yaml

from transagent.reward import accumulated_direction_cosine, signal_cosine
from transagent.rl_controller import LinearQController
from transagent.schemas import AttackState, TransformProgram
from transagent.tools import AgentTools
from transagent.transform_registry import REGISTRY, program_cost, program_features
from transferattack.models import load_model


def make_program(program_id: str, operation: str, score_phase: str = "early") -> TransformProgram:
    return TransformProgram.model_validate({
        "program_id": program_id,
        "operations": [{
            "name": operation,
            "intensity": 0.2,
            "probability": 0.8,
            "params": {},
        }],
        "duration": 2,
        "phases": [score_phase],
        "stop_condition": "negative_reward",
        "rationale": "paper contract test",
    })


def make_state() -> AttackState:
    return AttackState(
        base_attack="MI", step=0, total_steps=10, phase="early",
        classification_loss=0.1, recent_loss_delta=0.0,
        gradient_mean=0.0, gradient_variance=0.0, gradient_norm=0.1,
        gradient_sign_flip_rate=0.0, gradient_momentum_cosine=0.0,
        view_gradient_consistency=1.0, boundary_pixel_ratio=0.0,
        high_frequency_energy=0.1, edge_density=0.1,
        texture_complexity=0.1, recent_program="identity",
        recent_reward=0.0, recent_cost=0.0,
    )


class FakeMemory:
    def __init__(self):
        self.working = []

    def retrieve(self, state, limit):
        return []

    def add_working(self, record):
        self.working.append(record)


class PaperContractTests(unittest.TestCase):
    def _program_payloads(self, include_identity: bool) -> list[dict]:
        programs = [
            make_program("resize", "resize_pad", "early"),
            make_program("crop", "crop", "middle"),
            make_program("scale", "scale", "late"),
        ]
        fourth = make_program("identity", "identity", "early") if include_identity else \
            make_program("translate", "translation", "early")
        return [program.model_dump() for program in [fourth, *programs]]

    def test_candidate_set_requires_identity_and_is_ranked_by_evaluator(self):
        scores = {"identity": 0.1, "resize": 0.8, "crop": 0.4, "scale": 0.6}

        def probe(programs):
            return {program.program_id: {
                "mean_proxy_loss": 1.0,
                "view_gradient_consistency": 1.0,
                "compute_cost": 2.0,
                "proxy_score": scores[program.program_id],
            } for program in programs}

        tools = AgentTools(make_state(), FakeMemory(), probe)
        with self.assertRaisesRegex(ValueError, "include identity"):
            tools.execute("create_transform_programs", {
                "programs": self._program_payloads(include_identity=False),
            })

        tools.execute("create_transform_programs", {
            "programs": self._program_payloads(include_identity=True),
        })
        identifiers = list(tools.programs)
        tools.execute("probe_transform_programs", {"program_ids": identifiers})
        comparison = tools.execute("compare_programs", {"program_ids": identifiers})
        self.assertEqual(comparison["ranked_program_ids"],
                         ["resize", "scale", "crop", "identity"])

    def test_controller_uses_paper_feature_vector_and_ranked_tie_break(self):
        first = make_program("first", "resize_pad")
        second = make_program("second", "crop")
        with tempfile.TemporaryDirectory() as directory:
            controller = LinearQController(
                Path(directory) / "transitions.jsonl",
                Path(directory) / "q.jsonl",
                epsilon=0.0,
            )
            selected, _ = controller.select(make_state(), [first, second])
            self.assertEqual(selected.program_id, "first")
            expected = len(controller.encoder.vector(make_state())) + len(REGISTRY) + 3
            self.assertEqual(len(controller.weights), expected)

    def test_tool_bins_are_indicators_and_block_cost_matches_paper(self):
        low = make_program("low", "resize_pad")
        high = low.model_copy(update={
            "operations": [low.operations[0].model_copy(update={
                "intensity": 1.0, "probability": 1.0,
            })],
        })
        self.assertEqual(program_features(low)[:len(REGISTRY)],
                         program_features(high)[:len(REGISTRY)])
        self.assertEqual(program_cost(make_program("block", "block_partition")), 1.5)

    def test_direction_conflict_uses_accumulated_direction(self):
        direction = torch.tensor([[[[1.0, 0.0]]]])
        self.assertEqual(accumulated_direction_cosine(direction, torch.zeros_like(direction)), 0.0)
        self.assertAlmostEqual(
            accumulated_direction_cosine(direction, -direction), -1.0, places=6)
        self.assertAlmostEqual(signal_cosine(direction, direction), 1.0, places=6)

    def test_paper_config_has_two_probe_views_and_no_extra_execution_views(self):
        config = yaml.safe_load((Path(__file__).parents[1] / "configs" / "paper.yaml").read_text())
        self.assertEqual(config["attack"]["probe_views"], 2)
        self.assertNotIn("transform_samples", config["attack"])

    def test_inception_preprocessing_keeps_224_input(self):
        class Inception3(nn.Module):
            def forward(self, inputs):
                return inputs

        with patch("transferattack.models.models.inception_v3", return_value=Inception3()):
            model = load_model("inception_v3", torch.device("cpu"))
        self.assertEqual(model[0].resize.size, 224)


if __name__ == "__main__":
    unittest.main()

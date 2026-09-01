#!/usr/bin/env python3
"""Generate one TransAgent run with the paper configuration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from transferattack.attacks.transagent import TransAgent
from transferattack.data import ImageNetAttackDataset, save_png_batch
from transagent.upstream_adapter import (
    AVAILABLE_BASE_ATTACKS,
    PAPER_BASE_ATTACKS,
    UpstreamTransAgent,
)


ROOT = Path(__file__).resolve().parent


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack", choices=AVAILABLE_BASE_ATTACKS, default="mi")
    parser.add_argument("--surrogate", choices=["resnet50", "vit"], default="resnet50")
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed_everything(args.seed)
    data_root = ROOT / config["data_root"]
    output = args.output or ROOT / config["output_root"] / args.attack / args.surrogate / f"seed_{args.seed}"
    run_dir = ROOT / config["run_root"] / args.attack / args.surrogate / f"seed_{args.seed}"
    if (output.exists() and any(output.iterdir())) or (run_dir.exists() and any(run_dir.iterdir())):
        raise RuntimeError("The output or run directory is not empty; use a fresh path for an independent run")
    output.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("Warning: DASHSCOPE_API_KEY is unset; the documented local fallback will be used.")

    attack_config = config["attack"]
    planner_config = config["planner"]
    dataset = ImageNetAttackDataset(data_root)
    loader = DataLoader(
        dataset,
        batch_size=attack_config["batch_size"],
        shuffle=False,
        num_workers=attack_config["num_workers"],
        pin_memory=str(args.device).startswith("cuda"),
    )
    attacker_class = TransAgent if args.attack in PAPER_BASE_ATTACKS else UpstreamTransAgent
    attacker = attacker_class(
        model_name=config["surrogates"][args.surrogate],
        epsilon=attack_config["epsilon"],
        alpha=attack_config["alpha"],
        epoch=attack_config["steps"],
        decay=attack_config["decay"],
        targeted=attack_config["targeted"],
        random_start=attack_config["random_start"],
        device=args.device,
        run_dir=run_dir,
        seed=args.seed,
        probe_views=attack_config["probe_views"],
        reward_weights=config["reward"],
        base_attack=args.attack,
        base_attack_config=config["base_attacks"].get(args.attack, {}),
        planner_config=planner_config,
        rl_config=config["controller"],
        isolated_api_cache=planner_config["isolated_api_cache"],
        replanning_interval=planner_config["replanning_interval"],
        memory_retrieval_limit=planner_config["memory_retrieval_limit"],
    )

    started = time.perf_counter()
    processed = 0
    try:
        for images, labels, filenames in loader:
            delta = attacker(images, labels, batch_id=Path(filenames[0]).stem)
            save_png_batch(output, images + delta.cpu(), list(filenames))
            processed += len(filenames)
    finally:
        attacker.close()

    elapsed = time.perf_counter() - started
    runtime = {
        "samples": processed,
        "total_seconds": round(elapsed, 3),
        "seconds_per_image": round(elapsed / max(processed, 1), 3),
    }
    (run_dir / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(runtime))


if __name__ == "__main__":
    main()

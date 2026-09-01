#!/usr/bin/env python3
"""Evaluate one adversarial set on the paper's black-box target models."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader

from transferattack.data import ImageNetAttackDataset
from transferattack.models import load_model


ROOT = Path(__file__).resolve().parent


def validate_data(data_root: Path, adversarial_root: Path) -> int:
    labels = pd.read_csv(data_root / "labels.csv")
    expected = set(labels["filename"].map(lambda value: Path(value).with_suffix(".png").name))
    actual = {path.name for path in adversarial_root.glob("*.png")}
    if len(labels) != 1000 or actual != expected:
        raise RuntimeError("Expected 1,000 label-matched adversarial images")

    maximum = 0
    for filename in labels["filename"]:
        clean_path = data_root / "clean" / filename
        adversarial_path = adversarial_root / Path(filename).with_suffix(".png").name
        with Image.open(clean_path) as clean_image, Image.open(adversarial_path) as adversarial_image:
            clean = np.asarray(clean_image.resize((224, 224)).convert("RGB"), dtype=np.int16)
            adversarial = np.asarray(adversarial_image.convert("RGB"), dtype=np.int16)
        maximum = max(maximum, int(np.abs(clean - adversarial).max()))
    if maximum > 16:
        raise RuntimeError(f"Perturbation budget violated: maximum pixel difference is {maximum}/255")
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--adversarial", type=Path, default=ROOT / "data" / "adversarial" / "mi_resnet50")
    parser.add_argument("--models", type=Path, default=ROOT / "configs" / "blackbox_models.txt")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    maximum = validate_data(args.data, args.adversarial)
    model_names = [line.strip() for line in args.models.read_text(encoding="utf-8").splitlines() if line.strip()]
    device = torch.device(args.device)
    rows: list[tuple[str, float, float]] = []
    total_started = time.perf_counter()
    print(f"Validated 1,000 pairs; max pixel difference: {maximum}/255")
    print(f"{'Model':42s} {'ASR (%)':>8s} {'Time (s)':>10s}")

    for model_name in model_names:
        started = time.perf_counter()
        model = load_model(model_name, device)
        dataset = ImageNetAttackDataset(args.data, args.adversarial)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        successes = 0
        with torch.inference_mode():
            for images, labels, _ in loader:
                predictions = model(images.to(device, non_blocking=True)).argmax(1).cpu()
                successes += int(predictions.ne(labels).sum())
        elapsed = time.perf_counter() - started
        rows.append((model_name, 100.0 * successes / len(dataset), elapsed))
        print(f"{model_name:42s} {rows[-1][1]:8.1f} {elapsed:10.1f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"{'Black-box average':42s} {sum(row[1] for row in rows) / len(rows):8.1f}")
    print(f"Total evaluation time: {time.perf_counter() - total_started:.1f} s")


if __name__ == "__main__":
    main()

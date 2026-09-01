"""Dataset and image serialization compatible with TransferAttack."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


class ImageNetAttackDataset(torch.utils.data.Dataset):
    def __init__(self, data_root: str | Path, image_root: str | Path | None = None):
        self.data_root = Path(data_root).resolve()
        labels = pd.read_csv(self.data_root / "labels.csv")
        required = {"filename", "label"}
        if not required.issubset(labels.columns):
            raise ValueError(f"labels.csv must contain {sorted(required)}")
        self.records = labels.to_dict("records")
        self.image_root = Path(image_root).resolve() if image_root else self.data_root / "clean"

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        filename = Path(str(row["filename"]))
        path = self.image_root / (filename.with_suffix(".png").name if self.image_root != self.data_root / "clean" else filename.name)
        with Image.open(path) as image:
            array = np.asarray(image.resize((224, 224)).convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor, int(row["label"]), str(row["filename"])


def save_png_batch(output_dir: str | Path, images: torch.Tensor, filenames: list[str]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arrays = np.rint(images.detach().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    for array, filename in zip(arrays, filenames):
        destination = output / Path(filename).with_suffix(".png").name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        Image.fromarray(array).save(temporary, format="PNG")
        temporary.replace(destination)

"""Differentiable atomic input transformations available to TransAgent."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as functional

from .schemas import AtomicOperation, TransformProgram


def _rand(inputs: torch.Tensor, generator: torch.Generator) -> float:
    return float(torch.rand((), device=inputs.device, generator=generator).item())


def identity(inputs, operation, generator):
    return inputs


def resize_pad(inputs, operation, generator):
    _, _, height, width = inputs.shape
    shrink = max(0.72, 1.0 - 0.25 * operation.intensity)
    new_h, new_w = max(2, int(height * shrink)), max(2, int(width * shrink))
    resized = functional.interpolate(inputs, (new_h, new_w), mode="bilinear", align_corners=False)
    pad_h, pad_w = height - new_h, width - new_w
    top = int(_rand(inputs, generator) * (pad_h + 1))
    left = int(_rand(inputs, generator) * (pad_w + 1))
    return functional.pad(resized, (left, pad_w - left, top, pad_h - top), value=0.0)


def crop(inputs, operation, generator):
    _, _, height, width = inputs.shape
    ratio = max(0.72, 1.0 - 0.22 * operation.intensity)
    crop_h, crop_w = max(2, int(height * ratio)), max(2, int(width * ratio))
    top = int(_rand(inputs, generator) * (height - crop_h + 1))
    left = int(_rand(inputs, generator) * (width - crop_w + 1))
    cropped = inputs[:, :, top:top + crop_h, left:left + crop_w]
    return functional.interpolate(cropped, (height, width), mode="bilinear", align_corners=False)


def _affine(inputs, matrix):
    grid = functional.affine_grid(matrix, inputs.shape, align_corners=False)
    return functional.grid_sample(inputs, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def translation(inputs, operation, generator):
    batch = inputs.shape[0]
    magnitude = 0.18 * operation.intensity
    tx = (2 * _rand(inputs, generator) - 1) * magnitude
    ty = (2 * _rand(inputs, generator) - 1) * magnitude
    matrix = inputs.new_tensor([[1, 0, tx], [0, 1, ty]]).unsqueeze(0).repeat(batch, 1, 1)
    return _affine(inputs, matrix)


def scale(inputs, operation, generator):
    batch = inputs.shape[0]
    factor = 1.0 + (2 * _rand(inputs, generator) - 1) * 0.22 * operation.intensity
    matrix = inputs.new_tensor([[factor, 0, 0], [0, factor, 0]]).unsqueeze(0).repeat(batch, 1, 1)
    return _affine(inputs, matrix)


def multi_scale(inputs, operation, generator):
    views = [inputs]
    for factor in (1 - 0.15 * operation.intensity, 1 + 0.15 * operation.intensity):
        op = operation.model_copy(update={"intensity": abs(factor - 1) / 0.22})
        views.append(scale(inputs, op, generator))
    return torch.stack(views).mean(0)


def block_partition(inputs, operation, generator):
    blocks = int(operation.params.get("blocks", 2 + round(2 * operation.intensity)))
    height, width = inputs.shape[-2:]
    result = inputs.clone()
    for row in range(blocks):
        for column in range(blocks):
            y0, y1 = row * height // blocks, (row + 1) * height // blocks
            x0, x1 = column * width // blocks, (column + 1) * width // blocks
            patch = inputs[:, :, y0:y1, x0:x1]
            result[:, :, y0:y1, x0:x1] = patch * (0.9 + 0.1 * _rand(inputs, generator))
    return result


def block_shuffle(inputs, operation, generator):
    blocks = int(operation.params.get("blocks", 2 if operation.intensity < 0.75 else 4))
    blocks = blocks if blocks in (2, 4, 7, 8, 14, 16) else 2
    height, width = inputs.shape[-2:]
    rows = torch.tensor_split(inputs, blocks, dim=2)
    patches = [patch for row in rows for patch in torch.tensor_split(row, blocks, dim=3)]
    order = torch.randperm(len(patches), generator=generator, device=inputs.device).tolist()
    shuffled_rows = [torch.cat([patches[order[r * blocks + c]] for c in range(blocks)], dim=3) for r in range(blocks)]
    result = torch.cat(shuffled_rows, dim=2)
    return functional.interpolate(result, (height, width), mode="bilinear", align_corners=False)


def block_rotation(inputs, operation, generator):
    blocks = int(operation.params.get("blocks", 2))
    rows = []
    for row in torch.tensor_split(inputs, blocks, dim=2):
        patches = []
        for patch in torch.tensor_split(row, blocks, dim=3):
            k = int(_rand(inputs, generator) * 4)
            rotated = torch.rot90(patch, k, dims=(-2, -1))
            patches.append(functional.interpolate(rotated, patch.shape[-2:], mode="bilinear", align_corners=False))
        rows.append(torch.cat(patches, dim=3))
    return torch.cat(rows, dim=2)


def block_resize(inputs, operation, generator):
    blocks = int(operation.params.get("blocks", 2))
    rows = []
    for row in torch.tensor_split(inputs, blocks, dim=2):
        patches = []
        for patch in torch.tensor_split(row, blocks, dim=3):
            ratio = 1.0 - 0.2 * operation.intensity * _rand(inputs, generator)
            size = (max(2, int(patch.shape[-2] * ratio)), max(2, int(patch.shape[-1] * ratio)))
            resized = functional.interpolate(patch, size, mode="bilinear", align_corners=False)
            patches.append(functional.interpolate(resized, patch.shape[-2:], mode="bilinear", align_corners=False))
        rows.append(torch.cat(patches, dim=3))
    return torch.cat(rows, dim=2)


def frequency_mask(inputs, operation, generator):
    spectrum = torch.fft.rfft2(inputs, norm="ortho")
    height, width = spectrum.shape[-2:]
    yy = torch.linspace(0, 1, height, device=inputs.device)[:, None]
    xx = torch.linspace(0, 1, width, device=inputs.device)[None, :]
    radius = torch.sqrt(yy.square() + xx.square())
    mask = (1.0 - 0.35 * operation.intensity * radius.clamp(max=1)).to(spectrum.dtype)
    return torch.fft.irfft2(spectrum * mask, s=inputs.shape[-2:], norm="ortho").real


def frequency_perturbation(inputs, operation, generator):
    spectrum = torch.fft.rfft2(inputs, norm="ortho")
    phase = (torch.rand(spectrum.shape, device=inputs.device, generator=generator) - 0.5) * (0.15 * operation.intensity)
    rotated = spectrum * torch.exp(1j * phase)
    return torch.fft.irfft2(rotated, s=inputs.shape[-2:], norm="ortho").real


def pixel_noise(inputs, operation, generator):
    noise = torch.randn(inputs.shape, device=inputs.device, generator=generator) * (0.03 * operation.intensity)
    return inputs + noise


def brightness(inputs, operation, generator):
    shift = (2 * _rand(inputs, generator) - 1) * 0.12 * operation.intensity
    return inputs + shift


def contrast(inputs, operation, generator):
    mean = inputs.mean(dim=(-2, -1), keepdim=True)
    factor = 1 + (2 * _rand(inputs, generator) - 1) * 0.25 * operation.intensity
    return mean + factor * (inputs - mean)


def admix_like_mixing(inputs, operation, generator):
    if inputs.shape[0] < 2:
        return inputs
    weight = 0.2 * operation.intensity
    shift = max(1, int(_rand(inputs, generator) * (inputs.shape[0] - 1)))
    return (inputs + weight * inputs.roll(shift, dims=0)) / (1 + weight)


REGISTRY: dict[str, Callable] = {
    name: function for name, function in {
        "identity": identity, "resize_pad": resize_pad, "crop": crop,
        "translation": translation, "scale": scale, "multi_scale": multi_scale,
        "block_partition": block_partition, "block_shuffle": block_shuffle,
        "block_rotation": block_rotation, "block_resize": block_resize,
        "frequency_mask": frequency_mask, "frequency_perturbation": frequency_perturbation,
        "pixel_noise": pixel_noise, "brightness": brightness, "contrast": contrast,
        "admix_like_mixing": admix_like_mixing,
    }.items()
}


def apply_program(inputs: torch.Tensor, program: TransformProgram, generator: torch.Generator) -> torch.Tensor:
    result = inputs
    for operation in program.operations:
        if _rand(result, generator) <= operation.probability:
            result = REGISTRY[operation.name](result, operation, generator)
    return result.clamp(0, 1)


def program_cost(program: TransformProgram) -> float:
    expensive = {"block_shuffle", "block_rotation", "block_resize", "frequency_mask", "frequency_perturbation"}
    return float(sum(1.5 if op.name in expensive else 1.0 for op in program.operations))


def program_features(program: TransformProgram) -> list[float]:
    names = list(REGISTRY)
    histogram = [0.0] * len(names)
    for op in program.operations:
        histogram[names.index(op.name)] += op.intensity * op.probability
    return histogram + [len(program.operations) / 3, program.duration / 10, program_cost(program) / 4.5]

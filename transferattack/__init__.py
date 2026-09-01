"""Independent attack registry for the TransAgent experiments."""

from __future__ import annotations

from importlib import import_module

ATTACK_REGISTRY = {
    "mifgsm": ("transferattack.gradient.mifgsm", "MIFGSM"),
    "transagent": ("transferattack.attacks.transagent", "TransAgent"),
}


def load_attack_class(name: str):
    try:
        module_name, class_name = ATTACK_REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported attack: {name}") from exc
    return getattr(import_module(module_name), class_name)


__all__ = ["ATTACK_REGISTRY", "load_attack_class"]

"""TransferAttack-compatible registry with the TransAgent extension."""

from __future__ import annotations

from importlib import import_module

from third_party.transferattack import attack_zoo as upstream_attack_zoo

ATTACK_REGISTRY = {
    "mifgsm": ("transferattack.gradient.mifgsm", "MIFGSM"),
    "transagent": ("transferattack.attacks.transagent", "TransAgent"),
}
for attack_name, (module_name, class_name) in upstream_attack_zoo.items():
    ATTACK_REGISTRY.setdefault(
        attack_name,
        ("third_party.transferattack" + module_name, class_name),
    )


def load_attack_class(name: str):
    try:
        module_name, class_name = ATTACK_REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported attack: {name}") from exc
    return getattr(import_module(module_name), class_name)


attack_zoo = ATTACK_REGISTRY

__all__ = ["ATTACK_REGISTRY", "attack_zoo", "load_attack_class"]

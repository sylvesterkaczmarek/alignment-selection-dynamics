"""Validate restricted CPU checkpoint contents before constructing live agents."""

from __future__ import annotations
from dataclasses import asdict
import math
import random
import torch
from .evolution import EvolutionConfig
from .integrity import finite, validate_tree
from .seeding import validate_seed

SHAPES = {
    "robust_head.weight": (1, 3),
    "robust_head.bias": (1,),
    "shortcut_head.weight": (1, 1),
    "shortcut_head.bias": (1,),
    "cheap_head.weight": (1, 1),
    "cheap_head.bias": (1,),
}
MODULES = {"", "robust_head", "shortcut_head", "cheap_head"}
FIELDS = {
    "protocol",
    "seed",
    "next_generation",
    "agents",
    "births",
    "parent_state",
    "trait_state",
    "weight_state",
    "config",
    "controls",
}


def _python_state(state):
    if (
        type(state) is not tuple
        or len(state) != 3
        or type(state[0]) is not int
        or state[0] != 3
        or type(state[1]) is not tuple
        or len(state[1]) != 625
        or any(type(x) is not int or not 0 <= x < 2**32 for x in state[1][:-1])
        or type(state[1][-1]) is not int
        or not 0 <= state[1][-1] <= 624
        or (
            state[2] is not None
            and (type(state[2]) not in (int, float) or not math.isfinite(state[2]))
        )
    ):
        raise ValueError("invalid Python generator state")
    random.Random().setstate(state)


def validate_payload(data):
    try:
        _validate(data)
    except (KeyError, TypeError, RuntimeError, IndexError, OverflowError) as exc:
        raise ValueError("invalid checkpoint structure or tensor state") from exc


def _validate(data):
    if type(data) is not dict or data.keys() - {"schema_version"} != FIELDS:
        raise ValueError("unsupported checkpoint fields")
    schema = data.get("schema_version", 1)
    if (
        type(schema) is not int
        or schema not in (1, 2)
        or data["protocol"] != "paired-v1"
    ):
        raise ValueError("unsupported checkpoint schema or protocol")
    validate_seed(data["seed"])
    config = data["config"]
    if type(config) is not dict or config.keys() != asdict(EvolutionConfig()).keys():
        raise ValueError("checkpoint configuration must be complete")
    validate_tree(config)
    cfg = EvolutionConfig(**config)
    if (
        type(data["next_generation"]) is not int
        or not 0 <= data["next_generation"] <= cfg.generations
    ):
        raise ValueError("checkpoint boundary is outside its configured horizon")
    controls = data["controls"]
    if (
        type(controls) is not dict
        or controls.keys() != {"selection", "mutate_traits", "freeze"}
        or controls["selection"] not in ("fitness", "random")
        or type(controls["selection"]) is not str
        or type(controls["mutate_traits"]) is not bool
        or controls["freeze"] not in ("none", "robust", "all")
        or type(controls["freeze"]) is not str
    ):
        raise ValueError("invalid checkpoint controls")
    for name in ("parent_state", "trait_state"):
        _python_state(data[name])
    weights_rng = data["weight_state"]
    if (
        not isinstance(weights_rng, torch.Tensor)
        or weights_rng.device.type != "cpu"
        or weights_rng.layout != torch.strided
        or weights_rng.dtype != torch.uint8
        or weights_rng.ndim != 1
    ):
        raise ValueError("invalid weight generator state")
    torch.Generator().set_state(weights_rng)
    if type(data["agents"]) is not list or len(data["agents"]) != cfg.population_size:
        raise ValueError("checkpoint population size mismatch")
    for item in data["agents"]:
        fields = {"weights", "traits", "training", "requires_grad"} | (
            {"modes"} if schema == 2 else set()
        )
        if type(item) is not dict or item.keys() != fields:
            raise ValueError("invalid saved agent fields")
        if type(item["training"]) is not bool:
            raise ValueError("invalid saved training mode")
        if schema == 2:
            modes = item["modes"]
            if (
                type(modes) is not dict
                or modes.keys() != MODULES
                or any(type(v) is not bool for v in modes.values())
                or modes[""] is not item["training"]
            ):
                raise ValueError("invalid per-module modes")
        if type(item["traits"]) is not dict or item["traits"].keys() != {
            "verifier_logit",
            "bypass_logit",
        }:
            raise ValueError("invalid trait fields")
        for value in item["traits"].values():
            finite(value, "trait logit")
        flags = item["requires_grad"]
        if (
            type(flags) is not dict
            or flags.keys() != SHAPES.keys()
            or any(type(v) is not bool for v in flags.values())
        ):
            raise ValueError("invalid parameter trainability")
        weights = item["weights"]
        if not isinstance(weights, dict) or weights.keys() != SHAPES.keys():
            raise ValueError("invalid model parameter names")
        dtype = None
        for name, tensor in weights.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.layout != torch.strided
                or tuple(tensor.shape) != SHAPES[name]
                or tensor.dtype not in (torch.float32, torch.float64)
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError("invalid checkpoint model tensor")
            if dtype is not None and tensor.dtype != dtype:
                raise ValueError("mixed parameter precision is not supported")
            dtype = tensor.dtype
    births = data["births"]
    generation = data["next_generation"]
    seed = data["seed"]
    if type(births) is not list or len(births) != cfg.population_size:
        raise ValueError("invalid ancestry size")
    validate_tree(births)
    for slot, birth in enumerate(births):
        if (
            type(birth) is not dict
            or birth.keys() != {"id", "parent_id", "mutation"}
            or birth["id"] != f"{seed}:{generation}:{slot}"
        ):
            raise ValueError("invalid ordered ancestry identity")
        if generation == 0:
            if birth["parent_id"] is not None or birth["mutation"] is not None:
                raise ValueError("initial population cannot have recorded parents")
            continue
        if birth["parent_id"] not in {
            f"{seed}:{generation - 1}:{i}" for i in range(cfg.population_size)
        }:
            raise ValueError("invalid parent identity")
        mutation = birth["mutation"]
        if type(mutation) is not dict or mutation.keys() != {
            "verifier_logit",
            "bypass_logit",
            "weights",
        }:
            raise ValueError("invalid mutation record")
        for name in ("verifier_logit", "bypass_logit"):
            finite(mutation[name], name)
        expected = set() if slot < cfg.elites else set(SHAPES)
        if (
            type(mutation["weights"]) is not dict
            or set(mutation["weights"]) != expected
        ):
            raise ValueError("incomplete recorded weight mutation")
        if slot < cfg.elites and (
            mutation["verifier_logit"] != 0 or mutation["bypass_logit"] != 0
        ):
            raise ValueError("elite copy cannot carry trait mutation")
        for name, value in mutation["weights"].items():
            tensor = torch.tensor(value, dtype=torch.float64)
            if tuple(tensor.shape) != SHAPES[name] or not bool(
                torch.isfinite(tensor).all()
            ):
                raise ValueError("invalid recorded weight delta")

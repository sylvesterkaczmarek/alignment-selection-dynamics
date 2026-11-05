"""Typed, finite evidence contracts; hashes establish consistency, not authenticity."""

from __future__ import annotations
import json
import math
from numbers import Real


def finite(value, name="value", *, probability=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number, excluding booleans")
    try:
        valid = math.isfinite(value)
    except (OverflowError, TypeError):
        valid = False
    if not valid or (probability and not 0 <= value <= 1):
        raise ValueError(f"{name} is nonfinite or outside its permitted range")
    return float(value)


def validate_tree(value, *, max_nodes=1000000, max_depth=64):
    stack = [(value, 0)]
    count = 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if depth > max_depth or count > max_nodes:
            raise ValueError("evidence exceeds structure limits")
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("evidence keys must be strings")
                key.encode("utf-8")
                stack.append((child, depth + 1))
        elif type(item) is list:
            stack.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            item.encode("utf-8")
        elif type(item) is float:
            finite(item)
        elif item is not None and type(item) not in (int, bool):
            raise ValueError("evidence must contain ordinary JSON values")


def canonical(value):
    validate_tree(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def strict_json(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate JSON field")
            out[key] = value
        return out

    def reject(value):
        raise ValueError("nonfinite JSON constant")

    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject)
        validate_tree(value)
        return value
    except (RecursionError, UnicodeError) as exc:
        raise ValueError("invalid or excessive JSON evidence") from exc

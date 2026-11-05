from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from .integrity import validate_tree


def write_bytes(path, raw, *, exclusive=False):
    out = Path(path)
    if out.is_symlink():
        raise ValueError("artifact destination must not be a symlink")
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=out.parent, prefix="." + out.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, out)
        else:
            os.replace(temporary, out)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return out


def write_json(path: str | Path, payload: Any, *, exclusive=False) -> Path:
    validate_tree(payload)
    raw = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return write_bytes(path, raw, exclusive=exclusive)


def refuse_existing(paths):
    for path in map(Path, paths):
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite existing experiment evidence: {path}"
            )

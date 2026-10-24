"""Load the project configuration and resolve paths relative to the repo root."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the YAML config and turn every entry under ``paths`` into an absolute Path."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    import os
    if os.environ.get("ISAGIPH_MANUAL"):                 # optional override for the source-data folder
        cfg["paths"]["manual"] = os.environ["ISAGIPH_MANUAL"]
    # relative paths are resolved from the project folder; absolute paths (e.g. C:/Users/.../manual) are kept
    cfg["paths"] = {k: (ROOT / Path(v).expanduser()) for k, v in cfg["paths"].items()}
    for p in cfg["paths"].values():
        p.mkdir(parents=True, exist_ok=True)
    return cfg

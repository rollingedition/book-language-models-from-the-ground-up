"""Read a recording's JSON file: examples/data/recorded/{recording}/{name}.json."""
from __future__ import annotations

import json
from pathlib import Path

RECORDED = Path(__file__).resolve().parent.parent / "data" / "recorded"


def read(recording: str, name: str):
    return json.loads((RECORDED / recording / f"{name}.json").read_text(encoding="utf-8"))

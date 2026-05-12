"""Path helpers."""

from __future__ import annotations

import re
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value.strip("-")


def default_flutter_output_dir(pet_id: str) -> Path:
    return project_root() / "output" / "flutter-assets" / pet_id

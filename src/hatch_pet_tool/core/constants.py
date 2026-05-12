"""Shared sprite atlas constants."""

from __future__ import annotations

from dataclasses import dataclass

COLUMNS = 8
ROWS = 9
CELL_WIDTH = 192
CELL_HEIGHT = 208
ATLAS_WIDTH = COLUMNS * CELL_WIDTH
ATLAS_HEIGHT = ROWS * CELL_HEIGHT


@dataclass(frozen=True)
class AnimationRow:
    state: str
    row: int
    frames: int
    fps: int


ANIMATION_ROWS: tuple[AnimationRow, ...] = (
    AnimationRow("idle", 0, 6, 6),
    AnimationRow("running-right", 1, 8, 8),
    AnimationRow("running-left", 2, 8, 8),
    AnimationRow("waving", 3, 4, 7),
    AnimationRow("jumping", 4, 5, 7),
    AnimationRow("failed", 5, 8, 4),
    AnimationRow("waiting", 6, 6, 5),
    AnimationRow("running", 7, 6, 8),
    AnimationRow("review", 8, 6, 6),
)

FLUTTER_ACTIONS = {
    "idle": "idle",
    "running-right": "runningRight",
    "running-left": "runningLeft",
    "waving": "pet",
    "review": "feed",
    "failed": "sleep",
    "jumping": "jumping",
    "waiting": "waiting",
    "running": "running",
}

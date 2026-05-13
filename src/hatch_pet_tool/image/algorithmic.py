"""Deterministic placeholder frame generation from one source image."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from hatch_pet_tool.core.constants import ANIMATION_ROWS, CELL_HEIGHT, CELL_WIDTH

def source_to_sprite(path: Path) -> Image.Image:
    """Load one image and normalize it into a transparent 192x208 sprite frame."""

    with Image.open(path) as opened:
        image = opened.convert("RGBA")

    bbox = image.getbbox()
    if bbox is None:
        raise SystemExit(f"input image has no non-background pixels: {path}")

    sprite = image.crop(bbox)
    sprite.thumbnail((CELL_WIDTH - 20, CELL_HEIGHT - 20), Image.Resampling.NEAREST)
    frame = Image.new("RGBA", (CELL_WIDTH, CELL_HEIGHT), (0, 0, 0, 0))
    left = (CELL_WIDTH - sprite.width) // 2
    top = (CELL_HEIGHT - sprite.height) // 2
    frame.alpha_composite(sprite, (left, top))
    return frame


def transform_frame(base: Image.Image, state: str, index: int, count: int) -> Image.Image:
    """Create one simple deterministic animation frame."""

    centered = Image.new("RGBA", (CELL_WIDTH, CELL_HEIGHT), (0, 0, 0, 0))
    if count <= 1:
        phase = 0.0
    else:
        phase = index / (count - 1)

    if state == "running-left":
        source = base.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    else:
        source = base.copy()

    offset_x = 0
    offset_y = 0
    scale_x = 1.0
    scale_y = 1.0

    if state == "idle":
        offset_y = [0, -2, -3, -2, 0, 1][index]
    elif state == "running-right":
        offset_x = [-7, -4, -1, 2, 5, 7, 4, 0][index]
        offset_y = [0, -3, 0, 2, 0, -3, 0, 2][index]
    elif state == "running-left":
        offset_x = [7, 4, 1, -2, -5, -7, -4, 0][index]
        offset_y = [0, -3, 0, 2, 0, -3, 0, 2][index]
    elif state == "waving":
        offset_x = [0, 3, 5, 2][index]
        offset_y = [0, -4, -5, -2][index]
    elif state == "jumping":
        offset_y = [5, -8, -22, -8, 2][index]
        if index == 0:
            scale_x, scale_y = 1.06, 0.94
    elif state == "failed":
        offset_y = [0, 3, 6, 8, 8, 7, 6, 6][index]
        if index >= 3:
            scale_x, scale_y = 1.08, 0.88
    elif state == "waiting":
        offset_x = [0, 1, 2, 1, 0, -1][index]
        offset_y = [0, -1, 0, 1, 0, -1][index]
    elif state == "running":
        offset_y = [0, -4, 0, 3, 0, -3][index]
        offset_x = [-2, 0, 2, 0, -2, 0][index]
    elif state == "review":
        offset_x = [0, -1, -2, -2, -1, 0][index]
        offset_y = [0, 0, -1, -1, 0, 0][index]

    bbox = source.getbbox()
    if bbox is None:
        return centered
    cropped = source.crop(bbox)
    width = max(1, round(cropped.width * scale_x))
    height = max(1, round(cropped.height * scale_y))
    if (width, height) != cropped.size:
        cropped = cropped.resize((width, height), Image.Resampling.NEAREST)

    left = (CELL_WIDTH - cropped.width) // 2 + offset_x
    top = (CELL_HEIGHT - cropped.height) // 2 + offset_y
    centered.alpha_composite(cropped, (left, top))
    return centered


def generate_algorithmic_frames(image_path: Path, frames_root: Path) -> dict[str, object]:
    """Generate all hatch-pet animation rows under frames_root."""

    base = source_to_sprite(image_path)
    rows = []
    for animation in ANIMATION_ROWS:
        state_dir = frames_root / animation.state
        state_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for index in range(animation.frames):
            frame = transform_frame(base, animation.state, index, animation.frames)
            frame_path = state_dir / f"frame_{index:02d}.png"
            frame.save(frame_path)
            files.append(str(frame_path))
        rows.append(
            {
                "state": animation.state,
                "row": animation.row,
                "frames": animation.frames,
                "method": "algorithmic-placeholder",
                "files": files,
            }
        )
    return {"ok": True, "source_image": str(image_path), "rows": rows}

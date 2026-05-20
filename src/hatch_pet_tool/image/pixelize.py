"""Pixel/bead subject extraction and cell normalization."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from hatch_pet_tool.core.constants import CELL_HEIGHT, CELL_WIDTH
from hatch_pet_tool.image.color_restore import (
    build_base_pixel_pet,
    palette_preview,
    render_base_to_cell,
    restore_colors,
    save_palette_preview,
)

DEFAULT_COLORS = 16
DEFAULT_PADDING = 10


def _nontransparent_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    bbox = image.getbbox()
    if bbox is None:
        raise SystemExit("pixelize input has no visible subject; check background removal")
    return bbox


def _nearest_palette_color(
    color: tuple[int, int, int],
    palette: list[tuple[int, int, int]],
) -> tuple[int, int, int]:
    return min(
        palette,
        key=lambda item: (
            (color[0] - item[0]) ** 2
            + (color[1] - item[1]) ** 2
            + (color[2] - item[2]) ** 2
        ),
    )


def _flattened_data(image: Image.Image):
    if hasattr(image, "get_flattened_data"):
        return image.get_flattened_data()
    return image.getdata()


def limit_colors(image: Image.Image, colors: int) -> tuple[Image.Image, dict[str, object]]:
    return restore_colors(image, colors)


def normalize_to_cell(
    image: Image.Image,
    *,
    colors: int = DEFAULT_COLORS,
    padding: int = DEFAULT_PADDING,
) -> tuple[Image.Image, dict[str, object]]:
    if padding < 0 or padding * 2 >= min(CELL_WIDTH, CELL_HEIGHT):
        raise SystemExit("pixelize padding is too large for the target cell")

    rgba = image.convert("RGBA")
    bbox = _nontransparent_bbox(rgba)
    subject = rgba.crop(bbox)
    subject, color_info = limit_colors(subject, colors)

    max_width = CELL_WIDTH - padding * 2
    max_height = CELL_HEIGHT - padding * 2
    scale = min(max_width / subject.width, max_height / subject.height)
    resized_size = (
        max(1, round(subject.width * scale)),
        max(1, round(subject.height * scale)),
    )
    if resized_size != subject.size:
        subject = subject.resize(resized_size, Image.Resampling.NEAREST)

    cell = Image.new("RGBA", (CELL_WIDTH, CELL_HEIGHT), (0, 0, 0, 0))
    left = (CELL_WIDTH - subject.width) // 2
    top = (CELL_HEIGHT - subject.height) // 2
    cell.alpha_composite(subject, (left, top))
    return cell, {
        "source_size": [rgba.width, rgba.height],
        "subject_bbox": list(bbox),
        "subject_size": [bbox[2] - bbox[0], bbox[3] - bbox[1]],
        "cell_size": [CELL_WIDTH, CELL_HEIGHT],
        "padding": padding,
        "scale": scale,
        "resized_size": [subject.width, subject.height],
        "offset": [left, top],
        "colors": color_info,
    }


def pixelize_image(
    *,
    image_path: Path,
    output_path: Path,
    colors: int = DEFAULT_COLORS,
    padding: int = DEFAULT_PADDING,
    base_output_path: Path | None = None,
    rendered_output_path: Path | None = None,
    render_style: str = "soft-pixel",
    render_scale: int = 2,
    palette_preview_path: Path | None = None,
) -> dict[str, object]:
    with Image.open(image_path) as opened:
        image = opened.convert("RGBA")
    max_width = CELL_WIDTH - padding * 2
    max_height = CELL_HEIGHT - padding * 2
    base, base_info = build_base_pixel_pet(
        image,
        colors=colors,
        max_width=max_width,
        max_height=max_height,
    )
    cell, render_info = render_base_to_cell(
        base,
        padding=padding,
        render_style=render_style,
        render_scale=render_scale,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cell.save(output_path)
    base_output = None
    if base_output_path is not None:
        base_output_path.parent.mkdir(parents=True, exist_ok=True)
        base.save(base_output_path)
        base_output = str(base_output_path)
    rendered_output = None
    if rendered_output_path is not None:
        rendered_output_path.parent.mkdir(parents=True, exist_ok=True)
        cell.save(rendered_output_path)
        rendered_output = str(rendered_output_path)
    if palette_preview_path is not None:
        preview = save_palette_preview(base_info["colors"], palette_preview_path)
        base_info["colors"]["palette_preview"] = preview
    return {
        "source_image": str(image_path),
        "base_pixel_pet": base_output,
        "rendered_pixel_pet": rendered_output,
        "pixelized_image": str(output_path),
        **base_info,
        **render_info,
    }

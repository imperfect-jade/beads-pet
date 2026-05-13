"""Pixel/bead subject extraction and cell normalization."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image

from hatch_pet_tool.core.constants import CELL_HEIGHT, CELL_WIDTH

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
    if colors <= 0:
        raise SystemExit("--colors must be positive")

    rgba = image.convert("RGBA")
    counter: Counter[tuple[int, int, int]] = Counter()
    for red, green, blue, alpha in _flattened_data(rgba):
        if alpha > 0:
            counter[(red, green, blue)] += 1

    if not counter:
        raise SystemExit("pixelize input has no visible subject; check background removal")

    palette = [color for color, _count in counter.most_common(colors)]
    if len(counter) <= colors:
        return rgba, {
            "requested_colors": colors,
            "source_colors": len(counter),
            "output_colors": len(counter),
            "quantized": False,
        }

    pixels = rgba.load()
    cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for y in range(rgba.height):
        for x in range(rgba.width):
            red, green, blue, alpha = pixels[x, y]
            if alpha == 0:
                continue
            source = (red, green, blue)
            mapped = cache.get(source)
            if mapped is None:
                mapped = _nearest_palette_color(source, palette)
                cache[source] = mapped
            pixels[x, y] = (*mapped, alpha)

    return rgba, {
        "requested_colors": colors,
        "source_colors": len(counter),
        "output_colors": len(palette),
        "quantized": True,
    }


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
) -> dict[str, object]:
    with Image.open(image_path) as opened:
        image = opened.convert("RGBA")
    cell, info = normalize_to_cell(image, colors=colors, padding=padding)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cell.save(output_path)
    return {
        "source_image": str(image_path),
        "pixelized_image": str(output_path),
        **info,
    }

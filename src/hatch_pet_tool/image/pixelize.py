"""Pixel/bead subject extraction and cell normalization."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from hatch_pet_tool.core.constants import CELL_HEIGHT, CELL_WIDTH

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - pyproject installs these for normal use.
    cv2 = None
    np = None

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


def _is_outline_color(color: tuple[int, int, int]) -> bool:
    return max(color) <= 48


def _is_saturated_color(color: tuple[int, int, int]) -> bool:
    red, green, blue = color
    return max(color) - min(color) >= 55 and max(color) >= 95


def _dedupe_palette(palette: list[tuple[int, int, int]], limit: int) -> list[tuple[int, int, int]]:
    output: list[tuple[int, int, int]] = []
    for color in palette:
        normalized = tuple(max(0, min(255, int(round(channel)))) for channel in color)
        if normalized not in output:
            output.append(normalized)
        if len(output) >= limit:
            break
    return output


def _cluster_palette(
    counter: Counter[tuple[int, int, int]],
    colors: int,
) -> tuple[list[tuple[int, int, int]], dict[str, object]] | None:
    if cv2 is None or np is None:
        return None

    total = sum(counter.values())
    outline_candidates = [
        (color, count)
        for color, count in counter.items()
        if _is_outline_color(color)
    ]
    outline_color = None
    if outline_candidates:
        outline_color = max(outline_candidates, key=lambda item: item[1])[0]

    sample_colors: list[tuple[int, int, int]] = []
    for color, count in counter.items():
        if outline_color is not None and _is_outline_color(color):
            continue
        weight = max(1, min(64, round(count / max(1, total) * 4096)))
        if _is_saturated_color(color):
            weight = max(weight, 5)
        rounded = tuple(round(channel / 8) * 8 for channel in color)
        sample_colors.extend([rounded] * weight)

    if not sample_colors:
        return None

    cluster_count = min(colors - (1 if outline_color else 0), len(set(sample_colors)))
    if cluster_count <= 0:
        return None

    samples = np.float32(sample_colors)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _compactness, _labels, centers = cv2.kmeans(
        samples,
        cluster_count,
        None,
        criteria,
        4,
        cv2.KMEANS_PP_CENTERS,
    )
    clustered = [tuple(int(round(channel)) for channel in center) for center in centers]
    clustered.sort(
        key=lambda color: (
            0 if _is_saturated_color(color) else 1,
            -counter.get(color, 0),
            sum(color),
        )
    )
    palette = []
    if outline_color is not None:
        palette.append(outline_color)
    palette.extend(clustered)
    palette = _dedupe_palette(palette, colors)
    return palette, {
        "palette_source": "clustered",
        "outline_color": list(outline_color) if outline_color else None,
        "cluster_count": cluster_count,
        "sample_count": len(sample_colors),
    }


def palette_preview(colors: list[tuple[int, int, int]], *, swatch_size: int = 24) -> Image.Image:
    width = max(1, len(colors)) * swatch_size
    image = Image.new("RGBA", (width, swatch_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for index, color in enumerate(colors):
        left = index * swatch_size
        draw.rectangle((left, 0, left + swatch_size - 1, swatch_size - 1), fill=(*color, 255))
    return image


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

    clustered = _cluster_palette(counter, colors)
    if clustered is None:
        palette = [color for color, _count in counter.most_common(colors)]
        palette_info: dict[str, object] = {"palette_source": "exact"}
    else:
        palette, palette_info = clustered
    if len(counter) <= colors:
        return rgba, {
            "requested_colors": colors,
            "source_colors": len(counter),
            "output_colors": len(counter),
            "quantized": False,
            "palette_source": "exact",
            "palette": [list(color) for color in counter],
            "palette_counts": [
                {"color": list(color), "pixels": count, "ratio": count / sum(counter.values())}
                for color, count in counter.most_common()
            ],
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
        **palette_info,
        "palette": [list(color) for color in palette],
        "palette_counts": [
            {
                "color": list(color),
                "pixels": counter[color],
                "ratio": round(counter[color] / sum(counter.values()), 6),
            }
            for color in palette
        ],
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
    palette_preview_path: Path | None = None,
) -> dict[str, object]:
    with Image.open(image_path) as opened:
        image = opened.convert("RGBA")
    cell, info = normalize_to_cell(image, colors=colors, padding=padding)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cell.save(output_path)
    if palette_preview_path is not None:
        raw_palette = info.get("colors", {}).get("palette", [])
        palette = [tuple(int(channel) for channel in color) for color in raw_palette]
        palette_preview_path.parent.mkdir(parents=True, exist_ok=True)
        palette_preview(palette).save(palette_preview_path)
        info["colors"]["palette_preview"] = str(palette_preview_path)
    return {
        "source_image": str(image_path),
        "pixelized_image": str(output_path),
        **info,
    }

"""Color restoration for bead and pixel pet references."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from hatch_pet_tool.core.constants import CELL_HEIGHT, CELL_WIDTH

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - pyproject installs these for normal use.
    cv2 = None
    np = None


def _flattened_data(image: Image.Image):
    if hasattr(image, "get_flattened_data"):
        return image.get_flattened_data()
    return image.getdata()


def visible_color_counter(image: Image.Image) -> Counter[tuple[int, int, int]]:
    rgba = image.convert("RGBA")
    counter: Counter[tuple[int, int, int]] = Counter()
    for red, green, blue, alpha in _flattened_data(rgba):
        if alpha > 0:
            counter[(red, green, blue)] += 1
    return counter


def is_outline_color(color: tuple[int, int, int]) -> bool:
    return max(color) <= 52


def is_saturated_color(color: tuple[int, int, int]) -> bool:
    return max(color) - min(color) >= 55 and max(color) >= 95


def palette_preview(colors: list[tuple[int, int, int]], *, swatch_size: int = 24) -> Image.Image:
    width = max(1, len(colors)) * swatch_size
    image = Image.new("RGBA", (width, swatch_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for index, color in enumerate(colors):
        left = index * swatch_size
        draw.rectangle((left, 0, left + swatch_size - 1, swatch_size - 1), fill=(*color, 255))
    return image


def _rgb_to_lab(colors: list[tuple[int, int, int]]):
    if cv2 is None or np is None:
        return None
    rgb = np.array(colors, dtype=np.uint8).reshape((len(colors), 1, 3))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape((len(colors), 3)).astype("float32")


def _nearest_palette_color(
    color: tuple[int, int, int],
    palette: list[tuple[int, int, int]],
    palette_lab=None,
) -> tuple[int, int, int]:
    if palette_lab is not None and cv2 is not None and np is not None:
        source_lab = _rgb_to_lab([color])
        if source_lab is not None:
            distances = np.linalg.norm(palette_lab - source_lab[0], axis=1)
            return palette[int(distances.argmin())]
    return min(
        palette,
        key=lambda item: (
            (color[0] - item[0]) ** 2
            + (color[1] - item[1]) ** 2
            + (color[2] - item[2]) ** 2
        ),
    )


def _merge_near_palette_colors(palette: list[tuple[int, int, int]], limit: int) -> list[tuple[int, int, int]]:
    merged: list[tuple[int, int, int]] = []
    for color in palette:
        normalized = tuple(max(0, min(255, int(round(channel)))) for channel in color)
        if normalized in merged:
            continue
        lab_pair = _rgb_to_lab([normalized] + merged) if merged else None
        too_close = False
        for index, existing in enumerate(merged):
            if is_outline_color(normalized) != is_outline_color(existing):
                continue
            if is_saturated_color(normalized) != is_saturated_color(existing):
                continue
            if lab_pair is not None:
                distance = float(np.linalg.norm(lab_pair[0] - lab_pair[index + 1]))
                too_close = distance < 8.0
            else:
                distance = sum((normalized[channel] - existing[channel]) ** 2 for channel in range(3)) ** 0.5
                too_close = distance < 18.0
            if too_close:
                break
        if not too_close:
            merged.append(normalized)
        if len(merged) >= limit:
            break
    return merged


def _cluster_palette(
    counter: Counter[tuple[int, int, int]],
    colors: int,
) -> tuple[list[tuple[int, int, int]], dict[str, object]] | None:
    if cv2 is None or np is None:
        return None

    total = sum(counter.values())
    outline_candidates = [(color, count) for color, count in counter.items() if is_outline_color(color)]
    outline_color = max(outline_candidates, key=lambda item: item[1])[0] if outline_candidates else None

    unique_colors = list(counter.keys())
    lab = _rgb_to_lab(unique_colors)
    if lab is None:
        return None

    samples: list[list[float]] = []
    for index, color in enumerate(unique_colors):
        if outline_color is not None and is_outline_color(color):
            continue
        count = counter[color]
        weight = max(1, min(80, round(count / max(1, total) * 4096)))
        if is_saturated_color(color):
            weight = max(weight, 8)
        if count / max(1, total) >= 0.06:
            weight = max(weight, 16)
        rounded_lab = np.round(lab[index] / 3.0) * 3.0
        samples.extend([rounded_lab.tolist()] * weight)

    if not samples:
        return None

    reserved = 1 if outline_color is not None else 0
    cluster_count = min(max(1, colors - reserved), len({tuple(sample) for sample in samples}))
    if cluster_count <= 0:
        return None

    sample_array = np.float32(samples)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.35)
    _compactness, _labels, centers = cv2.kmeans(
        sample_array,
        cluster_count,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    centers_rgb = cv2.cvtColor(
        np.uint8(np.clip(centers, 0, 255)).reshape((len(centers), 1, 3)),
        cv2.COLOR_LAB2RGB,
    ).reshape((len(centers), 3))
    clustered = [tuple(int(channel) for channel in center) for center in centers_rgb]

    protected_saturated = [
        color
        for color, _count in sorted(counter.items(), key=lambda item: item[1], reverse=True)
        if is_saturated_color(color)
    ][: max(0, min(4, colors // 3))]

    palette: list[tuple[int, int, int]] = []
    if outline_color is not None:
        palette.append(outline_color)
    palette.extend(protected_saturated)
    palette.extend(
        sorted(
            clustered,
            key=lambda color: (
                0 if is_saturated_color(color) else 1,
                min(
                    (color[0] - source[0]) ** 2 + (color[1] - source[1]) ** 2 + (color[2] - source[2]) ** 2
                    for source in counter
                ),
                sum(color),
            ),
        )
    )
    palette = _merge_near_palette_colors(palette, colors)
    return palette, {
        "palette_source": "perceptual-clustered",
        "outline_color": list(outline_color) if outline_color else None,
        "cluster_count": cluster_count,
        "sample_count": len(samples),
    }


def restore_colors(image: Image.Image, colors: int) -> tuple[Image.Image, dict[str, object]]:
    if colors <= 0:
        raise SystemExit("--colors must be positive")

    rgba = image.convert("RGBA")
    counter = visible_color_counter(rgba)
    if not counter:
        raise SystemExit("reference input has no visible subject; check background removal")

    if len(counter) <= colors:
        palette = list(counter)
        restored = rgba
        palette_info: dict[str, object] = {"palette_source": "exact", "quantized": False}
    else:
        clustered = _cluster_palette(counter, colors)
        if clustered is None:
            palette = [color for color, _count in counter.most_common(colors)]
            palette_info = {"palette_source": "exact", "quantized": True}
        else:
            palette, palette_info = clustered
            palette_info["quantized"] = True

        palette_lab = _rgb_to_lab(palette)
        restored = rgba.copy()
        pixels = restored.load()
        cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        for y in range(restored.height):
            for x in range(restored.width):
                red, green, blue, alpha = pixels[x, y]
                if alpha == 0:
                    continue
                source = (red, green, blue)
                mapped = cache.get(source)
                if mapped is None:
                    mapped = _nearest_palette_color(source, palette, palette_lab)
                    cache[source] = mapped
                pixels[x, y] = (*mapped, alpha)

    mapped_counts = visible_color_counter(restored)
    palette = [color for color, _count in mapped_counts.most_common()]
    return restored, {
        "requested_colors": colors,
        "source_colors": len(counter),
        "output_colors": len(mapped_counts),
        **palette_info,
        "palette": [list(color) for color in palette],
        "palette_counts": [
            {
                "color": list(color),
                "pixels": count,
                "ratio": round(count / sum(mapped_counts.values()), 6),
            }
            for color, count in mapped_counts.most_common()
        ],
    }


def restore_template_grid_colors(image: Image.Image, colors: int) -> tuple[Image.Image, dict[str, object]]:
    """Merge JPEG/sample noise while preserving template grid cell colors.

    Template images often contain text labels inside otherwise finished color
    cells. This path keeps color fidelity ahead of strict palette reduction:
    near-identical colors are collapsed, but small semantic colors are not
    smoothed away to satisfy the default color count.
    """

    if colors <= 0:
        raise SystemExit("--colors must be positive")

    rgba = image.convert("RGBA")
    counter = visible_color_counter(rgba)
    if not counter:
        raise SystemExit("reference input has no visible subject; check grid sampling")

    palette_limit = max(colors, 24)
    total = sum(counter.values())
    min_semantic_count = max(3, round(total * 0.004))
    if len(counter) <= palette_limit:
        restored = rgba
        palette_info: dict[str, object] = {
            "palette_source": "grid-template-exact",
            "quantized": False,
            "palette_limit": palette_limit,
        }
    else:
        outline_candidates = [(color, count) for color, count in counter.items() if is_outline_color(color)]
        outline = [max(outline_candidates, key=lambda item: item[1])[0]] if outline_candidates else []
        saturated = [
            color
            for color, _count in sorted(counter.items(), key=lambda item: item[1], reverse=True)
            if is_saturated_color(color) and _count >= min_semantic_count
        ][: max(4, colors // 2)]
        frequent = [color for color, count in counter.most_common() if count >= min_semantic_count]
        ordered = outline + saturated + frequent
        if not ordered:
            ordered = [color for color, _count in counter.most_common(min(colors, len(counter)))]
        palette = _merge_near_palette_colors(ordered, palette_limit)
        palette_lab = _rgb_to_lab(palette)

        restored = rgba.copy()
        pixels = restored.load()
        cache: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        for y in range(restored.height):
            for x in range(restored.width):
                red, green, blue, alpha = pixels[x, y]
                if alpha == 0:
                    continue
                source = (red, green, blue)
                mapped = cache.get(source)
                if mapped is None:
                    mapped = _nearest_palette_color(source, palette, palette_lab)
                    cache[source] = mapped
                pixels[x, y] = (*mapped, alpha)
        palette_info = {
            "palette_source": "grid-template-merged",
            "quantized": True,
            "palette_limit": palette_limit,
            "min_semantic_count": min_semantic_count,
        }

    mapped_counts = visible_color_counter(restored)
    palette = [color for color, _count in mapped_counts.most_common()]
    return restored, {
        "requested_colors": colors,
        "source_colors": len(counter),
        "output_colors": len(mapped_counts),
        **palette_info,
        "palette": [list(color) for color in palette],
        "palette_counts": [
            {
                "color": list(color),
                "pixels": count,
                "ratio": round(count / sum(mapped_counts.values()), 6),
            }
            for color, count in mapped_counts.most_common()
        ],
    }


def smooth_similar_neighbors(image: Image.Image, *, max_distance: float = 10.0) -> tuple[Image.Image, dict[str, object]]:
    rgba = image.convert("RGBA")
    if rgba.width < 3 or rgba.height < 3:
        return rgba, {"changed_pixels": 0, "method": "neighbor-majority"}

    pixels = rgba.load()
    output = rgba.copy()
    out_pixels = output.load()
    changed = 0
    for y in range(1, rgba.height - 1):
        for x in range(1, rgba.width - 1):
            current = pixels[x, y]
            if current[3] == 0 or is_outline_color(current[:3]) or is_saturated_color(current[:3]):
                continue
            neighbors = [
                pixels[x - 1, y],
                pixels[x + 1, y],
                pixels[x, y - 1],
                pixels[x, y + 1],
            ]
            visible = [neighbor[:3] for neighbor in neighbors if neighbor[3] > 0]
            if len(visible) < 3:
                continue
            dominant, count = Counter(visible).most_common(1)[0]
            if count < 3 or is_outline_color(dominant) or is_saturated_color(dominant):
                continue
            source_lab = _rgb_to_lab([current[:3], dominant])
            if source_lab is not None:
                distance = float(np.linalg.norm(source_lab[0] - source_lab[1]))
            else:
                distance = sum((current[channel] - dominant[channel]) ** 2 for channel in range(3)) ** 0.5
            if distance <= max_distance:
                out_pixels[x, y] = (*dominant, current[3])
                changed += 1
    return output, {"changed_pixels": changed, "method": "neighbor-majority"}


def fit_base_to_cell(
    base: Image.Image,
    *,
    padding: int,
) -> tuple[Image.Image, dict[str, object]]:
    if padding < 0 or padding * 2 >= min(CELL_WIDTH, CELL_HEIGHT):
        raise SystemExit("pixelize padding is too large for the target cell")

    rgba = base.convert("RGBA")
    bbox = rgba.getbbox()
    if bbox is None:
        raise SystemExit("base pixel pet has no visible subject")
    subject = rgba.crop(bbox)
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
        "base_size": [rgba.width, rgba.height],
        "subject_bbox": list(bbox),
        "subject_size": [bbox[2] - bbox[0], bbox[3] - bbox[1]],
        "cell_size": [CELL_WIDTH, CELL_HEIGHT],
        "padding": padding,
        "scale": scale,
        "resized_size": [subject.width, subject.height],
        "offset": [left, top],
    }


def render_base_to_cell(
    base: Image.Image,
    *,
    padding: int,
    render_style: str = "soft-pixel",
    render_scale: int = 2,
) -> tuple[Image.Image, dict[str, object]]:
    style = render_style.lower()
    if style not in {"soft-pixel", "pixel"}:
        raise SystemExit("--render-style must be soft-pixel or pixel")
    if render_scale <= 0:
        raise SystemExit("--render-scale must be positive")
    if style == "pixel":
        cell, info = fit_base_to_cell(base, padding=padding)
        info.update(
            {
                "render_style": "pixel",
                "render_scale": render_scale,
                "render_resampling": "nearest",
            }
        )
        return cell, info

    rgba = base.convert("RGBA")
    bbox = rgba.getbbox()
    if bbox is None:
        raise SystemExit("base pixel pet has no visible subject")
    subject = rgba.crop(bbox)
    padded_subject = Image.new("RGBA", (subject.width + 2, subject.height + 2), (0, 0, 0, 0))
    padded_subject.alpha_composite(subject, (1, 1))
    max_width = CELL_WIDTH - padding * 2
    max_height = CELL_HEIGHT - padding * 2
    scale = min(max_width / padded_subject.width, max_height / padded_subject.height)
    resized_size = (
        max(1, round(subject.width * scale)),
        max(1, round(subject.height * scale)),
    )
    internal_size = (
        max(1, round(padded_subject.width * scale) * render_scale),
        max(1, round(padded_subject.height * scale) * render_scale),
    )
    enlarged = padded_subject.resize(internal_size, Image.Resampling.NEAREST)
    smoothed = enlarged.resize(
        (
            max(1, round(padded_subject.width * scale)),
            max(1, round(padded_subject.height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )
    alpha = smoothed.getchannel("A").filter(ImageFilter.GaussianBlur(radius=0.35))
    alpha = alpha.point(lambda value: 0 if value < 3 else value)
    smoothed.putalpha(alpha)

    cell = Image.new("RGBA", (CELL_WIDTH, CELL_HEIGHT), (0, 0, 0, 0))
    left = (CELL_WIDTH - smoothed.width) // 2
    top = (CELL_HEIGHT - smoothed.height) // 2
    cell.alpha_composite(smoothed, (left, top))
    return cell, {
        "base_size": [rgba.width, rgba.height],
        "subject_bbox": list(bbox),
        "subject_size": [bbox[2] - bbox[0], bbox[3] - bbox[1]],
        "cell_size": [CELL_WIDTH, CELL_HEIGHT],
        "padding": padding,
        "scale": scale,
        "resized_size": [smoothed.width, smoothed.height],
        "offset": [left, top],
        "render_style": "soft-pixel",
        "render_scale": render_scale,
        "render_resampling": "nearest-lanczos",
    }


def build_base_pixel_pet(
    image: Image.Image,
    *,
    colors: int,
    max_width: int,
    max_height: int,
) -> tuple[Image.Image, dict[str, object]]:
    rgba = image.convert("RGBA")
    bbox = rgba.getbbox()
    if bbox is None:
        raise SystemExit("reference input has no visible subject; check background removal")
    subject = rgba.crop(bbox)
    scale = min(1.0, max_width / subject.width, max_height / subject.height)
    resized_size = (
        max(1, round(subject.width * scale)),
        max(1, round(subject.height * scale)),
    )
    if resized_size != subject.size:
        subject = subject.resize(resized_size, Image.Resampling.BOX)
    restored, color_info = restore_colors(subject, colors)
    smoothed, smooth_info = smooth_similar_neighbors(restored)
    return smoothed, {
        "source_size": [rgba.width, rgba.height],
        "subject_bbox": list(bbox),
        "subject_size": [bbox[2] - bbox[0], bbox[3] - bbox[1]],
        "base_size": [smoothed.width, smoothed.height],
        "base_scale": scale,
        "colors": color_info,
        "smoothing": smooth_info,
    }


def save_palette_preview(info: dict[str, object], path: Path | None) -> str | None:
    if path is None:
        return None
    raw_palette = info.get("palette", [])
    palette = [tuple(int(channel) for channel in color) for color in raw_palette]
    path.parent.mkdir(parents=True, exist_ok=True)
    palette_preview(palette).save(path)
    return str(path)

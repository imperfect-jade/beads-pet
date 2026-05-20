"""Lightweight bead-grid detection and sampling.

The first grid pass is intentionally conservative. It looks for regular color
changes in a preprocessed subject image, samples bead centers when confidence is
high, and lets callers fall back to pixelize mode when the image is not a clear
front-facing bead grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from PIL import Image, ImageDraw

from hatch_pet_tool.image.pixelize import DEFAULT_COLORS, DEFAULT_PADDING, normalize_to_cell

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - pyproject installs these for normal use.
    cv2 = None
    np = None

DEFAULT_GRID_MIN_CONFIDENCE = 0.48
GRID_PREVIEW_CELL = 8
MIN_GRID_DIVISIONS = 4
MAX_GRID_DIVISIONS = 96
MIN_CELL_SIZE = 4.0
PEAK_FLOOR = 0.055


class GridLowConfidenceError(Exception):
    """Raised when an image does not look like a regular bead grid."""

    def __init__(self, *, confidence: float, details: dict[str, Any] | None = None) -> None:
        self.confidence = confidence
        self.details = details or {}
        super().__init__(
            "bead grid confidence is too low; use --reference-mode pixelize, "
            "manual --crop, or a cleaner front-facing bead image"
        )


@dataclass(frozen=True)
class AxisFit:
    divisions: int
    cell_size: float
    offset: float
    score: float
    threshold: float
    boundary_strength: float
    hit_ratio: float


@dataclass(frozen=True)
class GridFit:
    image: Image.Image
    bbox: tuple[int, int, int, int]
    rotation: int
    columns: int
    rows: int
    cell_width: float
    cell_height: float
    confidence: float
    x_axis: AxisFit
    y_axis: AxisFit
    method: str
    reject_reason: str | None = None


def _flattened_data(image: Image.Image):
    if hasattr(image, "get_flattened_data"):
        return image.get_flattened_data()
    return image.getdata()


def _rotation_variants(image: Image.Image) -> list[tuple[int, Image.Image]]:
    return [
        (0, image),
        (90, image.transpose(Image.Transpose.ROTATE_90)),
        (180, image.transpose(Image.Transpose.ROTATE_180)),
        (270, image.transpose(Image.Transpose.ROTATE_270)),
    ]


def _color_distance(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    if left[3] == 0 and right[3] == 0:
        return 0.0
    if left[3] == 0 or right[3] == 0:
        return 0.25
    return sqrt(
        (left[0] - right[0]) ** 2
        + (left[1] - right[1]) ** 2
        + (left[2] - right[2]) ** 2
    ) / 441.673


def _axis_profile(image: Image.Image, *, axis: str) -> list[float]:
    pixels = image.load()
    width, height = image.size
    if axis == "x":
        length = width
        other = height
    else:
        length = height
        other = width

    step = max(1, other // 96)
    profile: list[float] = []
    for boundary in range(1, length):
        values: list[float] = []
        for cursor in range(0, other, step):
            if axis == "x":
                left = pixels[boundary - 1, cursor]
                right = pixels[boundary, cursor]
            else:
                left = pixels[cursor, boundary - 1]
                right = pixels[cursor, boundary]
            values.append(_color_distance(left, right))
        profile.append(sum(values) / len(values) if values else 0.0)

    if len(profile) < 3:
        return profile
    smoothed: list[float] = []
    for index in range(len(profile)):
        window = profile[max(0, index - 1) : min(len(profile), index + 2)]
        smoothed.append(sum(window) / len(window))
    return smoothed


def _boundary_value(profile: list[float], position: float) -> float:
    if not profile:
        return 0.0
    center = round(position) - 1
    left = max(0, center - 1)
    right = min(len(profile), center + 2)
    return max(profile[left:right] or [0.0])


def _fit_axis(profile: list[float], length: int) -> AxisFit:
    if length < MIN_GRID_DIVISIONS * MIN_CELL_SIZE or not profile:
        return AxisFit(0, 0.0, 0.0, 0.0, PEAK_FLOOR, 0.0, 0.0)

    profile_mean = mean(profile)
    profile_std = pstdev(profile) if len(profile) > 1 else 0.0
    threshold = max(PEAK_FLOOR, profile_mean + profile_std * 0.75)
    max_divisions = min(MAX_GRID_DIVISIONS, int(length // MIN_CELL_SIZE))
    best = AxisFit(0, 0.0, 0.0, 0.0, threshold, 0.0, 0.0)

    for divisions in range(MIN_GRID_DIVISIONS, max_divisions + 1):
        cell_size = length / divisions
        if cell_size < MIN_CELL_SIZE:
            continue
        boundaries = [index * cell_size for index in range(1, divisions)]
        if len(boundaries) < 3:
            continue
        values = [_boundary_value(profile, boundary) for boundary in boundaries]
        hits = [value for value in values if value >= threshold]
        hit_ratio = len(hits) / len(values)
        boundary_strength = (sum(values) / len(values)) / threshold if threshold else 0.0
        strength_score = min(1.0, max(0.0, (boundary_strength - 0.65) / 1.35))
        score = hit_ratio * 0.72 + strength_score * 0.28
        if score > best.score:
            best = AxisFit(
                divisions=divisions,
                cell_size=cell_size,
                offset=0.0,
                score=score,
                threshold=threshold,
                boundary_strength=boundary_strength,
                hit_ratio=hit_ratio,
            )
    return best


def _detect_in_orientation(image: Image.Image, rotation: int) -> GridFit | None:
    bbox = image.getbbox()
    if bbox is None:
        return None
    crop = image.crop(bbox)
    if crop.width < MIN_GRID_DIVISIONS * MIN_CELL_SIZE or crop.height < MIN_GRID_DIVISIONS * MIN_CELL_SIZE:
        return None

    x_fit = _fit_axis(_axis_profile(crop, axis="x"), crop.width)
    y_fit = _fit_axis(_axis_profile(crop, axis="y"), crop.height)
    reject_reason = None
    cell_aspect = min(x_fit.cell_size, y_fit.cell_size) / max(x_fit.cell_size, y_fit.cell_size) if x_fit.cell_size and y_fit.cell_size else 0.0
    if x_fit.divisions < MIN_GRID_DIVISIONS or y_fit.divisions < MIN_GRID_DIVISIONS:
        confidence = 0.0
        reject_reason = "too_few_divisions"
    else:
        aspect_score = cell_aspect
        confidence = sqrt(x_fit.score * y_fit.score) * (0.65 + 0.35 * aspect_score)
        if cell_aspect < 0.72:
            confidence *= 0.35
            reject_reason = "cell_aspect_mismatch"

    return GridFit(
        image=image,
        bbox=bbox,
        rotation=rotation,
        columns=x_fit.divisions,
        rows=y_fit.divisions,
        cell_width=x_fit.cell_size,
        cell_height=y_fit.cell_size,
        confidence=round(confidence, 3),
        x_axis=x_fit,
        y_axis=y_fit,
        method="color-profile",
        reject_reason=reject_reason,
    )


def _cluster_positions(positions: list[int], *, max_gap: int) -> list[int]:
    clusters: list[list[int]] = []
    for position in positions:
        if not clusters or position - clusters[-1][-1] > max_gap:
            clusters.append([position])
        else:
            clusters[-1].append(position)
    return [round(sum(cluster) / len(cluster)) for cluster in clusters]


def _projection_axis_candidates(edge_image, *, axis: int) -> list[AxisFit]:
    length = edge_image.shape[1] if axis == 0 else edge_image.shape[0]
    projection = edge_image.mean(axis=axis)
    if length < MIN_GRID_DIVISIONS * MIN_CELL_SIZE:
        return []

    kernel = max(3, (length // 220) | 1)
    smoothed = np.convolve(projection, np.ones(kernel) / kernel, mode="same")
    threshold = float(smoothed.mean() + smoothed.std() * 0.85)
    peaks = [
        index
        for index in range(1, len(smoothed) - 1)
        if smoothed[index] >= threshold
        and smoothed[index] >= smoothed[index - 1]
        and smoothed[index] >= smoothed[index + 1]
    ]
    centers = _cluster_positions(peaks, max_gap=max(4, length // 240))
    diffs = [right - left for left, right in zip(centers, centers[1:])]
    bins: dict[int, int] = {}
    for diff in diffs:
        if 8 <= diff <= max(80, length // MIN_GRID_DIVISIONS):
            bucket = max(1, round(diff / 2) * 2)
            bins[bucket] = bins.get(bucket, 0) + 1
    if not bins:
        return []

    candidates: list[AxisFit] = []
    for spacing, support in sorted(bins.items(), key=lambda item: item[1], reverse=True)[:8]:
        divisions = max(MIN_GRID_DIVISIONS, min(MAX_GRID_DIVISIONS, round(length / spacing)))
        if divisions < MIN_GRID_DIVISIONS:
            continue
        expected = max(1, divisions - 1)
        coverage = min(1.0, support / expected)
        strength = min(1.0, float(smoothed.max() / threshold) / 2.5) if threshold > 0 else 0.0
        boundary_score = 0.7 * coverage + 0.3 * strength
        candidates.append(
            AxisFit(
                divisions=divisions,
                cell_size=float(spacing),
                offset=0.0,
                score=boundary_score,
                threshold=threshold,
                boundary_strength=float(smoothed.max() / threshold) if threshold > 0 else 0.0,
                hit_ratio=coverage,
            )
        )
    return candidates


def _detect_with_projection(image: Image.Image, rotation: int) -> GridFit | None:
    if cv2 is None or np is None:
        return None
    bbox = image.getbbox()
    if bbox is None:
        bbox = (0, 0, image.width, image.height)
    crop = image.crop(bbox).convert("RGB")
    if crop.width < MIN_GRID_DIVISIONS * MIN_CELL_SIZE or crop.height < MIN_GRID_DIVISIONS * MIN_CELL_SIZE:
        return None

    rgb = np.array(crop)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 140)
    x_candidates = _projection_axis_candidates(edges, axis=0)
    y_candidates = _projection_axis_candidates(edges, axis=1)
    best: tuple[float, AxisFit, AxisFit] | None = None
    for x_axis in x_candidates:
        for y_axis in y_candidates:
            aspect = min(x_axis.cell_size, y_axis.cell_size) / max(x_axis.cell_size, y_axis.cell_size)
            combined = sqrt(x_axis.score * y_axis.score) * (0.55 + 0.45 * aspect)
            if aspect < 0.72:
                combined *= 0.35
            if best is None or combined > best[0]:
                best = (combined, x_axis, y_axis)
    if best is None:
        return None

    confidence, x_axis, y_axis = best
    cell_aspect = min(x_axis.cell_size, y_axis.cell_size) / max(x_axis.cell_size, y_axis.cell_size)
    return GridFit(
        image=image,
        bbox=bbox,
        rotation=rotation,
        columns=x_axis.divisions,
        rows=y_axis.divisions,
        cell_width=(bbox[2] - bbox[0]) / x_axis.divisions,
        cell_height=(bbox[3] - bbox[1]) / y_axis.divisions,
        confidence=round(confidence, 3),
        x_axis=x_axis,
        y_axis=y_axis,
        method="edge-projection",
        reject_reason="cell_aspect_mismatch" if cell_aspect < 0.72 else None,
    )


def detect_grid(image: Image.Image) -> GridFit | None:
    rgba = image.convert("RGBA")
    fits: list[GridFit] = []
    for rotation, rotated in _rotation_variants(rgba):
        profile_fit = _detect_in_orientation(rotated, rotation)
        if profile_fit is not None:
            fits.append(profile_fit)
        projection_fit = _detect_with_projection(rotated, rotation)
        if projection_fit is not None:
            fits.append(projection_fit)
    if not fits:
        return None

    valid_fits = [
        fit
        for fit in fits
        if fit.reject_reason is None
        and fit.confidence >= DEFAULT_GRID_MIN_CONFIDENCE
    ]
    if not valid_fits:
        valid_fits = fits

    projection_fits = [
        fit
        for fit in valid_fits
        if fit.method == "edge-projection"
        and fit.confidence >= DEFAULT_GRID_MIN_CONFIDENCE
        and fit.columns >= 8
        and fit.rows >= 8
    ]
    if projection_fits:
        best_confidence = max(fit.confidence for fit in projection_fits)
        near_best = [fit for fit in projection_fits if best_confidence - fit.confidence <= 0.08]
        return max(
            near_best,
            key=lambda fit: (
                1 if fit.rotation == 0 else 0,
                fit.confidence,
                fit.columns * fit.rows,
            ),
        )

    def sort_key(fit: GridFit) -> tuple[float, int, int]:
        landscape_bonus = 1 if fit.columns >= fit.rows else 0
        no_rotation_bonus = 1 if fit.rotation == 0 else 0
        return (fit.confidence, landscape_bonus, no_rotation_bonus)

    best_confidence = max(fit.confidence for fit in valid_fits)
    near_best = [fit for fit in valid_fits if best_confidence - fit.confidence <= 0.04]
    return max(near_best, key=sort_key)


def _sample_region_color(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    region = image.crop(box).convert("RGBA")
    pixels = [pixel for pixel in _flattened_data(region) if pixel[3] > 0]
    if not pixels:
        return (0, 0, 0, 0)
    pixels.sort(key=lambda pixel: pixel[3])
    alpha = pixels[len(pixels) // 2][3]
    reds = sorted(pixel[0] for pixel in pixels)
    greens = sorted(pixel[1] for pixel in pixels)
    blues = sorted(pixel[2] for pixel in pixels)
    middle = len(pixels) // 2
    return (reds[middle], greens[middle], blues[middle], alpha)


def sample_grid(fit: GridFit) -> Image.Image:
    left, top, right, bottom = fit.bbox
    crop = fit.image.crop((left, top, right, bottom)).convert("RGBA")
    sample = Image.new(
        "RGBA",
        (fit.columns * GRID_PREVIEW_CELL, fit.rows * GRID_PREVIEW_CELL),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(sample)
    for row in range(fit.rows):
        for column in range(fit.columns):
            x0 = column * fit.cell_width
            y0 = row * fit.cell_height
            x1 = (column + 1) * fit.cell_width
            y1 = (row + 1) * fit.cell_height
            inset_x = max(1, int((x1 - x0) * 0.25))
            inset_y = max(1, int((y1 - y0) * 0.25))
            box = (
                max(0, int(round(x0)) + inset_x),
                max(0, int(round(y0)) + inset_y),
                min(crop.width, int(round(x1)) - inset_x),
                min(crop.height, int(round(y1)) - inset_y),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                box = (
                    max(0, int(round(x0))),
                    max(0, int(round(y0))),
                    min(crop.width, int(round(x1))),
                    min(crop.height, int(round(y1))),
                )
            color = _sample_region_color(crop, box)
            px0 = column * GRID_PREVIEW_CELL
            py0 = row * GRID_PREVIEW_CELL
            draw.rectangle(
                (px0, py0, px0 + GRID_PREVIEW_CELL - 1, py0 + GRID_PREVIEW_CELL - 1),
                fill=color,
            )
    return sample


def grid_debug_overlay(fit: GridFit) -> Image.Image:
    overlay = fit.image.convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    left, top, right, bottom = fit.bbox
    draw.rectangle((left, top, right - 1, bottom - 1), outline=(255, 40, 40, 255), width=2)
    for column in range(1, fit.columns):
        x = left + round(column * fit.cell_width)
        draw.line((x, top, x, bottom), fill=(0, 180, 255, 170), width=1)
    for row in range(1, fit.rows):
        y = top + round(row * fit.cell_height)
        draw.line((left, y, right, y), fill=(0, 180, 255, 170), width=1)
    draw.text(
        (max(0, left), max(0, top - 12)),
        f"{fit.columns}x{fit.rows} conf={fit.confidence:.2f} rot={fit.rotation}",
        fill=(255, 40, 40, 255),
    )
    return overlay


def grid_metadata(fit: GridFit) -> dict[str, Any]:
    return {
        "columns": fit.columns,
        "rows": fit.rows,
        "cellSize": round((fit.cell_width + fit.cell_height) / 2.0, 3),
        "cellWidth": round(fit.cell_width, 3),
        "cellHeight": round(fit.cell_height, 3),
        "rotation": fit.rotation,
        "confidence": fit.confidence,
        "bbox": list(fit.bbox),
        "method": fit.method,
        "rejectReason": fit.reject_reason,
        "axis": {
            "x": {
                "score": round(fit.x_axis.score, 3),
                "hitRatio": round(fit.x_axis.hit_ratio, 3),
                "boundaryStrength": round(fit.x_axis.boundary_strength, 3),
            },
            "y": {
                "score": round(fit.y_axis.score, 3),
                "hitRatio": round(fit.y_axis.hit_ratio, 3),
                "boundaryStrength": round(fit.y_axis.boundary_strength, 3),
            },
        },
    }


def build_grid_reference(
    *,
    image_path: Path,
    pixel_reference_path: Path,
    grid_sample_path: Path,
    debug_overlay_path: Path,
    colors: int = DEFAULT_COLORS,
    padding: int = DEFAULT_PADDING,
    min_confidence: float = DEFAULT_GRID_MIN_CONFIDENCE,
    source_label: str = "clean-subject",
) -> dict[str, Any]:
    with Image.open(image_path) as opened:
        image = opened.convert("RGBA")

    fit = detect_grid(image)
    if fit is None:
        raise GridLowConfidenceError(confidence=0.0, details={"reason": "no_visible_grid"})
    if fit.confidence < min_confidence:
        debug_overlay_path.parent.mkdir(parents=True, exist_ok=True)
        grid_debug_overlay(fit).save(debug_overlay_path)
        details = grid_metadata(fit)
        details["reason"] = "low_confidence"
        details["debugOverlay"] = str(debug_overlay_path)
        raise GridLowConfidenceError(confidence=fit.confidence, details=details)

    grid_sample = sample_grid(fit)
    pixel_reference, pixelize = normalize_to_cell(grid_sample, colors=colors, padding=padding)
    grid_sample_path.parent.mkdir(parents=True, exist_ok=True)
    pixel_reference_path.parent.mkdir(parents=True, exist_ok=True)
    debug_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    grid_sample.save(grid_sample_path)
    pixel_reference.save(pixel_reference_path)
    grid_debug_overlay(fit).save(debug_overlay_path)

    metadata = grid_metadata(fit)
    metadata.update(
        {
        "gridSample": str(grid_sample_path),
        "debugOverlay": str(debug_overlay_path),
        "previewCellSize": GRID_PREVIEW_CELL,
        "source": source_label,
        "sourceImage": str(image_path),
        }
    )
    return {
        "grid": metadata,
        "pixelize": {
            "source_image": str(grid_sample_path),
            "pixelized_image": str(pixel_reference_path),
            **pixelize,
        },
    }

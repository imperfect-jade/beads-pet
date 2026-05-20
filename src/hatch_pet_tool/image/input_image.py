"""Input image preprocessing for the no-AI run-image pipeline."""

from __future__ import annotations

from collections import deque
import math
import re
from pathlib import Path
from statistics import median

from PIL import Image

from hatch_pet_tool.image.subject import alpha_mask, analyze_subject, debug_overlay, keep_largest_component

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - pyproject installs these for normal use.
    cv2 = None
    np = None

DEFAULT_MAX_INPUT_SIDE = 1024
BACKGROUND_DISTANCE_THRESHOLD = 36.0
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def load_input_image(path: Path) -> Image.Image:
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise SystemExit(
            f"unsupported image format: {path.suffix}; expected PNG, JPG, JPEG, or WebP"
        )
    with Image.open(path) as opened:
        return opened.convert("RGBA")


def parse_crop(raw: str | None) -> tuple[int, int, int, int] | None:
    if raw is None or not raw.strip():
        return None
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 4 or not all(re.fullmatch(r"-?\d+", part) for part in parts):
        raise SystemExit("--crop must use x,y,w,h integer format")
    x, y, width, height = (int(part) for part in parts)
    if width <= 0 or height <= 0:
        raise SystemExit("--crop width and height must be positive")
    return (x, y, width, height)


def apply_crop(image: Image.Image, crop: tuple[int, int, int, int] | None) -> Image.Image:
    if crop is None:
        return image
    x, y, width, height = crop
    left = max(0, min(image.width, x))
    top = max(0, min(image.height, y))
    right = max(left, min(image.width, x + width))
    bottom = max(top, min(image.height, y + height))
    if right <= left or bottom <= top:
        raise SystemExit("--crop does not overlap the input image")
    return image.crop((left, top, right, bottom))


def resize_longest_edge(image: Image.Image, max_side: int = DEFAULT_MAX_INPUT_SIDE) -> Image.Image:
    if max_side <= 0:
        raise SystemExit("--max-input-side must be positive")
    longest = max(image.width, image.height)
    if longest <= max_side:
        return image
    scale = max_side / longest
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def parse_hex_color(raw: str) -> tuple[int, int, int]:
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", raw):
        raise SystemExit("--remove-bg must be auto, none, or #RRGGBB")
    return tuple(int(raw[index : index + 2], 16) for index in (1, 3, 5))


def edge_background_color(image: Image.Image) -> tuple[int, int, int]:
    pixels = image.load()
    step = max(1, min(image.width, image.height) // 64)
    samples = []
    for x in range(0, image.width, step):
        samples.append(pixels[x, 0])
        samples.append(pixels[x, image.height - 1])
    for y in range(0, image.height, step):
        samples.append(pixels[0, y])
        samples.append(pixels[image.width - 1, y])
    opaque_samples = [sample for sample in samples if sample[3] > 0]
    if not opaque_samples:
        return (0, 0, 0)
    return tuple(round(median(sample[index] for sample in opaque_samples)) for index in range(3))


def corner_background_color(image: Image.Image) -> tuple[int, int, int]:
    return edge_background_color(image)


def color_distance(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def _remove_background_edge_connected(
    image: Image.Image,
    background: tuple[int, int, int],
    *,
    threshold: float = BACKGROUND_DISTANCE_THRESHOLD,
) -> tuple[Image.Image, dict[str, object]]:
    output = image.copy()
    pixels = output.load()
    width, height = output.size
    visited = bytearray(width * height)
    queue: deque[tuple[int, int]] = deque()

    def enqueue_if_background(x: int, y: int) -> bool:
        index = y * width + x
        if visited[index]:
            return False
        red, green, blue, alpha = pixels[x, y]
        if alpha <= 0 or color_distance((red, green, blue), background) > threshold:
            return False
        visited[index] = 1
        queue.append((x, y))
        return True

    seed_pixels = 0
    for x in range(width):
        seed_pixels += int(enqueue_if_background(x, 0))
        seed_pixels += int(enqueue_if_background(x, height - 1))
    for y in range(height):
        seed_pixels += int(enqueue_if_background(0, y))
        seed_pixels += int(enqueue_if_background(width - 1, y))

    removed = 0
    while queue:
        x, y = queue.popleft()
        red, green, blue, alpha = pixels[x, y]
        if alpha > 0:
            pixels[x, y] = (red, green, blue, 0)
            removed += 1
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height:
                enqueue_if_background(nx, ny)

    return output, {
        "method": "edge-connected",
        "removed_pixels": removed,
        "seed_pixels": seed_pixels,
        "background_rgb": list(background),
        "threshold": threshold,
    }


def _mask_to_image(mask, size: tuple[int, int]) -> Image.Image:
    return Image.fromarray(mask.astype("uint8"), mode="L").resize(size, Image.Resampling.NEAREST)


def _fill_mask_holes(mask):
    flood = mask.copy()
    height, width = flood.shape
    canvas = np.zeros((height + 2, width + 2), dtype=np.uint8)
    cv2.floodFill(flood, canvas, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def _largest_mask_component(mask):
    components, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if components <= 1:
        return mask, {"component_count": 0, "kept_area": 0, "removed_components": 0}
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(areas.argmax()) + 1
    kept = np.where(labels == best_label, 255, 0).astype("uint8")
    return kept, {
        "component_count": int(components - 1),
        "kept_area": int(stats[best_label, cv2.CC_STAT_AREA]),
        "removed_components": int(max(0, components - 2)),
    }


def _mask_bbox(mask) -> tuple[int, int, int, int] | None:
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    return (x, y, x + width, y + height)


def _apply_alpha_mask(image: Image.Image, mask) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = Image.fromarray(mask.astype("uint8"), mode="L")
    output = rgba.copy()
    output.putalpha(alpha)
    return output


def _contours_overlay(image: Image.Image, mask) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (255, 40, 40), 2)
    return Image.fromarray(rgb).convert("RGBA")


def _rim_cleanup(
    image: Image.Image,
    mask,
    *,
    background: tuple[int, int, int],
) -> tuple[object, dict[str, object], Image.Image]:
    rgb = np.array(image.convert("RGB"))
    height, width = mask.shape
    subject_pixels = int(np.count_nonzero(mask))
    if subject_pixels == 0:
        info = {
            "enabled": True,
            "applied": False,
            "removed_pixels": 0,
            "removed_ratio": 0.0,
            "rollback_reason": "empty_mask",
            "band_width": 0,
            "background_rgb": list(background),
            "connected_regions": 0,
        }
        return mask, info, _mask_to_image(np.zeros_like(mask), image.size)

    band_width = max(3, round(min(width, height) * 0.018))
    max_depth = max(4, round(min(width, height) * 0.025))
    mask_binary = np.where(mask > 0, 255, 0).astype("uint8")
    distance_to_edge = cv2.distanceTransform(mask_binary, cv2.DIST_L2, 3)
    rim_zone = (mask_binary > 0) & (distance_to_edge <= max_depth)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    distance = np.linalg.norm(rgb.astype("int16") - np.array(background, dtype=np.int16), axis=2)
    gray_background = float(sum(background) / 3.0)
    background_delta = np.abs(value.astype("float32") - gray_background)

    dark_outline = (value <= 70) & (saturation <= 100)
    saturated_detail = saturation >= 80
    bright_subject = (value >= 205) & ((distance >= 55) | (saturation <= 28))
    edges = cv2.Canny(gray, 48, 128)
    detail_seed = np.where(dark_outline | saturated_detail | bright_subject | (edges > 0), 255, 0).astype("uint8")
    detail_protection = cv2.dilate(detail_seed, np.ones((5, 5), np.uint8), iterations=1) > 0
    residue_candidate = (
        rim_zone
        & ~dark_outline
        & ~saturated_detail
        & ~bright_subject
        & ~detail_protection
        & (
            (distance <= 78)
            | ((saturation <= 54) & (background_delta <= 72))
            | ((saturation <= 34) & (value >= 92) & (distance <= 126))
        )
    )

    cleanup_mask = np.where(residue_candidate, 255, 0).astype("uint8")
    cleanup_mask = cv2.morphologyEx(cleanup_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    component_count, labels = cv2.connectedComponents(cleanup_mask, connectivity=8)

    removed = int(np.count_nonzero(cleanup_mask))
    removed_ratio = removed / float(subject_pixels)
    connected_regions = int(max(0, component_count - 1))
    rollback_reason = None
    if removed_ratio > 0.055:
        rollback_reason = "removed_ratio_too_high"
    elif connected_regions > 180 and removed_ratio > 0.018:
        rollback_reason = "too_many_cleanup_regions"

    cleaned = mask.copy()
    applied = rollback_reason is None and removed > 0
    if applied:
        cleaned[cleanup_mask > 0] = 0

    info = {
        "enabled": True,
        "applied": applied,
        "removed_pixels": removed,
        "removed_ratio": round(removed_ratio, 6),
        "rollback_reason": rollback_reason,
        "band_width": band_width,
        "max_depth": max_depth,
        "background_rgb": list(background),
        "connected_regions": connected_regions,
    }
    return cleaned, info, _mask_to_image(cleanup_mask, image.size)


def _remove_background_auto_v2(
    image: Image.Image,
    *,
    threshold: float,
) -> tuple[Image.Image, dict[str, object], Image.Image | None, Image.Image | None, Image.Image | None]:
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV is unavailable")

    background = edge_background_color(image)
    rgb = np.array(image.convert("RGB"))
    height, width = rgb.shape[:2]
    distance = np.linalg.norm(rgb.astype("int16") - np.array(background, dtype=np.int16), axis=2)
    bg_like = distance <= max(18.0, threshold + 18.0)
    non_bg = np.where(~bg_like, 255, 0).astype("uint8")

    kernel_size = max(3, round(min(width, height) / 180))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    bbox = _mask_bbox(non_bg)
    rect = (1, 1, max(1, width - 2), max(1, height - 2))
    if bbox is not None:
        left, top, right, bottom = bbox
        pad = max(4, round(min(width, height) * 0.025))
        left = max(1, left - pad)
        top = max(1, top - pad)
        right = min(width - 1, right + pad)
        bottom = min(height - 1, bottom + pad)
        rect = (left, top, max(1, right - left), max(1, bottom - top))

    grabcut_mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    border = max(2, round(min(width, height) * 0.02))
    grabcut_mask[:border, :] = cv2.GC_BGD
    grabcut_mask[-border:, :] = cv2.GC_BGD
    grabcut_mask[:, :border] = cv2.GC_BGD
    grabcut_mask[:, -border:] = cv2.GC_BGD
    grabcut_mask[~bg_like] = cv2.GC_PR_FGD
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(rgb, grabcut_mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)

    raw_mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype("uint8")
    refined = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel, iterations=1)
    refined = _fill_mask_holes(refined)
    refined, component_info = _largest_mask_component(refined)
    refined, rim_info, rim_mask = _rim_cleanup(image, refined, background=background)

    visible_ratio = float(np.count_nonzero(refined)) / float(width * height)
    if visible_ratio < 0.015 or visible_ratio > 0.92:
        raise RuntimeError(f"auto v2 mask coverage is not reliable: {visible_ratio:.3f}")

    output = _apply_alpha_mask(image, refined)
    removed = int(width * height - np.count_nonzero(refined))
    info = {
        "mode": "auto",
        "method": "opencv-grabcut-v2",
        "removed_pixels": removed,
        "seed_pixels": int(np.count_nonzero(bg_like)),
        "background_rgb": list(background),
        "threshold": threshold,
        "mask_visible_ratio": round(visible_ratio, 6),
        "component_cleanup": component_info,
        "rim_cleanup": rim_info,
        "rect": list(rect),
    }
    return output, info, _mask_to_image(refined, image.size), _contours_overlay(image, refined), rim_mask


def remove_background(
    image: Image.Image,
    mode: str,
    *,
    threshold: float = BACKGROUND_DISTANCE_THRESHOLD,
) -> tuple[Image.Image, dict[str, object]]:
    mode = mode.strip()
    if mode.lower() == "none":
        return image.copy(), {
            "mode": "none",
            "method": "none",
            "removed_pixels": 0,
            "seed_pixels": 0,
            "background_rgb": None,
            "threshold": threshold,
        }
    if mode.lower() == "auto":
        background = edge_background_color(image)
        output, info = _remove_background_edge_connected(image, background, threshold=threshold)
        info["mode"] = "auto"
        return output, info

    background = parse_hex_color(mode)
    output, info = _remove_background_edge_connected(image, background, threshold=threshold)
    info["mode"] = mode.upper()
    return output, info


def crop_to_subject(image: Image.Image) -> Image.Image:
    bbox = image.getbbox()
    if bbox is None:
        return image
    return image.crop(bbox)


def _save_optional(image: Image.Image, path: Path | None) -> str | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return str(path)


def preprocess_input_image(
    *,
    image_path: Path,
    output_path: Path,
    crop: str | None = None,
    remove_bg: str = "auto",
    max_side: int = DEFAULT_MAX_INPUT_SIDE,
    bg_threshold: float = BACKGROUND_DISTANCE_THRESHOLD,
    source_output_path: Path | None = None,
    cropped_output_path: Path | None = None,
    background_removed_output_path: Path | None = None,
    mask_output_path: Path | None = None,
    debug_overlay_output_path: Path | None = None,
    refined_mask_output_path: Path | None = None,
    contours_overlay_output_path: Path | None = None,
    rim_cleanup_mask_output_path: Path | None = None,
    debug: bool = False,
) -> dict[str, object]:
    parsed_crop = parse_crop(crop)
    image = load_input_image(image_path)
    original_size = [image.width, image.height]
    source_output = _save_optional(image, source_output_path)
    image = apply_crop(image, parsed_crop)
    cropped_size = [image.width, image.height]
    image = resize_longest_edge(image, max_side)
    cropped_output = _save_optional(image, cropped_output_path)
    resized_size = [image.width, image.height]
    background_base = image
    refined_mask_output = None
    contours_overlay_output = None
    rim_cleanup_mask_output = None
    if remove_bg.strip().lower() == "auto":
        background = edge_background_color(image)
        edge_image, edge_info = _remove_background_edge_connected(image, background, threshold=bg_threshold)
        edge_info["mode"] = "auto"
        edge_subject = analyze_subject(edge_image)
        use_v2 = bool(edge_subject["touches_edge"]) or float(edge_subject["visible_ratio"]) > 0.55 or float(edge_subject["bbox_area_ratio"]) > 0.90
        try:
            if use_v2:
                image, background_info, refined_mask, contours, rim_mask = _remove_background_auto_v2(
                    image,
                    threshold=bg_threshold,
                )
                if debug:
                    refined_mask_output = _save_optional(refined_mask, refined_mask_output_path)
                    contours_overlay_output = _save_optional(contours, contours_overlay_output_path)
                    rim_cleanup_mask_output = _save_optional(rim_mask, rim_cleanup_mask_output_path)
            else:
                image = edge_image
                background_info = edge_info
        except Exception as exc:
            image = edge_image
            background_info = edge_info
            background_info.update(
                {
                    "mode": "auto",
                    "fallback_from": "opencv-grabcut-v2",
                    "fallback_reason": str(exc),
                }
            )
    else:
        image, background_info = remove_background(image, remove_bg, threshold=bg_threshold)
    subject_before_filter = analyze_subject(image)
    largest_component = {"kept": False, "removed_components": 0, "kept_component": None}
    if remove_bg.strip().lower() != "none":
        image, largest_component = keep_largest_component(image)
    subject_info = analyze_subject(image)
    background_removed_output = _save_optional(image, background_removed_output_path)
    mask_output = None
    debug_overlay_output = None
    if debug:
        mask_output = _save_optional(alpha_mask(image), mask_output_path)
        debug_overlay_output = _save_optional(
            debug_overlay(background_base, image, subject_info),
            debug_overlay_output_path,
        )
    image = crop_to_subject(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {
        "source_image": str(image_path),
        "clean_image": str(output_path),
        "source_output": source_output,
        "cropped_image": cropped_output,
        "background_removed_image": background_removed_output,
        "mask_image": mask_output,
        "debug_overlay": debug_overlay_output,
        "refined_mask_image": refined_mask_output,
        "contours_overlay": contours_overlay_output,
        "rim_cleanup_mask": rim_cleanup_mask_output,
        "original_size": original_size,
        "crop": list(parsed_crop) if parsed_crop else None,
        "cropped_size": cropped_size,
        "max_input_side": max_side,
        "resized_size": resized_size,
        "final_size": [image.width, image.height],
        "remove_bg": background_info,
        "subject_before_filter": subject_before_filter,
        "largest_component": largest_component,
        "subject": subject_info,
    }

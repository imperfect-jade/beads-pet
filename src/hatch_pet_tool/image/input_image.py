"""Input image preprocessing for the no-AI run-image pipeline."""

from __future__ import annotations

from collections import deque
import math
import re
from pathlib import Path
from statistics import median

from PIL import Image

from hatch_pet_tool.image.subject import alpha_mask, analyze_subject, debug_overlay, keep_largest_component

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
        mode_label = "auto"
    else:
        background = parse_hex_color(mode)
        mode_label = mode.upper()

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
        "mode": mode_label,
        "method": "edge-connected",
        "removed_pixels": removed,
        "seed_pixels": seed_pixels,
        "background_rgb": list(background),
        "threshold": threshold,
    }


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

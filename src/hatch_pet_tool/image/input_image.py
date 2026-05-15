"""Input image preprocessing for the no-AI run-image pipeline."""

from __future__ import annotations

import math
import re
from pathlib import Path

from PIL import Image

from hatch_pet_tool.image.subject import analyze_subject

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


def corner_background_color(image: Image.Image) -> tuple[int, int, int]:
    pixels = image.load()
    corners = [
        pixels[0, 0],
        pixels[image.width - 1, 0],
        pixels[0, image.height - 1],
        pixels[image.width - 1, image.height - 1],
    ]
    return tuple(round(sum(color[index] for color in corners) / 4) for index in range(3))


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
        return image, {"mode": "none", "removed_pixels": 0, "background_rgb": None}
    if mode.lower() == "auto":
        background = corner_background_color(image)
        mode_label = "auto"
    else:
        background = parse_hex_color(mode)
        mode_label = mode.upper()

    output = image.copy()
    pixels = output.load()
    removed = 0
    for y in range(output.height):
        for x in range(output.width):
            red, green, blue, alpha = pixels[x, y]
            if alpha > 0 and color_distance((red, green, blue), background) <= threshold:
                pixels[x, y] = (red, green, blue, 0)
                removed += 1
    return output, {
        "mode": mode_label,
        "removed_pixels": removed,
        "background_rgb": list(background),
        "threshold": threshold,
    }


def crop_to_subject(image: Image.Image) -> Image.Image:
    bbox = image.getbbox()
    if bbox is None:
        return image
    return image.crop(bbox)


def preprocess_input_image(
    *,
    image_path: Path,
    output_path: Path,
    crop: str | None = None,
    remove_bg: str = "auto",
    max_side: int = DEFAULT_MAX_INPUT_SIDE,
) -> dict[str, object]:
    parsed_crop = parse_crop(crop)
    image = load_input_image(image_path)
    original_size = [image.width, image.height]
    image = apply_crop(image, parsed_crop)
    cropped_size = [image.width, image.height]
    image = resize_longest_edge(image, max_side)
    resized_size = [image.width, image.height]
    image, background_info = remove_background(image, remove_bg)
    subject_info = analyze_subject(image)
    image = crop_to_subject(image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return {
        "source_image": str(image_path),
        "clean_image": str(output_path),
        "original_size": original_size,
        "crop": list(parsed_crop) if parsed_crop else None,
        "cropped_size": cropped_size,
        "max_input_side": max_side,
        "resized_size": resized_size,
        "final_size": [image.width, image.height],
        "remove_bg": background_info,
        "subject": subject_info,
    }

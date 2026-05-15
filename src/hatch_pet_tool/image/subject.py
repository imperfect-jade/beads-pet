"""Alpha-mask subject analysis for bead image candidates."""

from __future__ import annotations

from collections import deque
from typing import Any

from PIL import Image

from hatch_pet_tool.pipeline.errors import PipelineError

MIN_COMPONENT_PIXELS = 16
MULTIPLE_SUBJECT_LARGEST_RATIO = 0.70
COMPLEX_BACKGROUND_VISIBLE_RATIO = 0.85
COMPLEX_BACKGROUND_BBOX_RATIO = 0.96


def _rgba_data(image: Image.Image):
    rgba = image.convert("RGBA")
    if hasattr(rgba, "get_flattened_data"):
        return rgba, rgba.get_flattened_data()
    return rgba, rgba.getdata()


def _component_stats(image: Image.Image) -> list[dict[str, Any]]:
    pixels = image.load()
    width, height = image.size
    visited = bytearray(width * height)
    components: list[dict[str, Any]] = []

    for start_y in range(height):
        for start_x in range(width):
            start_index = start_y * width + start_x
            if visited[start_index] or pixels[start_x, start_y][3] == 0:
                continue

            queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
            visited[start_index] = 1
            count = 0
            left = right = start_x
            top = bottom = start_y

            while queue:
                x, y = queue.popleft()
                count += 1
                left = min(left, x)
                right = max(right, x)
                top = min(top, y)
                bottom = max(bottom, y)

                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    index = ny * width + nx
                    if visited[index] or pixels[nx, ny][3] == 0:
                        continue
                    visited[index] = 1
                    queue.append((nx, ny))

            if count >= MIN_COMPONENT_PIXELS:
                components.append(
                    {
                        "pixels": count,
                        "bbox": [left, top, right + 1, bottom + 1],
                    }
                )

    components.sort(key=lambda item: int(item["pixels"]), reverse=True)
    return components


def analyze_subject(image: Image.Image) -> dict[str, Any]:
    rgba, data = _rgba_data(image)
    total_pixels = rgba.width * rgba.height
    visible_pixels = 0
    colors: set[tuple[int, int, int]] = set()
    for red, green, blue, alpha in data:
        if alpha > 0:
            visible_pixels += 1
            if len(colors) <= 2048:
                colors.add((red, green, blue))

    bbox = rgba.getbbox()
    if bbox is None:
        return {
            "image_size": [rgba.width, rgba.height],
            "visible_pixels": 0,
            "visible_ratio": 0.0,
            "bbox": None,
            "bbox_area_ratio": 0.0,
            "touches_edge": False,
            "component_count": 0,
            "largest_component_ratio": 0.0,
            "unique_colors": 0,
        }

    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    components = _component_stats(rgba)
    largest_pixels = int(components[0]["pixels"]) if components else visible_pixels
    return {
        "image_size": [rgba.width, rgba.height],
        "visible_pixels": visible_pixels,
        "visible_ratio": visible_pixels / total_pixels if total_pixels else 0.0,
        "bbox": list(bbox),
        "bbox_area_ratio": (bbox_width * bbox_height) / total_pixels if total_pixels else 0.0,
        "touches_edge": bbox[0] == 0 or bbox[1] == 0 or bbox[2] == rgba.width or bbox[3] == rgba.height,
        "component_count": len(components),
        "largest_component_ratio": largest_pixels / visible_pixels if visible_pixels else 0.0,
        "unique_colors": len(colors),
        "components": components[:8],
    }


def subject_problem(analysis: dict[str, Any], *, remove_bg: str) -> PipelineError | None:
    if int(analysis["visible_pixels"]) == 0:
        return PipelineError(
            "preprocess",
            "NO_SUBJECT",
            "未检测到主体。",
            "Check that the image contains a bead subject, or try --remove-bg none / manual --crop.",
        )

    if (
        str(remove_bg).lower() == "auto"
        and bool(analysis["touches_edge"])
        and (
            float(analysis["visible_ratio"]) >= COMPLEX_BACKGROUND_VISIBLE_RATIO
            or float(analysis["bbox_area_ratio"]) >= COMPLEX_BACKGROUND_BBOX_RATIO
        )
    ):
        return PipelineError(
            "preprocess",
            "COMPLEX_BACKGROUND",
            "背景过复杂，自动背景移除后主体仍占满画面。",
            "Use a cleaner background, pass --remove-bg #RRGGBB, or crop closer with --crop x,y,w,h.",
        )

    if (
        int(analysis["component_count"]) > 1
        and float(analysis["largest_component_ratio"]) < MULTIPLE_SUBJECT_LARGEST_RATIO
    ):
        return PipelineError(
            "preprocess",
            "MULTIPLE_SUBJECTS",
            "检测到多个拼豆主体。",
            "Use --crop x,y,w,h to isolate one bead object.",
        )
    return None


def score_subject(analysis: dict[str, Any]) -> float:
    visible_ratio = float(analysis["visible_ratio"])
    bbox_ratio = float(analysis["bbox_area_ratio"])
    largest_ratio = float(analysis["largest_component_ratio"])
    unique_colors = int(analysis["unique_colors"])

    area_score = max(0.0, 1.0 - abs(visible_ratio - 0.25) / 0.25) * 35.0
    bbox_score = max(0.0, 1.0 - abs(bbox_ratio - 0.35) / 0.35) * 25.0
    component_score = largest_ratio * 25.0
    edge_penalty = 18.0 if analysis["touches_edge"] else 0.0
    color_penalty = min(12.0, max(0, unique_colors - 64) / 16.0)
    return round(max(0.0, area_score + bbox_score + component_score - edge_penalty - color_penalty), 3)

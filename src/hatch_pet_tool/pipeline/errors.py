"""Stable pipeline errors and failure summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hatch_pet_tool.core.json_io import write_json


@dataclass
class PipelineError(Exception):
    stage: str
    error_code: str
    message: str
    suggestion: str

    def __str__(self) -> str:
        return self.message


def normalize_error(exc: BaseException, *, stage: str = "pipeline") -> PipelineError:
    if isinstance(exc, PipelineError):
        return exc

    text = str(exc) or exc.__class__.__name__
    lower = text.lower()
    if "unsupported image format" in lower:
        return PipelineError(
            "input",
            "UNSUPPORTED_FORMAT",
            text,
            "Use a PNG, JPG, JPEG, or WebP input image.",
        )
    if "no visible subject" in lower or "no non-background pixels" in lower:
        return PipelineError(
            stage,
            "NO_SUBJECT",
            "未检测到主体。",
            "Check the input image, reduce background removal, or pass a manual --crop.",
        )
    if "mask coverage is not reliable" in lower or "segmentation" in lower:
        return PipelineError(
            "preprocess",
            "BACKGROUND_SEGMENTATION_FAILED",
            text,
            "Use --crop x,y,w,h, --remove-bg none, or choose a cleaner photo.",
        )
    if "crop" in lower:
        return PipelineError(
            "preprocess",
            "MANUAL_CROP_REQUIRED",
            text,
            "Use --crop x,y,w,h that overlaps the bead subject.",
        )
    if "input image not found" in lower:
        return PipelineError(
            "input",
            "INPUT_NOT_FOUND",
            text,
            "Check the image path and run the command again.",
        )
    return PipelineError(
        stage,
        "PIPELINE_FAILED",
        text,
        "Review the input image and command options, then rerun with a simpler image or manual --crop.",
    )


def failure_summary(
    *,
    run_dir: Path,
    error: PipelineError,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ok": False,
        "run_dir": str(run_dir),
        "stage": error.stage,
        "error_code": error.error_code,
        "message": error.message,
        "suggestion": error.suggestion,
    }
    if extra:
        summary.update(extra)
    return summary


def write_failure_summary(
    *,
    run_dir: Path,
    error: PipelineError,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = failure_summary(run_dir=run_dir, error=error, extra=extra)
    write_json(run_dir / "qa" / "run-summary.json", summary)
    return summary

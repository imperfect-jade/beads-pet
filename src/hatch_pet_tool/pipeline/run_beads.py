"""Run the no-AI bead pipeline from multiple candidate images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hatch_pet_tool.core.json_io import write_json
from hatch_pet_tool.core.paths import default_flutter_output_dir, slugify
from hatch_pet_tool.image.input_image import BACKGROUND_DISTANCE_THRESHOLD, DEFAULT_MAX_INPUT_SIDE, preprocess_input_image
from hatch_pet_tool.image.pixelize import DEFAULT_COLORS, DEFAULT_PADDING
from hatch_pet_tool.image.subject import score_subject, subject_problem
from hatch_pet_tool.pipeline.errors import PipelineError, normalize_error, write_failure_summary
from hatch_pet_tool.pipeline.run_image import default_run_dir, run_image_pipeline


def evaluate_candidate(
    *,
    image_path: Path,
    output_path: Path,
    source_output_path: Path,
    crop: str | None,
    remove_bg: str,
    max_input_side: int,
    bg_threshold: float,
    debug: bool,
) -> dict[str, object]:
    try:
        if not image_path.is_file():
            raise PipelineError(
                "input",
                "INPUT_NOT_FOUND",
                f"input image not found: {image_path}",
                "Check the image path and run the command again.",
            )
        preprocess = preprocess_input_image(
            image_path=image_path,
            output_path=output_path,
            crop=crop,
            remove_bg=remove_bg,
            max_side=max_input_side,
            bg_threshold=bg_threshold,
            source_output_path=source_output_path,
            cropped_output_path=output_path.parent / "cropped.png",
            background_removed_output_path=output_path.parent / "background-removed.png",
            mask_output_path=output_path.parent / "mask.png",
            debug_overlay_output_path=output_path.parent / "debug-overlay.png",
            refined_mask_output_path=output_path.parent / "mask-refined.png",
            contours_overlay_output_path=output_path.parent / "contours-overlay.png",
            rim_cleanup_mask_output_path=output_path.parent / "rim-cleanup-mask.png",
            debug=debug,
        )
        problem = subject_problem(preprocess["subject"], remove_bg=remove_bg)
        if problem is not None:
            raise problem
        score = score_subject(preprocess["subject"])
        return {
            "ok": True,
            "image": str(image_path),
            "clean_image": str(output_path),
            "score": score,
            "preprocess": preprocess,
        }
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        error = normalize_error(exc, stage="select-primary")
        return {
            "ok": False,
            "image": str(image_path),
            "clean_image": str(output_path),
            "score": 0.0,
            "stage": error.stage,
            "error_code": error.error_code,
            "message": error.message,
            "suggestion": error.suggestion,
        }


def run_beads_pipeline(
    *,
    image_paths: list[Path],
    pet_id: str,
    display_name: str,
    description: str,
    run_dir: Path,
    flutter_output_dir: Path,
    crop: str | None = None,
    remove_bg: str = "auto",
    max_input_side: int = DEFAULT_MAX_INPUT_SIDE,
    bg_threshold: float = BACKGROUND_DISTANCE_THRESHOLD,
    colors: int = DEFAULT_COLORS,
    subject_padding: int = DEFAULT_PADDING,
    render_style: str = "soft-pixel",
    render_scale: int = 2,
    reference_mode: str = "auto",
    debug: bool = False,
    force: bool = False,
) -> dict[str, object]:
    if not image_paths:
        error = PipelineError(
            "input",
            "INPUT_NOT_FOUND",
            "--images requires at least one image path",
            "Pass one or more PNG, JPG, JPEG, or WebP images.",
        )
        return write_failure_summary(run_dir=run_dir, error=error)

    if run_dir.exists() and any(run_dir.iterdir()) and not force:
        error = PipelineError(
            "setup",
            "OUTPUT_EXISTS",
            f"{run_dir} already exists and is not empty; pass --force",
            "Use --force or choose a different --output-dir.",
        )
        return write_failure_summary(run_dir=run_dir, error=error)

    candidates_dir = run_dir / "input" / "candidates"
    candidates: list[dict[str, object]] = []
    for index, image_path in enumerate(image_paths):
        clean_path = candidates_dir / f"{index:02d}" / "clean-subject.png"
        candidates.append(
            evaluate_candidate(
                image_path=image_path,
                output_path=clean_path,
                source_output_path=run_dir / "input" / f"source-{index:02d}.png",
                crop=crop,
                remove_bg=remove_bg,
                max_input_side=max_input_side,
                bg_threshold=bg_threshold,
                debug=debug,
            )
        )

    usable = [candidate for candidate in candidates if candidate["ok"]]
    if not usable:
        error = PipelineError(
            "select-primary",
            "MANUAL_CROP_REQUIRED",
            "没有可用的主图候选。",
            "Use --crop x,y,w,h, choose a cleaner background, or pass a single better image to run-image.",
        )
        return write_failure_summary(
            run_dir=run_dir,
            error=error,
            extra={"candidate_images": candidates, "primary_image": None},
        )

    primary = max(usable, key=lambda candidate: float(candidate["score"]))
    extra_summary = {
        "entrypoint": "run-beads",
        "candidate_images": candidates,
        "primary_image": primary["image"],
        "primary_score": primary["score"],
    }
    result = run_image_pipeline(
        image_path=Path(str(primary["image"])),
        pet_id=pet_id,
        display_name=display_name,
        description=description,
        run_dir=run_dir,
        flutter_output_dir=flutter_output_dir,
        crop=crop,
        remove_bg=remove_bg,
        max_input_side=max_input_side,
        bg_threshold=bg_threshold,
        colors=colors,
        subject_padding=subject_padding,
        render_style=render_style,
        render_scale=render_scale,
        reference_mode=reference_mode,
        debug=debug,
        force=force,
        allow_existing_run_dir=True,
        source_output_path=run_dir / "input" / "source-primary.png",
        extra_summary=extra_summary,
    )
    if result.get("ok"):
        write_json(run_dir / "qa" / "run-summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", required=True, help="Candidate bead or pixel reference images.")
    parser.add_argument("--pet-id", required=True)
    parser.add_argument("--display-name", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--flutter-output-dir", default="")
    parser.add_argument("--crop", default="", help="Optional crop rectangle as x,y,w,h applied to every candidate.")
    parser.add_argument("--remove-bg", default="auto", help="Background removal: auto, none, or #RRGGBB.")
    parser.add_argument("--max-input-side", type=int, default=DEFAULT_MAX_INPUT_SIDE)
    parser.add_argument("--bg-threshold", type=float, default=BACKGROUND_DISTANCE_THRESHOLD)
    parser.add_argument("--colors", type=int, default=DEFAULT_COLORS, help="Maximum visible colors in the normalized pixel subject.")
    parser.add_argument("--subject-padding", type=int, default=DEFAULT_PADDING)
    parser.add_argument(
        "--render-style",
        choices=("soft-pixel", "pixel"),
        default="soft-pixel",
        help="Reference rendering style. soft-pixel keeps pixel shapes with smoother edges.",
    )
    parser.add_argument("--render-scale", type=int, default=2, help="Internal render scale for soft-pixel output.")
    parser.add_argument(
        "--reference-mode",
        choices=("auto", "grid", "pixelize"),
        default="auto",
        help="Reference generation mode. auto tries bead-grid sampling and falls back to pixelize.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    image_paths = [Path(raw).expanduser().resolve() for raw in args.images]
    pet_id = slugify(args.pet_id)
    if not pet_id:
        raise SystemExit("--pet-id must contain at least one letter or number")
    display_name = args.display_name or pet_id.replace("-", " ").title()
    description = args.description or "A no-AI placeholder hatch-pet generated from bead image candidates."
    run_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_run_dir(pet_id).resolve()
    )
    flutter_output_dir = (
        Path(args.flutter_output_dir).expanduser().resolve()
        if args.flutter_output_dir
        else default_flutter_output_dir(pet_id)
    )

    result = run_beads_pipeline(
        image_paths=image_paths,
        pet_id=pet_id,
        display_name=display_name,
        description=description,
        run_dir=run_dir,
        flutter_output_dir=flutter_output_dir,
        crop=args.crop or None,
        remove_bg=args.remove_bg,
        max_input_side=args.max_input_side,
        bg_threshold=args.bg_threshold,
        colors=args.colors,
        subject_padding=args.subject_padding,
        render_style=args.render_style,
        render_scale=args.render_scale,
        reference_mode=args.reference_mode,
        debug=args.debug,
        force=args.force,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

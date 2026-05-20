"""Run the no-AI one-image pipeline and export Flutter assets."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from hatch_pet_tool.core.constants import ANIMATION_ROWS, ATLAS_HEIGHT, ATLAS_WIDTH, CELL_HEIGHT, CELL_WIDTH, COLUMNS, ROWS
from hatch_pet_tool.core.json_io import write_json
from hatch_pet_tool.core.paths import default_flutter_output_dir, slugify
from hatch_pet_tool.flutter.export import export_flutter_asset
from hatch_pet_tool.image.algorithmic import generate_algorithmic_frames
from hatch_pet_tool.image.bead_grid import GridLowConfidenceError, build_grid_reference
from hatch_pet_tool.image.compose import compose_from_frames, save_outputs
from hatch_pet_tool.image.contact_sheet import main as _contact_sheet_main
from hatch_pet_tool.image.input_image import BACKGROUND_DISTANCE_THRESHOLD, DEFAULT_MAX_INPUT_SIDE, preprocess_input_image
from hatch_pet_tool.image.pixelize import DEFAULT_COLORS, DEFAULT_PADDING, pixelize_image
from hatch_pet_tool.image.subject import subject_problem
from hatch_pet_tool.image.validate import main as _validate_main
from hatch_pet_tool.pipeline.errors import PipelineError, normalize_error, write_failure_summary

import sys


def default_run_dir(pet_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path.cwd() / "output" / "runs" / f"{pet_id}-{timestamp}"


def _run_module_main(main_func, argv: list[str]) -> None:
    old_argv = sys.argv
    try:
        sys.argv = argv
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                main_func()
            except SystemExit as exc:
                if exc.code not in (0, None):
                    raise
    finally:
        sys.argv = old_argv


def write_request(
    *,
    run_dir: Path,
    image_path: Path,
    pet_id: str,
    display_name: str,
    description: str,
    clean_image_path: Path,
    base_pixel_pet_path: Path,
    rendered_pixel_pet_path: Path,
    pixelized_image_path: Path,
    preprocess: dict[str, object],
    pixelize: dict[str, object],
    reference_mode: str,
    reference_source: str,
    grid: dict[str, object] | None = None,
    grid_error: dict[str, object] | None = None,
) -> Path:
    request = {
        "pet_id": pet_id,
        "display_name": display_name,
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "algorithmic-run-image",
        "source_image": str(image_path),
        "clean_image": str(clean_image_path),
        "base_pixel_pet": str(base_pixel_pet_path),
        "rendered_pixel_pet": str(rendered_pixel_pet_path),
        "pixelized_image": str(pixelized_image_path),
        "reference_mode": reference_mode,
        "reference_source": reference_source,
        "preprocess": preprocess,
        "pixelize": pixelize,
        "atlas": {
            "columns": COLUMNS,
            "rows": ROWS,
            "cell_width": CELL_WIDTH,
            "cell_height": CELL_HEIGHT,
            "width": ATLAS_WIDTH,
            "height": ATLAS_HEIGHT,
        },
        "animation_rows": [
            {"state": row.state, "row": row.row, "frames": row.frames, "fps": row.fps}
            for row in ANIMATION_ROWS
        ],
    }
    if grid is not None:
        request["grid"] = grid
    if grid_error is not None:
        request["grid_error"] = grid_error
    request_path = run_dir / "pet_request.json"
    write_json(request_path, request)
    return request_path


def _grid_error_summary(exc: GridLowConfidenceError) -> dict[str, object]:
    return {
        "error_code": "GRID_LOW_CONFIDENCE",
        "message": str(exc),
        "confidence": exc.confidence,
        "details": exc.details,
    }


def build_reference(
    *,
    clean_input: Path,
    reference_dir: Path,
    preprocess_dir: Path,
    reference_mode: str,
    colors: int,
    subject_padding: int,
    render_style: str = "soft-pixel",
    render_scale: int = 2,
    grid_candidates: list[tuple[str, Path]] | None = None,
    allow_pixelize_fallback: bool = True,
) -> dict[str, object]:
    pixel_reference = reference_dir / "pixel-reference.png"
    base_pixel_pet = reference_dir / "base-pixel-pet.png"
    rendered_pixel_pet = reference_dir / "rendered-pixel-pet.png"
    mode = reference_mode.lower()
    if mode not in {"auto", "grid", "pixelize"}:
        raise PipelineError(
            "reference",
            "INVALID_REFERENCE_MODE",
            f"unsupported reference mode: {reference_mode}",
            "Use --reference-mode auto, grid, or pixelize.",
        )

    grid_candidates = grid_candidates or [("clean-subject", clean_input)]
    candidate_results: list[dict[str, object]] = []
    if mode in {"auto", "grid"}:
        best_error: GridLowConfidenceError | None = None
        for label, candidate_path in grid_candidates:
            if not candidate_path.is_file():
                continue
            try:
                grid_result = build_grid_reference(
                    image_path=candidate_path,
                    pixel_reference_path=pixel_reference,
                    base_pixel_pet_path=base_pixel_pet,
                    rendered_pixel_pet_path=rendered_pixel_pet,
                    grid_sample_path=reference_dir / "grid-sample.png",
                    debug_overlay_path=preprocess_dir / "grid-debug-overlay.png",
                    palette_preview_path=reference_dir / "palette-preview.png",
                    colors=colors,
                    padding=subject_padding,
                    render_style=render_style,
                    render_scale=render_scale,
                    source_label=label,
                )
                candidate_results.append(
                    {
                        "source": label,
                        "image": str(candidate_path),
                        "ok": True,
                        "grid": grid_result["grid"],
                    }
                )
                write_json(preprocess_dir / "grid-candidates.json", candidate_results)
                return {
                    "reference_mode": mode,
                    "reference_source": "grid",
                    "base_pixel_pet": base_pixel_pet,
                    "rendered_pixel_pet": rendered_pixel_pet,
                    "pixel_reference": pixel_reference,
                    "pixelize": grid_result["pixelize"],
                    "grid": grid_result["grid"],
                    "grid_error": None,
                    "grid_candidates": candidate_results,
                }
            except GridLowConfidenceError as exc:
                if best_error is None or exc.confidence > best_error.confidence:
                    best_error = exc
                candidate_results.append(
                    {
                        "source": label,
                        "image": str(candidate_path),
                        "ok": False,
                        **_grid_error_summary(exc),
                    }
                )

        write_json(preprocess_dir / "grid-candidates.json", candidate_results)
        grid_error = _grid_error_summary(best_error or GridLowConfidenceError(confidence=0.0))
        if mode == "grid" or not allow_pixelize_fallback:
            raise PipelineError(
                "reference",
                "GRID_LOW_CONFIDENCE",
                grid_error["message"],
                "Use --crop x,y,w,h, choose a cleaner front-facing bead image, or use --reference-mode pixelize.",
            )
    else:
        grid_error = None

    pixelize = pixelize_image(
        image_path=clean_input,
        output_path=pixel_reference,
        colors=colors,
        padding=subject_padding,
        base_output_path=base_pixel_pet,
        rendered_output_path=rendered_pixel_pet,
        render_style=render_style,
        render_scale=render_scale,
        palette_preview_path=reference_dir / "palette-preview.png",
    )
    return {
        "reference_mode": mode,
        "reference_source": "pixelize",
        "base_pixel_pet": base_pixel_pet,
        "rendered_pixel_pet": rendered_pixel_pet,
        "pixel_reference": pixel_reference,
        "pixelize": pixelize,
        "grid": None,
        "grid_error": grid_error,
        "grid_candidates": candidate_results,
    }


def run_image_pipeline(
    *,
    image_path: Path,
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
    allow_existing_run_dir: bool = False,
    source_output_path: Path | None = None,
    extra_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    frames_root = run_dir / "frames"
    final_dir = run_dir / "final"
    input_dir = run_dir / "input"
    preprocess_dir = run_dir / "preprocess"
    reference_dir = run_dir / "reference"
    qa_dir = run_dir / "qa"
    try:
        if not image_path.is_file():
            raise PipelineError(
                "input",
                "INPUT_NOT_FOUND",
                f"input image not found: {image_path}",
                "Check the image path and run the command again.",
            )
        if run_dir.exists() and any(run_dir.iterdir()) and not force and not allow_existing_run_dir:
            raise PipelineError(
                "setup",
                "OUTPUT_EXISTS",
                f"{run_dir} already exists and is not empty; pass --force",
                "Use --force or choose a different --output-dir.",
            )

        final_dir.mkdir(parents=True, exist_ok=True)
        input_dir.mkdir(parents=True, exist_ok=True)
        preprocess_dir.mkdir(parents=True, exist_ok=True)
        reference_dir.mkdir(parents=True, exist_ok=True)
        qa_dir.mkdir(parents=True, exist_ok=True)
        clean_input = preprocess_dir / "clean-subject.png"
        background_removed = preprocess_dir / "background-removed.png"
        preprocess = preprocess_input_image(
            image_path=image_path,
            output_path=clean_input,
            crop=crop,
            remove_bg=remove_bg,
            max_side=max_input_side,
            bg_threshold=bg_threshold,
            source_output_path=source_output_path or input_dir / "source-00.png",
            cropped_output_path=preprocess_dir / "cropped.png",
            background_removed_output_path=background_removed,
            mask_output_path=preprocess_dir / "mask.png",
            debug_overlay_output_path=preprocess_dir / "debug-overlay.png",
            refined_mask_output_path=preprocess_dir / "mask-refined.png",
            contours_overlay_output_path=preprocess_dir / "contours-overlay.png",
            rim_cleanup_mask_output_path=preprocess_dir / "rim-cleanup-mask.png",
            debug=debug,
        )
        problem = subject_problem(preprocess["subject"], remove_bg=remove_bg)
        if problem is not None and reference_mode == "pixelize":
            raise problem
        if problem is not None and problem.error_code == "NO_SUBJECT":
            raise problem

        grid_candidates = [
            ("cropped", Path(str(preprocess["cropped_image"]))),
            ("background-removed", Path(str(preprocess["background_removed_image"]))),
            ("clean-subject", clean_input),
        ]

        try:
            reference = build_reference(
                clean_input=clean_input,
                reference_dir=reference_dir,
                preprocess_dir=preprocess_dir,
                reference_mode=reference_mode,
                colors=colors,
                subject_padding=subject_padding,
                render_style=render_style,
                render_scale=render_scale,
                grid_candidates=grid_candidates,
                allow_pixelize_fallback=problem is None,
            )
        except PipelineError:
            if problem is not None and reference_mode == "auto":
                raise problem
            raise
        if problem is not None and reference["reference_source"] != "grid":
            raise problem
        pixelized_input = Path(str(reference["pixel_reference"]))
        base_pixel_pet = Path(str(reference["base_pixel_pet"]))
        rendered_pixel_pet = Path(str(reference["rendered_pixel_pet"]))
        pixelize = reference["pixelize"]

        request_path = write_request(
            run_dir=run_dir,
            image_path=image_path,
            pet_id=pet_id,
            display_name=display_name,
            description=description,
            clean_image_path=clean_input,
            base_pixel_pet_path=base_pixel_pet,
            rendered_pixel_pet_path=rendered_pixel_pet,
            pixelized_image_path=pixelized_input,
            preprocess=preprocess,
            pixelize=pixelize,
            reference_mode=str(reference["reference_mode"]),
            reference_source=str(reference["reference_source"]),
            grid=reference["grid"] if isinstance(reference["grid"], dict) else None,
            grid_error=reference["grid_error"] if isinstance(reference["grid_error"], dict) else None,
        )
        frames_manifest = generate_algorithmic_frames(
            pixelized_input,
            frames_root,
            render_style=render_style,
        )
        write_json(frames_root / "frames-manifest.json", frames_manifest)

        atlas = compose_from_frames(frames_root)
        spritesheet_png = final_dir / "spritesheet.png"
        spritesheet_webp = final_dir / "spritesheet.webp"
        save_outputs(atlas, spritesheet_png, spritesheet_webp)

        validation_path = final_dir / "validation.json"
        _run_module_main(
            _validate_main,
            [
                "hatch-pet-tool validate",
                str(spritesheet_webp),
                "--json-out",
                str(validation_path),
            ],
        )

        contact_sheet = qa_dir / "contact-sheet.png"
        _run_module_main(
            _contact_sheet_main,
            [
                "hatch-pet-tool contact-sheet",
                str(spritesheet_webp),
                "--output",
                str(contact_sheet),
            ],
        )

        flutter_result = export_flutter_asset(
            spritesheet=spritesheet_webp,
            output_dir=flutter_output_dir,
            pet_id=pet_id,
            display_name=display_name,
            description=description,
            force=force,
        )

        summary = {
            "ok": True,
            "run_dir": str(run_dir),
            "request": str(request_path),
            "clean_input": str(clean_input),
            "base_pixel_pet": str(base_pixel_pet),
            "rendered_pixel_pet": str(rendered_pixel_pet),
            "pixelized_input": str(pixelized_input),
            "background_removed": str(background_removed),
            "pixel_reference": str(pixelized_input),
            "palette": pixelize.get("colors", {}).get("palette"),
            "palette_source": pixelize.get("colors", {}).get("palette_source"),
            "palette_counts": pixelize.get("colors", {}).get("palette_counts"),
            "color_restore": pixelize.get("colors"),
            "render_style": render_style,
            "render_scale": render_scale,
            "reference_mode": reference["reference_mode"],
            "reference_source": reference["reference_source"],
            "grid_color_mode": pixelize.get("grid_color_mode"),
            "raw_sample_colors": pixelize.get("raw_sample_colors"),
            "final_sample_colors": pixelize.get("final_sample_colors"),
            "changed_cells": pixelize.get("changed_cells"),
            "template_background_removed_cells": pixelize.get("template_background_removed_cells"),
            "opaque_cells": pixelize.get("opaque_cells"),
            "preprocess": preprocess,
            "pixelize": pixelize,
            "spritesheet": str(spritesheet_webp),
            "validation": str(validation_path),
            "contact_sheet": str(contact_sheet),
            "flutter_manifest": str(flutter_result["manifest"]),
            "flutter_spritesheet": str(flutter_result["spritesheet"]),
        }
        if reference["grid"] is not None:
            summary["grid"] = reference["grid"]
        if reference["grid_error"] is not None:
            summary["grid_error"] = reference["grid_error"]
        if reference.get("grid_candidates"):
            summary["grid_candidates"] = reference["grid_candidates"]
        if extra_summary:
            summary.update(extra_summary)
        write_json(qa_dir / "run-summary.json", summary)
        return summary
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        error = normalize_error(exc, stage="run-image")
        return write_failure_summary(run_dir=run_dir, error=error, extra=extra_summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Input bead or pixel pet reference image.")
    parser.add_argument("--pet-id", default="")
    parser.add_argument("--display-name", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--flutter-output-dir", default="")
    parser.add_argument("--crop", default="", help="Optional crop rectangle as x,y,w,h in source pixels.")
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

    image_path = Path(args.image).expanduser().resolve()
    display_name = args.display_name or image_path.stem.replace("-", " ").replace("_", " ").title()
    pet_id = slugify(args.pet_id or display_name)
    if not pet_id:
        raise SystemExit("--pet-id or a usable image filename is required")
    description = args.description or "A no-AI placeholder hatch-pet generated from one input image."
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

    result = run_image_pipeline(
        image_path=image_path,
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

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
from hatch_pet_tool.image.compose import compose_from_frames, save_outputs
from hatch_pet_tool.image.contact_sheet import main as _contact_sheet_main
from hatch_pet_tool.image.input_image import DEFAULT_MAX_INPUT_SIDE, preprocess_input_image
from hatch_pet_tool.image.pixelize import DEFAULT_COLORS, pixelize_image
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
    pixelized_image_path: Path,
    preprocess: dict[str, object],
    pixelize: dict[str, object],
) -> Path:
    request = {
        "pet_id": pet_id,
        "display_name": display_name,
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "algorithmic-run-image",
        "source_image": str(image_path),
        "clean_image": str(clean_image_path),
        "pixelized_image": str(pixelized_image_path),
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
    request_path = run_dir / "pet_request.json"
    write_json(request_path, request)
    return request_path


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
    colors: int = DEFAULT_COLORS,
    force: bool = False,
    allow_existing_run_dir: bool = False,
    extra_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    frames_root = run_dir / "frames"
    final_dir = run_dir / "final"
    input_dir = run_dir / "input"
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
        qa_dir.mkdir(parents=True, exist_ok=True)
        clean_input = input_dir / "clean.png"
        preprocess = preprocess_input_image(
            image_path=image_path,
            output_path=clean_input,
            crop=crop,
            remove_bg=remove_bg,
            max_side=max_input_side,
        )
        problem = subject_problem(preprocess["subject"], remove_bg=remove_bg)
        if problem is not None:
            raise problem

        pixelized_input = input_dir / "pixelized.png"
        pixelize = pixelize_image(
            image_path=clean_input,
            output_path=pixelized_input,
            colors=colors,
        )

        request_path = write_request(
            run_dir=run_dir,
            image_path=image_path,
            pet_id=pet_id,
            display_name=display_name,
            description=description,
            clean_image_path=clean_input,
            pixelized_image_path=pixelized_input,
            preprocess=preprocess,
            pixelize=pixelize,
        )
        frames_manifest = generate_algorithmic_frames(pixelized_input, frames_root)
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
            "pixelized_input": str(pixelized_input),
            "preprocess": preprocess,
            "pixelize": pixelize,
            "spritesheet": str(spritesheet_webp),
            "validation": str(validation_path),
            "contact_sheet": str(contact_sheet),
            "flutter_manifest": str(flutter_result["manifest"]),
            "flutter_spritesheet": str(flutter_result["spritesheet"]),
        }
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
    parser.add_argument("--colors", type=int, default=DEFAULT_COLORS, help="Maximum visible colors in the normalized pixel subject.")
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
        colors=args.colors,
        force=args.force,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

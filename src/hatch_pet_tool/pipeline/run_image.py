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
from hatch_pet_tool.image.validate import main as _validate_main

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
) -> Path:
    request = {
        "pet_id": pet_id,
        "display_name": display_name,
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": "algorithmic-run-image",
        "source_image": str(image_path),
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
    force: bool = False,
) -> dict[str, object]:
    if not image_path.is_file():
        raise SystemExit(f"input image not found: {image_path}")
    if run_dir.exists() and any(run_dir.iterdir()) and not force:
        raise SystemExit(f"{run_dir} already exists and is not empty; pass --force")

    frames_root = run_dir / "frames"
    final_dir = run_dir / "final"
    qa_dir = run_dir / "qa"
    final_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    request_path = write_request(
        run_dir=run_dir,
        image_path=image_path,
        pet_id=pet_id,
        display_name=display_name,
        description=description,
    )
    frames_manifest = generate_algorithmic_frames(image_path, frames_root)
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
        "spritesheet": str(spritesheet_webp),
        "validation": str(validation_path),
        "contact_sheet": str(contact_sheet),
        "flutter_manifest": str(flutter_result["manifest"]),
        "flutter_spritesheet": str(flutter_result["spritesheet"]),
    }
    write_json(qa_dir / "run-summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Input bead or pixel pet reference image.")
    parser.add_argument("--pet-id", default="")
    parser.add_argument("--display-name", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--flutter-output-dir", default="")
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
        force=args.force,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

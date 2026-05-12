"""Export validated hatch-pet assets in the Flutter Todolist manifest shape."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from hatch_pet_tool.core.constants import (
    ANIMATION_ROWS,
    ATLAS_HEIGHT,
    ATLAS_WIDTH,
    CELL_HEIGHT,
    CELL_WIDTH,
    COLUMNS,
    FLUTTER_ACTIONS,
    ROWS,
)
from hatch_pet_tool.core.json_io import read_json, write_json
from hatch_pet_tool.core.paths import default_flutter_output_dir, slugify


def _load_request(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    request_path = run_dir / "pet_request.json"
    if not request_path.is_file():
        return {}
    return read_json(request_path)


def _infer_spritesheet(run_dir: Path | None, raw_spritesheet: str) -> Path:
    if raw_spritesheet:
        return Path(raw_spritesheet).expanduser().resolve()
    if run_dir is None:
        raise SystemExit("--spritesheet is required when --run-dir is not provided")
    spritesheet = run_dir / "final" / "spritesheet.webp"
    if not spritesheet.is_file():
        raise SystemExit(f"spritesheet not found: {spritesheet}")
    return spritesheet.resolve()


def validate_spritesheet(path: Path) -> None:
    with Image.open(path) as image:
        if image.size != (ATLAS_WIDTH, ATLAS_HEIGHT):
            raise SystemExit(
                f"expected {ATLAS_WIDTH}x{ATLAS_HEIGHT}, got {image.width}x{image.height}"
            )
        if image.format not in {"PNG", "WEBP"}:
            raise SystemExit(f"expected PNG or WebP, got {image.format}")


def build_flutter_manifest(
    *,
    pet_id: str,
    display_name: str,
    description: str,
    image_name: str,
) -> dict[str, Any]:
    actions = {}
    for animation in ANIMATION_ROWS:
        action = FLUTTER_ACTIONS[animation.state]
        actions[action] = {
            "row": animation.row,
            "frames": animation.frames,
            "fps": animation.fps,
        }
    return {
        "id": pet_id,
        "displayName": display_name,
        "description": description,
        "image": image_name,
        "frameWidth": CELL_WIDTH,
        "frameHeight": CELL_HEIGHT,
        "columns": COLUMNS,
        "rows": ROWS,
        "actions": actions,
    }


def export_flutter_asset(
    *,
    spritesheet: Path,
    output_dir: Path,
    pet_id: str,
    display_name: str,
    description: str,
    force: bool = False,
) -> dict[str, Any]:
    validate_spritesheet(spritesheet)
    image_name = f"{pet_id}_hatch_spritesheet.webp"
    manifest_name = f"{pet_id}_hatch_pet.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_sheet = output_dir / image_name
    target_manifest = output_dir / manifest_name
    if not force and (target_sheet.exists() or target_manifest.exists()):
        raise SystemExit(f"{output_dir} already contains Flutter pet files; pass --force")

    with Image.open(spritesheet) as image:
        image.convert("RGBA").save(
            target_sheet,
            format="WEBP",
            lossless=True,
            quality=100,
            method=6,
        )
    manifest = build_flutter_manifest(
        pet_id=pet_id,
        display_name=display_name,
        description=description,
        image_name=image_name,
    )
    write_json(target_manifest, manifest)
    return {
        "ok": True,
        "output_dir": str(output_dir),
        "spritesheet": str(target_sheet),
        "manifest": str(target_manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--spritesheet", default="")
    parser.add_argument("--pet-id", default="")
    parser.add_argument("--display-name", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    request = _load_request(run_dir)
    display_name = args.display_name or str(request.get("display_name") or "")
    pet_id = slugify(args.pet_id or str(request.get("pet_id") or display_name))
    if not pet_id:
        raise SystemExit("--pet-id or pet_request.json pet_id is required")
    if not display_name:
        display_name = pet_id.replace("-", " ").title()
    description = args.description or str(
        request.get("description") or "A hatch-pet style pixel companion."
    )

    spritesheet = _infer_spritesheet(run_dir, args.spritesheet)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_flutter_output_dir(pet_id)
    )
    result = export_flutter_asset(
        spritesheet=spritesheet,
        output_dir=output_dir,
        pet_id=pet_id,
        display_name=display_name,
        description=description,
        force=args.force,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

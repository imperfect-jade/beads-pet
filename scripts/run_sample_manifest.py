"""Run local bead samples from a manifest without committing real photos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hatch_pet_tool.core.paths import default_flutter_output_dir, slugify
from hatch_pet_tool.pipeline.run_image import run_image_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="samples/manifest.local.json")
    parser.add_argument("--output-dir", default="output/sample-runs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_root = Path(args.output_dir).expanduser().resolve()
    results = []
    for item in data.get("samples", []):
        sample_id = slugify(str(item["id"]))
        raw_image_path = Path(str(item["image"]))
        image_path = (
            raw_image_path.expanduser().resolve()
            if raw_image_path.is_absolute()
            else (Path.cwd() / raw_image_path).resolve()
        )
        result = run_image_pipeline(
            image_path=image_path,
            pet_id=sample_id,
            display_name=str(item.get("displayName") or sample_id.replace("-", " ").title()),
            description=str(item.get("notes") or "Local bead sample."),
            run_dir=output_root / sample_id,
            flutter_output_dir=default_flutter_output_dir(sample_id),
            crop=item.get("crop"),
            remove_bg=str(item.get("removeBg") or "auto"),
            force=args.force,
        )
        results.append({"id": sample_id, "ok": result.get("ok"), "run_dir": result.get("run_dir")})
    print(json.dumps({"ok": True, "results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

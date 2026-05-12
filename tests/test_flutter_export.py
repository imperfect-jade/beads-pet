import json

from PIL import Image

from hatch_pet_tool.core.constants import ATLAS_HEIGHT, ATLAS_WIDTH
from hatch_pet_tool.flutter.export import build_flutter_manifest, export_flutter_asset


def test_build_flutter_manifest_matches_todolist_shape():
    manifest = build_flutter_manifest(
        pet_id="test-pet",
        display_name="Test Pet",
        description="A test pet.",
        image_name="test-pet_hatch_spritesheet.webp",
    )

    assert manifest["image"] == "test-pet_hatch_spritesheet.webp"
    assert manifest["frameWidth"] == 192
    assert manifest["frameHeight"] == 208
    assert manifest["columns"] == 8
    assert manifest["rows"] == 9
    assert manifest["actions"]["runningRight"] == {"row": 1, "frames": 8, "fps": 8}
    assert manifest["actions"]["feed"] == {"row": 8, "frames": 6, "fps": 6}


def test_export_flutter_asset_writes_manifest_and_webp(tmp_path):
    spritesheet = tmp_path / "source.png"
    Image.new("RGBA", (ATLAS_WIDTH, ATLAS_HEIGHT), (0, 0, 0, 0)).save(spritesheet)

    result = export_flutter_asset(
        spritesheet=spritesheet,
        output_dir=tmp_path / "out",
        pet_id="test-pet",
        display_name="Test Pet",
        description="A test pet.",
    )

    assert result["ok"] is True
    manifest_path = tmp_path / "out" / "test-pet_hatch_pet.json"
    sheet_path = tmp_path / "out" / "test-pet_hatch_spritesheet.webp"
    assert manifest_path.is_file()
    assert sheet_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["image"] == sheet_path.name

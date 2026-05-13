import json

from PIL import Image, ImageDraw

from hatch_pet_tool.core.constants import ANIMATION_ROWS, ATLAS_HEIGHT, ATLAS_WIDTH, CELL_HEIGHT, CELL_WIDTH
from hatch_pet_tool.image.algorithmic import generate_algorithmic_frames
from hatch_pet_tool.pipeline.run_image import run_image_pipeline


def _sample_image(path):
    image = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 12, 46, 48), fill=(230, 80, 90, 255), outline=(20, 20, 20, 255))
    draw.rectangle((26, 22, 30, 26), fill=(20, 20, 20, 255))
    draw.rectangle((36, 22, 40, 26), fill=(20, 20, 20, 255))
    image.save(path)


def test_algorithmic_frames_have_expected_rows_and_cell_size(tmp_path):
    source = tmp_path / "beads.png"
    _sample_image(source)

    manifest = generate_algorithmic_frames(source, tmp_path / "frames")

    assert manifest["ok"] is True
    assert [row["state"] for row in manifest["rows"]] == [row.state for row in ANIMATION_ROWS]
    for row in ANIMATION_ROWS:
        files = sorted((tmp_path / "frames" / row.state).glob("*.png"))
        assert len(files) == row.frames
        with Image.open(files[0]) as frame:
            assert frame.size == (CELL_WIDTH, CELL_HEIGHT)


def test_run_image_pipeline_exports_flutter_assets(tmp_path, monkeypatch):
    source = tmp_path / "beads.png"
    run_dir = tmp_path / "run"
    flutter_dir = tmp_path / "flutter"
    _sample_image(source)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_image_pipeline(
        image_path=source,
        pet_id="beads-test",
        display_name="Beads Test",
        description="A test pet.",
        run_dir=run_dir,
        flutter_output_dir=flutter_dir,
        crop="0,0,64,64",
        remove_bg="auto",
        max_input_side=64,
    )

    assert result["ok"] is True
    assert (run_dir / "input" / "clean.png").is_file()
    assert result["clean_input"] == str(run_dir / "input" / "clean.png")
    assert result["preprocess"]["remove_bg"]["mode"] == "auto"
    assert (run_dir / "final" / "spritesheet.webp").is_file()
    assert (run_dir / "final" / "validation.json").is_file()
    assert (run_dir / "qa" / "contact-sheet.png").is_file()
    assert (flutter_dir / "beads-test_hatch_pet.json").is_file()
    assert (flutter_dir / "beads-test_hatch_spritesheet.webp").is_file()

    with Image.open(run_dir / "final" / "spritesheet.webp") as spritesheet:
        assert spritesheet.size == (ATLAS_WIDTH, ATLAS_HEIGHT)
    validation = json.loads((run_dir / "final" / "validation.json").read_text(encoding="utf-8"))
    assert validation["ok"] is True
    manifest = json.loads((flutter_dir / "beads-test_hatch_pet.json").read_text(encoding="utf-8"))
    assert manifest["image"] == "beads-test_hatch_spritesheet.webp"
    assert manifest["actions"]["idle"] == {"row": 0, "frames": 6, "fps": 6}

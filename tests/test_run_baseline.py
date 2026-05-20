import json

from PIL import Image, ImageDraw

from hatch_pet_tool.pipeline.run_beads import run_beads_pipeline
from hatch_pet_tool.pipeline.run_image import run_image_pipeline


def _transparent_subject(path):
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 16, 44, 48), fill=(220, 50, 80, 255))
    image.save(path)


def _complex_background(path):
    image = Image.new("RGBA", (48, 48), (80, 80, 80, 255))
    pixels = image.load()
    for y in range(48):
        for x in range(48):
            pixels[x, y] = ((x * 5 + y * 3) % 255, (x * 7) % 255, (y * 9) % 255, 255)
    image.save(path)


def _multiple_subjects(path):
    image = Image.new("RGBA", (80, 48), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 12, 28, 32), fill=(220, 50, 80, 255))
    draw.rectangle((50, 12, 70, 32), fill=(50, 80, 220, 255))
    image.save(path)


def test_run_image_writes_failure_summary_when_no_subject(tmp_path):
    source = tmp_path / "empty.png"
    run_dir = tmp_path / "run"
    Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(source)

    result = run_image_pipeline(
        image_path=source,
        pet_id="empty",
        display_name="Empty",
        description="",
        run_dir=run_dir,
        flutter_output_dir=tmp_path / "flutter",
        remove_bg="none",
    )

    assert result["ok"] is False
    assert result["error_code"] == "NO_SUBJECT"
    summary = json.loads((run_dir / "qa" / "run-summary.json").read_text(encoding="utf-8"))
    assert summary["error_code"] == "NO_SUBJECT"


def test_run_image_reports_complex_background(tmp_path):
    source = tmp_path / "complex.png"
    _complex_background(source)

    result = run_image_pipeline(
        image_path=source,
        pet_id="complex",
        display_name="Complex",
        description="",
        run_dir=tmp_path / "run",
        flutter_output_dir=tmp_path / "flutter",
        remove_bg="auto",
        bg_threshold=0,
    )

    assert result["ok"] is False
    assert result["error_code"] == "COMPLEX_BACKGROUND"
    assert "背景过复杂" in result["message"]
    summary = json.loads((tmp_path / "run" / "qa" / "run-summary.json").read_text(encoding="utf-8"))
    assert summary["message"] == result["message"]


def test_run_image_keeps_largest_subject(tmp_path, monkeypatch):
    source = tmp_path / "multiple.png"
    _multiple_subjects(source)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_image_pipeline(
        image_path=source,
        pet_id="multiple",
        display_name="Multiple",
        description="",
        run_dir=tmp_path / "run",
        flutter_output_dir=tmp_path / "flutter",
        remove_bg="auto",
    )

    assert result["ok"] is True
    assert result["preprocess"]["largest_component"]["removed_components"] == 1


def test_run_beads_selects_best_candidate_and_exports_assets(tmp_path, monkeypatch):
    bad = tmp_path / "bad.png"
    good = tmp_path / "good.png"
    _complex_background(bad)
    _transparent_subject(good)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_beads_pipeline(
        image_paths=[bad, good],
        pet_id="beads-candidate",
        display_name="Beads Candidate",
        description="",
        run_dir=tmp_path / "run",
        flutter_output_dir=tmp_path / "flutter",
        remove_bg="auto",
        bg_threshold=0,
        colors=8,
        debug=True,
    )

    assert result["ok"] is True
    assert result["primary_image"] == str(good)
    assert len(result["candidate_images"]) == 2
    assert result["candidate_images"][0]["error_code"] == "COMPLEX_BACKGROUND"
    assert (tmp_path / "run" / "input" / "source-00.png").is_file()
    assert (tmp_path / "run" / "input" / "source-01.png").is_file()
    assert (tmp_path / "run" / "input" / "source-primary.png").is_file()
    assert (tmp_path / "run" / "input" / "candidates" / "01" / "mask.png").is_file()
    assert (tmp_path / "run" / "qa" / "run-summary.json").is_file()
    assert (tmp_path / "flutter" / "beads-candidate_hatch_pet.json").is_file()


def test_run_image_debug_outputs_and_subject_padding(tmp_path, monkeypatch):
    source = tmp_path / "subject.png"
    _transparent_subject(source)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_image_pipeline(
        image_path=source,
        pet_id="debug-subject",
        display_name="Debug Subject",
        description="",
        run_dir=tmp_path / "run",
        flutter_output_dir=tmp_path / "flutter",
        remove_bg="auto",
        subject_padding=30,
        debug=True,
    )

    assert result["ok"] is True
    assert (tmp_path / "run" / "preprocess" / "mask.png").is_file()
    assert (tmp_path / "run" / "preprocess" / "debug-overlay.png").is_file()
    assert result["pixelize"]["padding"] == 30


def test_run_image_high_threshold_does_not_destroy_output(tmp_path, monkeypatch):
    source = tmp_path / "high-threshold.png"
    image = Image.new("RGBA", (120, 100), (245, 245, 245, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 100, 80), fill=(10, 10, 10, 255))
    image.save(source)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_image_pipeline(
        image_path=source,
        pet_id="high-threshold",
        display_name="High Threshold",
        description="",
        run_dir=tmp_path / "run",
        flutter_output_dir=tmp_path / "flutter",
        remove_bg="auto",
        bg_threshold=200,
    )

    assert result["ok"] is True
    assert result["preprocess"]["subject"]["visible_pixels"] > 0
    with Image.open(tmp_path / "run" / "reference" / "pixel-reference.png") as reference:
        assert reference.getbbox() is not None

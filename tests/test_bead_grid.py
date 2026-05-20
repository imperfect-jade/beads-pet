from pathlib import Path

from PIL import Image, ImageDraw

from hatch_pet_tool.image.bead_grid import GRID_PREVIEW_CELL, _detect_in_orientation, detect_grid, build_grid_reference
from hatch_pet_tool.pipeline.run_image import run_image_pipeline


def _grid_image(
    path: Path,
    *,
    columns: int = 8,
    rows: int = 6,
    cell: int = 10,
    border: int = 1,
) -> None:
    image = Image.new("RGBA", (columns * cell, rows * cell), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    palette = [
        (210, 48, 72, 255),
        (45, 120, 220, 255),
        (240, 210, 62, 255),
        (60, 170, 90, 255),
    ]
    for row in range(rows):
        for column in range(columns):
            x0 = column * cell
            y0 = row * cell
            color = palette[(row + column) % len(palette)]
            draw.rectangle((x0, y0, x0 + cell - 1, y0 + cell - 1), fill=color)
            if border:
                draw.rectangle((x0, y0, x0 + cell - 1, y0 + border - 1), fill=(245, 245, 245, 255))
                draw.rectangle((x0, y0, x0 + border - 1, y0 + cell - 1), fill=(245, 245, 245, 255))
    image.save(path)


def _pixel_subject(path: Path) -> None:
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((14, 12, 50, 48), fill=(230, 80, 90, 255))
    draw.rectangle((24, 24, 28, 28), fill=(20, 20, 20, 255))
    draw.rectangle((38, 24, 42, 28), fill=(20, 20, 20, 255))
    image.save(path)


def test_detect_grid_finds_synthetic_rows_and_columns(tmp_path):
    source = tmp_path / "grid.png"
    _grid_image(source, columns=9, rows=7, cell=12)

    with Image.open(source) as opened:
        fit = detect_grid(opened.convert("RGBA"))

    assert fit is not None
    assert fit.columns == 9
    assert fit.rows == 7
    assert fit.confidence >= 0.62


def test_grid_sampling_uses_center_region_not_border(tmp_path):
    source = tmp_path / "grid.png"
    pixel_reference = tmp_path / "pixel-reference.png"
    grid_sample = tmp_path / "grid-sample.png"
    overlay = tmp_path / "grid-debug-overlay.png"
    _grid_image(source, columns=4, rows=4, cell=14, border=3)

    result = build_grid_reference(
        image_path=source,
        pixel_reference_path=pixel_reference,
        grid_sample_path=grid_sample,
        debug_overlay_path=overlay,
        colors=16,
        padding=10,
    )

    assert result["grid"]["columns"] == 4
    assert result["grid"]["rows"] == 4
    assert pixel_reference.is_file()
    assert grid_sample.is_file()
    assert overlay.is_file()
    with Image.open(grid_sample) as sampled:
        assert sampled.getpixel((GRID_PREVIEW_CELL // 2, GRID_PREVIEW_CELL // 2))[:3] == (210, 48, 72)


def test_rotated_grid_can_be_sampled(tmp_path):
    source = tmp_path / "grid.png"
    rotated = tmp_path / "grid-rotated.png"
    _grid_image(source, columns=8, rows=5, cell=10)
    with Image.open(source) as opened:
        opened.transpose(Image.Transpose.ROTATE_90).save(rotated)

    with Image.open(rotated) as opened:
        fit = detect_grid(opened.convert("RGBA"))

    assert fit is not None
    assert fit.confidence >= 0.62
    assert fit.rotation in {0, 90, 180, 270}
    assert sorted([fit.columns, fit.rows]) == [5, 8]


def test_run_image_auto_uses_grid_for_regular_beads(tmp_path, monkeypatch):
    source = tmp_path / "beads.png"
    _grid_image(source, columns=8, rows=6, cell=10)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_image_pipeline(
        image_path=source,
        pet_id="grid-beads",
        display_name="Grid Beads",
        description="",
        run_dir=tmp_path / "run",
        flutter_output_dir=tmp_path / "flutter",
        remove_bg="none",
        reference_mode="auto",
        colors=16,
    )

    assert result["ok"] is True
    assert result["reference_source"] == "grid"
    assert result["grid"]["columns"] == 8
    assert (tmp_path / "run" / "reference" / "grid-sample.png").is_file()
    assert (tmp_path / "run" / "preprocess" / "grid-debug-overlay.png").is_file()


def test_run_image_auto_falls_back_for_regular_pixel_art(tmp_path, monkeypatch):
    source = tmp_path / "pixel.png"
    _pixel_subject(source)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_image_pipeline(
        image_path=source,
        pet_id="pixel-art",
        display_name="Pixel Art",
        description="",
        run_dir=tmp_path / "run",
        flutter_output_dir=tmp_path / "flutter",
        remove_bg="none",
        reference_mode="auto",
    )

    assert result["ok"] is True
    assert result["reference_source"] == "pixelize"
    assert result["grid_error"]["error_code"] == "GRID_LOW_CONFIDENCE"
    assert (tmp_path / "run" / "reference" / "pixel-reference.png").is_file()


def test_run_image_grid_mode_fails_for_non_grid_pixel_art(tmp_path):
    source = tmp_path / "pixel.png"
    _pixel_subject(source)

    result = run_image_pipeline(
        image_path=source,
        pet_id="pixel-art",
        display_name="Pixel Art",
        description="",
        run_dir=tmp_path / "run",
        flutter_output_dir=tmp_path / "flutter",
        remove_bg="none",
        reference_mode="grid",
    )

    assert result["ok"] is False
    assert result["error_code"] == "GRID_LOW_CONFIDENCE"
    assert (tmp_path / "run" / "qa" / "run-summary.json").is_file()


def test_run_image_pixelize_mode_skips_grid(tmp_path, monkeypatch):
    source = tmp_path / "beads.png"
    _grid_image(source, columns=8, rows=6, cell=10)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_image_pipeline(
        image_path=source,
        pet_id="pixelize-only",
        display_name="Pixelize Only",
        description="",
        run_dir=tmp_path / "run",
        flutter_output_dir=tmp_path / "flutter",
        remove_bg="none",
        reference_mode="pixelize",
    )

    assert result["ok"] is True
    assert result["reference_source"] == "pixelize"
    assert "grid" not in result
    assert not (tmp_path / "run" / "reference" / "grid-sample.png").exists()


def test_run_image_detects_full_template_grid_before_subject_failure(tmp_path, monkeypatch):
    source = tmp_path / "template.png"
    cell = 16
    columns = 12
    rows = 10
    image = Image.new("RGBA", (columns * cell, rows * cell), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    for x in range(0, image.width, cell):
        draw.line((x, 0, x, image.height), fill=(160, 160, 160, 255), width=1)
    for y in range(0, image.height, cell):
        draw.line((0, y, image.width, y), fill=(160, 160, 160, 255), width=1)
    draw.rectangle((3 * cell, 2 * cell, 8 * cell - 1, 7 * cell - 1), fill=(30, 30, 30, 255))
    image.save(source)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_image_pipeline(
        image_path=source,
        pet_id="template-grid",
        display_name="Template Grid",
        description="",
        run_dir=tmp_path / "run",
        flutter_output_dir=tmp_path / "flutter",
        remove_bg="auto",
        reference_mode="auto",
        debug=True,
    )

    assert result["ok"] is True
    assert result["reference_source"] == "grid"
    assert result["grid"]["columns"] == columns
    assert result["grid"]["rows"] == rows
    assert result["grid"]["source"] == "cropped"
    assert (tmp_path / "run" / "preprocess" / "grid-candidates.json").is_file()


def test_color_profile_rejects_bad_cell_aspect_candidate(tmp_path):
    source = tmp_path / "bad-grid.png"
    image = Image.new("RGBA", (140, 120), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    for x in range(0, 140, 4):
        draw.line((x, 0, x, 119), fill=(0, 0, 0, 255), width=1)
    for y in range(0, 120, 30):
        draw.line((0, y, 139, y), fill=(0, 0, 0, 255), width=1)
    image.save(source)

    with Image.open(source) as opened:
        fit = _detect_in_orientation(opened.convert("RGBA"), 0)

    assert fit is not None
    assert fit.reject_reason == "cell_aspect_mismatch"
    assert fit.confidence < 0.48


def test_grid_detection_prefers_zero_rotation_when_confidence_is_close(tmp_path):
    source = tmp_path / "grid.png"
    _grid_image(source, columns=10, rows=10, cell=12)

    with Image.open(source) as opened:
        fit = detect_grid(opened.convert("RGBA"))

    assert fit is not None
    assert fit.rotation == 0

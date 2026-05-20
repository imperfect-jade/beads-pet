from PIL import Image, ImageDraw

from hatch_pet_tool.core.constants import CELL_HEIGHT, CELL_WIDTH
from hatch_pet_tool.image.bead_grid import GRID_PREVIEW_CELL, build_grid_reference
from hatch_pet_tool.image.color_restore import (
    build_base_pixel_pet,
    fit_base_to_cell,
    render_base_to_cell,
    restore_colors,
    restore_template_grid_colors,
)


def _visible_colors(image):
    rgba = image.convert("RGBA")
    data = rgba.get_flattened_data() if hasattr(rgba, "get_flattened_data") else rgba.getdata()
    return {(red, green, blue) for red, green, blue, alpha in data if alpha > 0}


def _pixels(image):
    rgba = image.convert("RGBA")
    return rgba.get_flattened_data() if hasattr(rgba, "get_flattened_data") else rgba.getdata()


def test_restore_colors_caps_palette_and_preserves_transparency():
    image = Image.new("RGBA", (8, 4), (0, 0, 0, 0))
    pixels = image.load()
    for x in range(8):
        pixels[x, 0] = (200 + x, 40, 60, 255)
        pixels[x, 1] = (40, 120 + x, 220, 255)
        pixels[x, 2] = (240, 210, 60 + x, 255)

    restored, info = restore_colors(image, 3)

    assert restored.getpixel((7, 3))[3] == 0
    assert len(_visible_colors(restored)) <= 3
    assert info["output_colors"] <= 3


def test_restore_template_grid_colors_preserves_small_semantic_colors():
    image = Image.new("RGBA", (8, 8), (248, 216, 106, 255))
    pixels = image.load()
    for x in range(8):
        pixels[x, 0] = (2, 2, 2, 255)
    for point in ((4, 4), (4, 5), (5, 4), (5, 5)):
        pixels[point] = (238, 146, 88, 255)

    restored, info = restore_template_grid_colors(image, 2)

    assert restored.getpixel((4, 4))[:3] == (238, 146, 88)
    assert restored.getpixel((0, 0))[:3] == (2, 2, 2)
    assert info["output_colors"] >= 3
    assert info["palette_source"].startswith("grid-template")


def test_build_base_pixel_pet_is_low_resolution_then_cell_is_nearest_centered():
    image = Image.new("RGBA", (400, 300), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 30, 360, 260), fill=(220, 120, 70, 255))
    draw.rectangle((120, 80, 150, 110), fill=(10, 10, 10, 255))

    base, base_info = build_base_pixel_pet(image, colors=8, max_width=80, max_height=70)
    cell, cell_info = fit_base_to_cell(base, padding=10)

    assert base.width <= 80
    assert base.height <= 70
    assert base_info["colors"]["output_colors"] <= 8
    assert cell.size == (CELL_WIDTH, CELL_HEIGHT)
    assert cell_info["offset"][0] >= 0
    assert cell_info["offset"][1] >= 0


def test_soft_pixel_render_outputs_cell_with_clean_transparency():
    base = Image.new("RGBA", (12, 12), (0, 0, 0, 0))
    draw = ImageDraw.Draw(base)
    draw.rectangle((2, 2, 9, 9), fill=(220, 120, 70, 255), outline=(10, 10, 10, 255))

    rendered, info = render_base_to_cell(base, padding=10, render_style="soft-pixel", render_scale=2)

    assert rendered.size == (CELL_WIDTH, CELL_HEIGHT)
    assert rendered.getpixel((0, 0))[3] == 0
    assert info["render_style"] == "soft-pixel"
    alpha_values = {alpha for *_rgb, alpha in _pixels(rendered)}
    assert 0 in alpha_values
    assert any(0 < alpha < 255 for alpha in alpha_values)


def test_pixel_render_keeps_hard_alpha():
    base = Image.new("RGBA", (12, 12), (0, 0, 0, 0))
    draw = ImageDraw.Draw(base)
    draw.rectangle((2, 2, 9, 9), fill=(220, 120, 70, 255), outline=(10, 10, 10, 255))

    rendered, info = render_base_to_cell(base, padding=10, render_style="pixel", render_scale=3)

    assert rendered.size == (CELL_WIDTH, CELL_HEIGHT)
    assert info["render_style"] == "pixel"
    alpha_values = {alpha for *_rgb, alpha in _pixels(rendered)}
    assert alpha_values <= {0, 255}


def test_grid_reference_ignores_center_hole_and_highlight(tmp_path):
    source = tmp_path / "beads.png"
    pixel_reference = tmp_path / "pixel-reference.png"
    base_pixel_pet = tmp_path / "base-pixel-pet.png"
    grid_sample = tmp_path / "grid-sample.png"
    overlay = tmp_path / "grid-debug-overlay.png"
    cell = 20
    image = Image.new("RGBA", (8 * cell, 8 * cell), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for y in range(8):
        for x in range(8):
            left = x * cell
            top = y * cell
            draw.rectangle((left, top, left + cell - 1, top + cell - 1), fill=(214, 132, 72, 255))
            draw.rectangle((left, top, left + cell - 1, top + 1), fill=(245, 245, 245, 255))
            draw.rectangle((left, top, left + 1, top + cell - 1), fill=(245, 245, 245, 255))
            draw.point((left + 9, top + 9), fill=(255, 255, 255, 255))
            draw.point((left + 10, top + 10), fill=(6, 6, 6, 255))
    image.save(source)

    result = build_grid_reference(
        image_path=source,
        pixel_reference_path=pixel_reference,
        base_pixel_pet_path=base_pixel_pet,
        rendered_pixel_pet_path=tmp_path / "rendered-pixel-pet.png",
        grid_sample_path=grid_sample,
        debug_overlay_path=overlay,
        colors=8,
    )

    assert result["grid"]["columns"] >= 8
    with Image.open(base_pixel_pet) as base:
        visible = _visible_colors(base)
        assert (255, 255, 255) not in visible
        assert (6, 6, 6) not in visible
    with Image.open(grid_sample) as sample:
        assert sample.size[0] >= 8 * GRID_PREVIEW_CELL
        assert sample.size[1] >= 8 * GRID_PREVIEW_CELL

from PIL import Image, ImageDraw

from hatch_pet_tool.core.constants import CELL_HEIGHT, CELL_WIDTH
from hatch_pet_tool.image.pixelize import limit_colors, normalize_to_cell, pixelize_image


def _visible_colors(image):
    rgba = image.convert("RGBA")
    data = rgba.get_flattened_data() if hasattr(rgba, "get_flattened_data") else rgba.getdata()
    return {
        (red, green, blue)
        for red, green, blue, alpha in data
        if alpha > 0
    }


def test_normalize_to_cell_outputs_fixed_transparent_cell():
    image = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 8, 20, 24), fill=(240, 80, 90, 255))

    cell, info = normalize_to_cell(image, colors=16)

    assert cell.size == (CELL_WIDTH, CELL_HEIGHT)
    assert cell.getpixel((0, 0))[3] == 0
    assert info["subject_bbox"] == [4, 8, 21, 25]
    assert info["resized_size"][0] > info["subject_size"][0]
    assert info["offset"][0] > 0
    assert info["offset"][1] > 0


def test_normalize_to_cell_uses_nearest_neighbor_without_soft_alpha():
    image = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 299, 149), fill=(255, 0, 0, 255))
    draw.rectangle((0, 150, 299, 299), fill=(0, 0, 255, 255))

    cell, _info = normalize_to_cell(image, colors=16, padding=0)
    data = cell.get_flattened_data() if hasattr(cell, "get_flattened_data") else cell.getdata()
    alpha_values = {alpha for *_rgb, alpha in data}

    assert alpha_values <= {0, 255}
    assert _visible_colors(cell) <= {(255, 0, 0), (0, 0, 255)}


def test_limit_colors_preserves_transparency_and_caps_visible_palette():
    image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    pixels = image.load()
    colors = [
        (255, 0, 0, 255),
        (0, 255, 0, 255),
        (0, 0, 255, 255),
        (255, 255, 0, 255),
    ]
    for index, color in enumerate(colors):
        pixels[index, 0] = color

    limited, info = limit_colors(image, 2)

    assert limited.getpixel((3, 3))[3] == 0
    assert len(_visible_colors(limited)) <= 2
    assert info["quantized"] is True


def test_limit_colors_keeps_saturated_small_details_with_heavy_outline():
    image = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((6, 6, 73, 73), fill=(8, 8, 10, 255))
    for offset in range(24):
        tone = 160 + offset % 16
        draw.rectangle((16 + offset, 16, 17 + offset, 58), fill=(tone, 112 + offset % 12, 62, 255))
    draw.rectangle((30, 30, 38, 38), fill=(245, 42, 58, 255))
    draw.rectangle((44, 30, 52, 38), fill=(42, 135, 235, 255))
    draw.rectangle((30, 44, 38, 52), fill=(246, 190, 54, 255))

    limited, info = limit_colors(image, 8)
    visible = _visible_colors(limited)

    assert info["palette_source"] == "clustered"
    assert any(red > 190 and green < 90 for red, green, _blue in visible)
    assert any(blue > 170 and red < 90 for red, _green, blue in visible)
    assert any(red > 190 and green > 140 and blue < 100 for red, green, blue in visible)


def test_pixelize_writes_palette_preview(tmp_path):
    source = tmp_path / "clean.png"
    output = tmp_path / "pixelized.png"
    preview = tmp_path / "palette-preview.png"
    image = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, 27, 27), fill=(210, 120, 60, 255))
    draw.rectangle((10, 10, 16, 16), fill=(230, 30, 50, 255))
    image.save(source)

    info = pixelize_image(image_path=source, output_path=output, colors=4, palette_preview_path=preview)

    assert preview.is_file()
    assert info["colors"]["palette_preview"] == str(preview)


def test_pixelize_image_writes_centered_cell(tmp_path):
    source = tmp_path / "clean.png"
    output = tmp_path / "pixelized.png"
    image = Image.new("RGBA", (48, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 6, 30, 22), fill=(20, 120, 220, 255))
    image.save(source)

    info = pixelize_image(image_path=source, output_path=output, colors=16)

    assert output.is_file()
    with Image.open(output) as pixelized:
        assert pixelized.size == (CELL_WIDTH, CELL_HEIGHT)
        assert pixelized.getbbox() is not None
    assert info["pixelized_image"] == str(output)

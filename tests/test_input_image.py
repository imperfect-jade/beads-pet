from PIL import Image, ImageDraw

from hatch_pet_tool.image.input_image import (
    apply_crop,
    load_input_image,
    parse_crop,
    preprocess_input_image,
    remove_background,
    resize_longest_edge,
)


def _subject_image(path, *, fmt="PNG", background=(250, 250, 250, 255)):
    image = Image.new("RGBA", (40, 30), background)
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 8, 27, 23), fill=(20, 120, 220, 255))
    if fmt == "JPEG":
        image = image.convert("RGB")
    image.save(path, format=fmt)


def test_load_input_image_supports_png_jpg_and_webp(tmp_path):
    paths = [
        (tmp_path / "sample.png", "PNG"),
        (tmp_path / "sample.jpg", "JPEG"),
        (tmp_path / "sample.webp", "WEBP"),
    ]
    for path, fmt in paths:
        _subject_image(path, fmt=fmt)
        loaded = load_input_image(path)
        assert loaded.mode == "RGBA"
        assert loaded.size == (40, 30)


def test_parse_and_apply_crop_clamps_to_image_bounds():
    image = Image.new("RGBA", (40, 30), (0, 0, 0, 0))
    crop = parse_crop("-5,10,30,40")
    cropped = apply_crop(image, crop)
    assert cropped.size == (25, 20)


def test_remove_background_auto_uses_corner_color():
    image = Image.new("RGBA", (10, 10), (240, 240, 240, 255))
    image.putpixel((5, 5), (10, 20, 30, 255))
    cleaned, info = remove_background(image, "auto")
    assert info["background_rgb"] == [240, 240, 240]
    assert cleaned.getpixel((0, 0))[3] == 0
    assert cleaned.getpixel((5, 5))[3] == 255


def test_remove_background_uses_explicit_hex_color():
    image = Image.new("RGBA", (10, 10), (255, 0, 255, 255))
    image.putpixel((5, 5), (0, 0, 0, 255))
    cleaned, info = remove_background(image, "#FF00FF")
    assert info["mode"] == "#FF00FF"
    assert cleaned.getpixel((0, 0))[3] == 0
    assert cleaned.getpixel((5, 5))[3] == 255


def test_remove_background_none_keeps_pixels_opaque():
    image = Image.new("RGBA", (10, 10), (255, 255, 255, 255))
    cleaned, info = remove_background(image, "none")
    assert info["mode"] == "none"
    assert cleaned.getpixel((0, 0))[3] == 255


def test_resize_longest_edge_limits_large_images():
    image = Image.new("RGBA", (2000, 1000), (0, 0, 0, 0))
    resized = resize_longest_edge(image, 500)
    assert resized.size == (500, 250)


def test_preprocess_input_image_writes_clean_subject(tmp_path):
    source = tmp_path / "source.png"
    output = tmp_path / "clean.png"
    _subject_image(source)

    info = preprocess_input_image(
        image_path=source,
        output_path=output,
        crop="0,0,40,30",
        remove_bg="auto",
        max_side=20,
    )

    assert output.is_file()
    assert info["clean_image"] == str(output)
    assert info["crop"] == [0, 0, 40, 30]
    with Image.open(output) as cleaned:
        assert cleaned.mode == "RGBA"
        assert max(cleaned.size) <= 20

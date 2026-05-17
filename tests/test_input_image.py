from PIL import Image, ImageDraw

from hatch_pet_tool.image.input_image import (
    apply_crop,
    edge_background_color,
    load_input_image,
    parse_crop,
    preprocess_input_image,
    remove_background,
    resize_longest_edge,
)
from hatch_pet_tool.image.subject import component_stats, keep_largest_component


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
    assert info["method"] == "edge-connected"
    assert info["seed_pixels"] > 0
    assert cleaned.getpixel((0, 0))[3] == 0
    assert cleaned.getpixel((5, 5))[3] == 255


def test_edge_background_color_uses_more_than_four_corners():
    image = Image.new("RGBA", (20, 20), (240, 240, 240, 255))
    pixels = image.load()
    for point in ((0, 0), (19, 0), (0, 19), (19, 19)):
        pixels[point] = (20, 20, 20, 255)

    assert edge_background_color(image) == (240, 240, 240)


def test_remove_background_uses_explicit_hex_color():
    image = Image.new("RGBA", (10, 10), (255, 0, 255, 255))
    image.putpixel((5, 5), (0, 0, 0, 255))
    cleaned, info = remove_background(image, "#FF00FF")
    assert info["mode"] == "#FF00FF"
    assert cleaned.getpixel((0, 0))[3] == 0
    assert cleaned.getpixel((5, 5))[3] == 255


def test_remove_background_threshold_controls_tolerance():
    image = Image.new("RGBA", (10, 10), (250, 250, 250, 255))
    image.putpixel((0, 5), (230, 230, 230, 255))
    image.putpixel((5, 5), (230, 230, 230, 255))

    strict, _strict_info = remove_background(image, "#FFFFFF", threshold=10)
    loose, _loose_info = remove_background(image, "#FFFFFF", threshold=50)

    assert strict.getpixel((0, 5))[3] == 255
    assert loose.getpixel((0, 5))[3] == 0
    assert strict.getpixel((5, 5))[3] == 255
    assert loose.getpixel((5, 5))[3] == 0


def test_remove_background_preserves_internal_white_subject_region():
    image = Image.new("RGBA", (20, 20), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 5, 14, 14), outline=(0, 0, 0, 255), width=1)
    draw.rectangle((6, 6, 13, 13), fill=(255, 255, 255, 255))

    cleaned, info = remove_background(image, "auto", threshold=10)

    assert info["method"] == "edge-connected"
    assert cleaned.getpixel((0, 0))[3] == 0
    assert cleaned.getpixel((10, 10))[3] == 255
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
    source_out = tmp_path / "source-out.png"
    cropped_out = tmp_path / "cropped.png"
    background_out = tmp_path / "background-removed.png"
    mask_out = tmp_path / "mask.png"
    overlay_out = tmp_path / "debug-overlay.png"
    _subject_image(source)

    info = preprocess_input_image(
        image_path=source,
        output_path=output,
        crop="0,0,40,30",
        remove_bg="auto",
        max_side=20,
        source_output_path=source_out,
        cropped_output_path=cropped_out,
        background_removed_output_path=background_out,
        mask_output_path=mask_out,
        debug_overlay_output_path=overlay_out,
        debug=True,
    )

    assert output.is_file()
    assert source_out.is_file()
    assert cropped_out.is_file()
    assert background_out.is_file()
    assert mask_out.is_file()
    assert overlay_out.is_file()
    assert info["clean_image"] == str(output)
    assert info["background_removed_image"] == str(background_out)
    assert info["mask_image"] == str(mask_out)
    assert info["crop"] == [0, 0, 40, 30]
    with Image.open(output) as cleaned:
        assert cleaned.mode == "RGBA"
        assert max(cleaned.size) <= 20


def test_preprocess_keeps_largest_connected_subject(tmp_path):
    source = tmp_path / "source.png"
    output = tmp_path / "clean.png"
    image = Image.new("RGBA", (60, 30), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 8, 15, 18), fill=(255, 0, 0, 255))
    draw.rectangle((30, 5, 55, 25), fill=(0, 0, 255, 255))
    image.save(source)

    info = preprocess_input_image(
        image_path=source,
        output_path=output,
        remove_bg="auto",
    )

    assert info["largest_component"]["removed_components"] == 1
    with Image.open(output) as cleaned:
        assert cleaned.size == (26, 21)


def test_keep_largest_component_uses_largest_seed_not_bbox_guess():
    image = Image.new("RGBA", (120, 90), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 22, 22), fill=(40, 40, 40, 255))
    draw.rectangle((35, 35, 115, 85), fill=(220, 120, 70, 255))

    components = component_stats(image)
    kept, info = keep_largest_component(image)

    assert components[0]["seed"] == [35, 35]
    assert info["kept_component"]["seed"] == [35, 35]
    assert kept.getpixel((40, 40))[3] == 255
    assert kept.getpixel((10, 10))[3] == 0


def test_preprocess_dog_sleep_style_keeps_large_body_instead_of_z(tmp_path):
    source = tmp_path / "dog-style.png"
    output = tmp_path / "clean.png"
    image = Image.new("RGBA", (180, 140), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.line((18, 15, 38, 15, 18, 35, 38, 35), fill=(30, 30, 30, 255), width=5)
    draw.line((55, 45, 82, 45, 55, 72, 82, 72), fill=(30, 30, 30, 255), width=6)
    draw.rectangle((95, 70, 168, 124), fill=(180, 100, 65, 255))
    draw.rectangle((112, 86, 142, 108), fill=(245, 235, 220, 255))
    image.save(source)

    info = preprocess_input_image(
        image_path=source,
        output_path=output,
        remove_bg="auto",
        bg_threshold=20,
    )

    assert info["largest_component"]["removed_components"] == 2
    assert info["largest_component"]["kept_component"]["bbox"] == [95, 70, 169, 125]
    with Image.open(output) as cleaned:
        assert cleaned.size == (74, 55)

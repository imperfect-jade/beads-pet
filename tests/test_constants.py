from hatch_pet_tool.core.constants import (
    ANIMATION_ROWS,
    ATLAS_HEIGHT,
    ATLAS_WIDTH,
    CELL_HEIGHT,
    CELL_WIDTH,
    COLUMNS,
    ROWS,
)


def test_atlas_dimensions_are_hatch_pet_contract():
    assert COLUMNS == 8
    assert ROWS == 9
    assert CELL_WIDTH == 192
    assert CELL_HEIGHT == 208
    assert ATLAS_WIDTH == 1536
    assert ATLAS_HEIGHT == 1872


def test_animation_rows_match_expected_order():
    assert [(row.state, row.row, row.frames) for row in ANIMATION_ROWS] == [
        ("idle", 0, 6),
        ("running-right", 1, 8),
        ("running-left", 2, 8),
        ("waving", 3, 4),
        ("jumping", 4, 5),
        ("failed", 5, 8),
        ("waiting", 6, 6),
        ("running", 7, 6),
        ("review", 8, 6),
    ]

import subprocess
import sys


def test_legacy_validate_wrapper_shows_help():
    result = subprocess.run(
        [sys.executable, "scripts/validate_atlas.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Validate a Codex pet spritesheet atlas" in result.stdout

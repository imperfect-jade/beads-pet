"""Unified command line entry point for hatch-pet-tool."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence

COMMANDS = {
    "prepare": "hatch_pet_tool.pipeline.prepare",
    "status": "hatch_pet_tool.pipeline.status",
    "record": "hatch_pet_tool.pipeline.record",
    "finalize": "hatch_pet_tool.pipeline.finalize",
    "export-flutter": "hatch_pet_tool.flutter.export",
    "repair": "hatch_pet_tool.pipeline.repair",
    "mirror-left": "hatch_pet_tool.pipeline.mirror_left",
    "validate": "hatch_pet_tool.image.validate",
    "contact-sheet": "hatch_pet_tool.image.contact_sheet",
    "render-videos": "hatch_pet_tool.image.videos",
    "compose-atlas": "hatch_pet_tool.image.compose",
    "extract-frames": "hatch_pet_tool.image.extract",
    "inspect-frames": "hatch_pet_tool.image.inspect",
    "generate-images": "hatch_pet_tool.generation.openai_images",
    "package-codex": "hatch_pet_tool.pipeline.package_codex",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hatch-pet-tool",
        description="Generate and export hatch-pet style Flutter sprite assets.",
    )
    parser.add_argument("command", nargs="?", choices=sorted(COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args:
        _build_parser().print_help()
        return
    if raw_args[0] in {"-h", "--help"}:
        _build_parser().print_help()
        return

    command = raw_args[0]
    if command not in COMMANDS:
        _build_parser().error(f"invalid command: {command}")

    module = importlib.import_module(COMMANDS[command])
    sys.argv = [f"hatch-pet-tool {command}", *raw_args[1:]]
    module.main()


if __name__ == "__main__":
    main()

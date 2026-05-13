# Hatch Pet Tool

This repository is now a regular Python tool project for building hatch-pet style
pixel pet assets for a Flutter Todolist app. The original Codex skill files are
kept in `docs/legacy-skill/` as reference material only.

## Install

```powershell
python -m pip install -e .[dev]
```

## Common Commands

```powershell
hatch-pet-tool run-image --image .\input\beads.png --pet-id beads-cat --display-name "Beads Cat"
hatch-pet-tool run-image --image .\input\beads.jpg --remove-bg auto
hatch-pet-tool run-image --image .\input\beads.webp --crop 120,80,640,640 --remove-bg "#FFFFFF"
hatch-pet-tool run-image --image .\input\beads.png --remove-bg none --max-input-side 768
hatch-pet-tool prepare --pet-name Calico --reference .\input\cat.png
hatch-pet-tool status --run-dir .\output\hatch-pet\calico-...
hatch-pet-tool record --run-dir .\output\hatch-pet\calico-... --job-id base --source <imagegen-output.png>
hatch-pet-tool finalize --run-dir .\output\hatch-pet\calico-... --skip-package
hatch-pet-tool export-flutter --run-dir .\output\hatch-pet\calico-...
```

`run-image` is the default no-AI MVP path. It takes one image, creates
algorithmic placeholder animation frames, validates the hatch-pet atlas, and
exports Flutter assets. The animation is intentionally simple in this first
version; later iterations can improve bead-grid detection, palette recovery, and
motion quality.

The `run-image` input preprocessor reads PNG/JPG/WebP, converts to RGBA, can crop
with `--crop x,y,w,h`, scales large images by longest edge, and removes simple
backgrounds with `--remove-bg auto` or an explicit `#RRGGBB`. The first version
uses four-corner color sampling, so it works best with solid or near-solid
backgrounds.

`export-flutter` writes to `output/flutter-assets/<pet_id>/` by default. It does
not write into the Todolist project. Treat
`D:\Trae_File\flutter_file\todolist` as a read-only integration reference unless
the user explicitly grants write permission.

`generate-images` is legacy/optional. It is not part of the default no-AI main
flow and requires explicit API credentials if used.

## Output Contract

The generated sprite sheet remains hatch-pet compatible:

- atlas size: `1536x1872`
- grid: 8 columns x 9 rows
- cell size: `192x208`
- transparent background

Flutter exports use a Todolist-compatible manifest:

```json
{
  "id": "pet-id",
  "displayName": "Pet Name",
  "description": "Short description.",
  "image": "pet-id_hatch_spritesheet.webp",
  "frameWidth": 192,
  "frameHeight": 208,
  "columns": 8,
  "rows": 9,
  "actions": {}
}
```

## Project Layout

- `src/hatch_pet_tool/`: installable Python package and CLI.
- `src/hatch_pet_tool/pipeline/`: run preparation, status, record, finalize,
  repair, mirroring, and optional Codex packaging.
- `src/hatch_pet_tool/image/`: frame extraction, atlas composition, validation,
  contact sheets, and preview videos.
- `src/hatch_pet_tool/flutter/`: Flutter asset export.
- `src/hatch_pet_tool/image/algorithmic.py`: no-AI placeholder frame generation
  from one input image.
- `docs/reference/`: sprite and QA contracts.
- `docs/legacy-skill/`: archived Codex skill metadata.
- `scripts/`: backward-compatible thin wrappers for old script commands.

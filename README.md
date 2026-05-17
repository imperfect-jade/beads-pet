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
hatch-pet-tool run-beads --images .\input\a.jpg .\input\b.webp .\input\c.png --pet-id beads-cat
hatch-pet-tool run-image --image .\input\beads.jpg --remove-bg auto
hatch-pet-tool run-image --image .\input\beads.webp --crop 120,80,640,640 --remove-bg "#FFFFFF"
hatch-pet-tool run-image --image .\input\beads.png --remove-bg none --max-input-side 768 --colors 16 --debug
hatch-pet-tool run-image --image .\input\beads.jpg --bg-threshold 48 --subject-padding 18
hatch-pet-tool prepare --pet-name Calico --reference .\input\cat.png
hatch-pet-tool status --run-dir .\output\hatch-pet\calico-...
hatch-pet-tool record --run-dir .\output\hatch-pet\calico-... --job-id base --source <imagegen-output.png>
hatch-pet-tool finalize --run-dir .\output\hatch-pet\calico-... --skip-package
hatch-pet-tool export-flutter --run-dir .\output\hatch-pet\calico-...
```

`run-image` is the default single-image no-AI MVP entrypoint. It expects an input image that is
already pixel-art or bead-art styled. The tool cleans the input, extracts the
visible pixel/bead subject, normalizes it into a transparent `192x208` hatch-pet
cell, creates algorithmic placeholder animation frames, validates the hatch-pet
atlas, and exports Flutter assets. The animation is intentionally simple in this
first version; later iterations can improve bead-grid detection, palette
recovery, and motion quality.

The `run-image` input preprocessor reads PNG/JPG/WebP, converts to RGBA, can crop
with `--crop x,y,w,h`, scales large images by longest edge, and removes simple
backgrounds with `--remove-bg auto` or an explicit `#RRGGBB`. Auto background
removal samples the image edges, uses `--bg-threshold` as the color tolerance,
and keeps the largest connected subject after removal. Use `--remove-bg none`
when the image already has a transparent background or the automatic cleanup is
too aggressive.

Each run writes explainable intermediate files:

- `input/source-00.png`
- `preprocess/cropped.png`
- `preprocess/background-removed.png`
- `reference/pixel-reference.png`
- `qa/contact-sheet.png`
- `final/spritesheet.webp`

When `--debug` is enabled, the run also writes `preprocess/mask.png` and
`preprocess/debug-overlay.png`. The pixel reference step is not a general
photo-to-pixel-art filter: it uses the transparent alpha mask to crop the subject,
preserves hard pixel/bead edges with nearest-neighbor scaling, optionally caps
visible colors with `--colors`, and centers the result in the hatch-pet cell.
Use `--subject-padding` to control the margin inside the `192x208` reference.
The first version does not perform perspective correction, grid-line detection,
or round bead reconstruction from real-world photos.

`run-beads` is the multi-image entrypoint. In this first version it does not
fuse images; it preprocesses each candidate, scores the visible subject, chooses
one `primary_image`, then runs the same `run-image` pipeline.

Both commands write `qa/run-summary.json`. Failed runs write `ok: false` with a
stable `error_code`, readable message, and suggested fix such as using manual
`--crop` or a cleaner background.

Flutter export defaults remain unchanged: generated Flutter assets are written to
`output/flutter-assets/<pet_id>/` unless `--flutter-output-dir` is provided. The
tool does not write into the Todolist project. Treat
`D:\Trae_File\flutter_file\todolist` as a read-only integration reference unless
the user explicitly grants write permission.

`generate-images` is legacy/optional. It is not part of the default no-AI main
flow and requires explicit API credentials if used.

## Local Samples

Real bead photos should stay local. Put them under `samples/local/` or
`output/samples/`; both generated outputs and local samples are ignored by Git.
Use `samples/manifest.example.json` as a template, then batch-check a local
manifest with:

```powershell
python scripts\run_sample_manifest.py --manifest samples\manifest.local.json --force
```

The first sample baseline should cover 10-20 images across white backgrounds,
tabletop backgrounds, tilted photos, strong shadows, multiple bead objects,
partial crops, and PNG/JPG/WebP formats.

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
  repair, mirroring, `run-image`, `run-beads`, and optional Codex packaging.
- `src/hatch_pet_tool/image/`: frame extraction, atlas composition, validation,
  contact sheets, and preview videos.
- `src/hatch_pet_tool/flutter/`: Flutter asset export.
- `src/hatch_pet_tool/image/algorithmic.py`: no-AI placeholder frame generation
  from one input image.
- `src/hatch_pet_tool/image/input_image.py`: source image loading, cropping,
  resizing, and simple background removal.
- `src/hatch_pet_tool/image/pixelize.py`: pixel/bead subject extraction and
  normalization into a hatch-pet cell.
- `docs/reference/`: sprite and QA contracts.
- `docs/legacy-skill/`: archived Codex skill metadata.
- `scripts/`: backward-compatible thin wrappers for old script commands.

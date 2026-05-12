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
hatch-pet-tool prepare --pet-name Calico --reference .\input\cat.png
hatch-pet-tool status --run-dir .\output\hatch-pet\calico-...
hatch-pet-tool record --run-dir .\output\hatch-pet\calico-... --job-id base --source <imagegen-output.png>
hatch-pet-tool finalize --run-dir .\output\hatch-pet\calico-... --skip-package
hatch-pet-tool export-flutter --run-dir .\output\hatch-pet\calico-...
```

`export-flutter` writes to `output/flutter-assets/<pet_id>/` by default. It does
not write into the Todolist project. Treat
`D:\Trae_File\flutter_file\todolist` as a read-only integration reference unless
the user explicitly grants write permission.

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
- `docs/reference/`: sprite and QA contracts.
- `docs/legacy-skill/`: archived Codex skill metadata.
- `scripts/`: backward-compatible thin wrappers for old script commands.

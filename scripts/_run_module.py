from __future__ import annotations

import runpy
import sys
from pathlib import Path


def run(module: str) -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    runpy.run_module(module, run_name="__main__", alter_sys=True)

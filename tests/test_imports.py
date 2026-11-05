"""The dependency rule (docs/design.md, section 6): the package imports numpy, scipy, pandas and the standard library
only, and nothing from the research or data packages that use it. Each module is read for its import statements, and
the whole package is imported in a fresh interpreter, so a stray import anywhere fails the suite."""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PKG = SRC / "gaspaths"
FORBIDDEN = {"meritorder", "gbpowerdata", "gbabm", "spikeabm", "cmmodel"}
ALLOWED_THIRD_PARTY = {"numpy", "scipy", "pandas"}


def _imported_tops(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            tops.add(node.module.split(".")[0])
    return tops


def test_modules_import_only_the_allowed_packages():
    allowed = set(sys.stdlib_module_names) | ALLOWED_THIRD_PARTY | {"gaspaths", "__future__"}
    bad = {}
    for f in sorted(PKG.rglob("*.py")):
        extra = _imported_tops(f) - allowed
        if extra:
            bad[f.relative_to(ROOT).as_posix()] = sorted(extra)
    assert not bad, f"imports outside numpy, scipy, pandas and the standard library: {bad}"


def test_package_imports_in_a_fresh_interpreter_without_the_research_packages():
    code = ("import importlib, pkgutil, sys, gaspaths\n"
            "for m in pkgutil.walk_packages(gaspaths.__path__, 'gaspaths.'):\n"
            "    importlib.import_module(m.name)\n"
            "print('LOADED', ' '.join(sorted(sys.modules)))\n")
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    loaded = {m.split(".")[0] for m in r.stdout.split("LOADED", 1)[1].split()}
    assert not (loaded & FORBIDDEN), f"gaspaths loads {sorted(loaded & FORBIDDEN)}"

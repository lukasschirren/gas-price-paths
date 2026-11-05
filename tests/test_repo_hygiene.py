"""Repository hygiene (docs/design.md, section 6): no data beyond small synthetic fixtures, nothing named after a
licensed market-data source, no token or credential file, no file over 5 MB, and no global random state in the
package. The file checks cover what git tracks and what it would track (untracked files not ignored), so they bite
before a commit; they skip outside a git work tree."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_SUFFIXES = {".csv", ".parquet", ".xlsx", ".xls", ".feather", ".h5", ".nc", ".pkl", ".npz"}
LICENSED = re.compile(r"nordpool|n2ex|epex|dayahead", re.I)
SECRET = re.compile(r"token|credential", re.I)
MAX_BYTES = 5 * 1024 * 1024
GLOBAL_RNG = re.compile(r"np\.random\.(seed|rand|randn|random|normal|uniform|standard_normal|choice|randint|"
                        r"multivariate_normal|permutation|shuffle)\s*\(")


def _files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git work tree")
    return [ROOT / f for f in out.splitlines() if f]


def test_no_data_files_outside_the_fixtures():
    bad = [f for f in _files() if f.suffix.lower() in DATA_SUFFIXES and "fixtures" not in f.parts]
    assert not bad, f"data files would be tracked: {bad}"


def test_nothing_licensed_or_secret_by_name():
    bad = [f for f in _files() if LICENSED.search(f.name) or SECRET.search(f.name)]
    assert not bad, f"licensed-data or credential names: {bad}"


def test_no_file_over_5_mb():
    bad = [(f, f.stat().st_size) for f in _files() if f.exists() and f.stat().st_size > MAX_BYTES]
    assert not bad, f"files over 5 MB: {bad}"


def test_no_global_random_state_in_the_package():
    bad = {}
    for f in sorted((ROOT / "src" / "gaspaths").rglob("*.py")):
        hits = GLOBAL_RNG.findall(f.read_text(encoding="utf-8"))
        if hits:
            bad[f.name] = hits
    assert not bad, f"global random state (pass a numpy Generator instead): {bad}"

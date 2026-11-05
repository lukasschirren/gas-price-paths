"""The example runs on a synthetic daily series written to a temporary CSV file (never into the repository)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_example_fits_a_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 900
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.995 * x[i - 1] + 0.03 * rng.standard_normal()
    d = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n), "price": 80.0 * np.exp(x)})
    d = d.drop(index=[100, 101])                         # a gap the example must turn into missing days
    f = tmp_path / "series.csv"
    d.to_csv(f, index=False)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run([sys.executable, str(ROOT / "examples" / "fit_daily_series.py"), "--csv", str(f), "--starts",
                        "2"], cwd=ROOT, capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "898 observations" in r.stdout and "half-lives" in r.stdout

"""Opt-in live Databricks test.

Run with:  pytest -m databricks
It needs ~/.config/ohdsi/databricks_connection.yml, circepy and
databricks-sql-connector. It is skipped everywhere else, including CI.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.databricks

ROOT = Path(__file__).resolve().parents[2]


def test_live_databricks_small_fit(tmp_path):
    card = Path(os.path.expanduser("~/.config/ohdsi/databricks_connection.yml"))
    if not card.exists():
        pytest.skip("no Databricks connection card")
    pytest.importorskip("circe")
    pytest.importorskip("databricks.sql")

    spec = importlib.util.spec_from_file_location(
        "fit_databricks", ROOT / "scripts" / "fit_databricks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    out = tmp_path / "fit.pkl"
    summary = tmp_path / "fit.json"
    rc = mod.main(["--n-patients", "100", "--k", "2",
                   "--out", str(out), "--summary", str(summary)])
    assert rc == 0
    data = json.loads(summary.read_text())
    assert data["patients"] > 0
    assert data["records"] > 0
    assert data["patients"] <= 100

#!/usr/bin/env bash
# Bootstrap topocdm on a fresh work laptop (macOS/Linux).
#
#   bash scripts/setup_work_laptop.sh
#
# What it does:
#   1. checks Python >= 3.10 and git
#   2. creates .venv (if absent) and installs topocdm with dev tools
#   3. installs the Databricks SQL connector
#   4. installs circepy from PR 60 (feature/decomposable-cohorts),
#      pinned to the PR head commit, with a branch fallback
#   5. builds the optional C kernel if a compiler is present
#   6. runs the test suite
#   7. prints the translated Databricks SQL for a 10k patient sample,
#      without connecting (proves the SQLRender path works)
#
# Optional environment:
#   PYTHON=python3.11      interpreter to build the venv from
#   VENV=/path/.venv       venv location (default <repo>/.venv)
#   SKIP_CIRCE=1           skip the circepy install and the dry-run
#   SKIP_TESTS=1           skip pytest
#   CDM_SCHEMA=main.omop   schema printed in the dry-run
#
# No Docker, no Postgres, no Synthea data, and no R are required for this path.
# The Databricks connection is the primary interface; the local OHDSI lab is
# optional.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO_DIR/.venv}"
PYTHON="${PYTHON:-python3}"
CIRCE_REPO="https://github.com/OHDSI/Circepy.git"
# PR 60 head at the time of writing (feat: decomposable cohort sets and shared
# checkpoint execution). Pinning the commit keeps the setup reproducible; set
# CIRCE_SHA to a newer commit when the PR moves.
CIRCE_SHA="${CIRCE_SHA:-7c32a12485fc788be9882c9ea9c7cf1cd89255d6}"
CIRCE_BRANCH="${CIRCE_BRANCH:-feature/decomposable-cohorts}"

say() { printf "\n== %s ==\n" "$1"; }

say "checking prerequisites"
command -v git >/dev/null 2>&1 || {
  echo "git is required: install it and re-run" >&2
  exit 1
}
command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "$PYTHON not found; install Python 3.10+ or set PYTHON=" >&2
  exit 1
}
"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        f"Python >= 3.10 required, found {sys.version.split()[0]}")
print("python", sys.version.split()[0])
PY

say "creating the virtual environment"
if [ ! -d "$VENV" ]; then
  "$PYTHON" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools

say "installing topocdm (editable, with dev tools)"
pip install -e "$REPO_DIR[dev]"

say "installing the Databricks SQL connector"
pip install "databricks-sql-connector[pyarrow]"

if [ "${SKIP_CIRCE:-0}" = "1" ]; then
  say "skipping circepy (SKIP_CIRCE=1)"
else
  say "installing circepy from PR 60"
  if ! pip install "ohdsi-circepy @ git+$CIRCE_REPO@$CIRCE_SHA"; then
    echo "pinned commit failed; trying branch $CIRCE_BRANCH" >&2
    pip install "ohdsi-circepy @ git+$CIRCE_REPO@$CIRCE_BRANCH"
  fi
fi

say "building the optional C kernel"
if command -v clang >/dev/null 2>&1; then
  CC=clang bash "$REPO_DIR/scripts/build_fast.sh" || \
    echo "C kernel build failed; the Python fallback will be used"
elif command -v gcc >/dev/null 2>&1; then
  CC=gcc bash "$REPO_DIR/scripts/build_fast.sh" || \
    echo "C kernel build failed; the Python fallback will be used"
else
  echo "no C compiler found; the Python fallback will be used"
fi

mkdir -p "$REPO_DIR/.mplconfig"

if [ "${SKIP_TESTS:-0}" = "1" ]; then
  say "skipping tests (SKIP_TESTS=1)"
else
  say "running the test suite"
  MPLCONFIGDIR="$REPO_DIR/.mplconfig" python -m pytest -q
fi

if [ "${SKIP_CIRCE:-0}" != "1" ]; then
  say "checking the Databricks SQL translation (no connection)"
  CDM="${CDM_SCHEMA:-main.omop}"
  if python "$REPO_DIR/scripts/fit_databricks.py" --cdm "$CDM" \
      --n-patients 10000 --dry-run >/dev/null; then
    echo "SQLRender dry-run OK: the 10k-patient Spark SQL was produced"
  else
    echo "SQLRender dry-run FAILED: run it without >/dev/null to see the error" >&2
    exit 1
  fi
fi

say "Databricks credentials"
missing=0
for v in DATABRICKS_SERVER_HOSTNAME DATABRICKS_HTTP_PATH DATABRICKS_TOKEN; do
  if [ -z "${!v:-}" ]; then
    echo "  $v is not set"
    missing=1
  fi
done
if [ "$missing" = 0 ]; then
  echo "  all three are set"
else
  echo "  set them before the first real fit (see docs/SETUP.md)"
fi

cat <<EOF

Setup complete.

Next:
  1. export DATABRICKS_SERVER_HOSTNAME=...
     export DATABRICKS_HTTP_PATH=...
     export DATABRICKS_TOKEN=...
  2. first fit, 10k patients, selected on the cluster:
       MPLCONFIGDIR=./.mplconfig .venv/bin/python scripts/fit_databricks.py \\
         --cdm <catalog.schema> --n-patients 10000 --k 20
  3. increase the sample: 20000, 50000, with the same --seed. The samples are
     nested, so 10k is a subset of 20k.

Docs: docs/SETUP.md and docs/DATABRICKS.md
EOF

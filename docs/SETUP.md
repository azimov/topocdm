# Work-laptop setup

Get `topocdm` running on a fresh machine and connect it to a Databricks-hosted
OMOP CDM. This path needs **no Docker, no local Postgres, no Synthea data and
no R**. The Databricks SQL warehouse is the primary interface; the local OHDSI
lab is optional.

## One command (macOS / Linux)

```bash
bash scripts/setup_work_laptop.sh
```

It checks Python and git, creates `.venv`, installs the project with dev tools,
installs the Databricks connector, installs circepy from PR 60, builds the
optional C kernel, runs the 28 tests, and prints the Spark SQL for a 10k
patient sample without connecting.

Useful environment switches:

```bash
PYTHON=python3.11 bash scripts/setup_work_laptop.sh
SKIP_CIRCE=1 bash scripts/setup_work_laptop.sh    # offline / no PR 60 access
SKIP_TESTS=1 bash scripts/setup_work_laptop.sh
VENV=/tmp/topocdm-venv bash scripts/setup_work_laptop.sh
```

## Prerequisites

| need | why |
|---|---|
| Python 3.10+ (3.11 recommended) | the project and circepy both require it |
| git | circepy is installed from a PR branch/commit |
| a C compiler (optional) | builds the fast kernel; the Python fallback works without it |
| Databricks SQL warehouse access | host, HTTP path and a personal access token |

You do **not** need Java, Docker, `psql`, R, or the Synthea extract for this
path.

## Manual install (if you prefer not to run the script)

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
pip install -e ".[dev]"
pip install "databricks-sql-connector[pyarrow]"
pip install "ohdsi-circepy @ git+https://github.com/OHDSI/Circepy.git@7c32a12485fc788be9882c9ea9c7cf1cd89255d6"
```

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip wheel setuptools
pip install -e ".[dev]"
pip install "databricks-sql-connector[pyarrow]"
pip install "ohdsi-circepy @ git+https://github.com/OHDSI/Circepy.git@7c32a12485fc788be9882c9ea9c7cf1cd89255d6"
```

The commit above is the head of PR 60 (`feature/decomposable-cohorts`). If the
PR moves, use the new head commit or the branch name as the `@...` suffix. The
branch is also a fallback in the setup script.

### Optional: the C kernel

```bash
bash scripts/build_fast.sh
```

It needs `clang` or `gcc`. If it fails, nothing breaks; the pure-Python kernel
is used. `topocdm/_fast.so` is gitignored.

## Databricks credentials

The script and `scripts/fit_databricks.py` read the standard environment
variables:

```bash
export DATABRICKS_SERVER_HOSTNAME=adb-....azuredatabricks.net
export DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/....
export DATABRICKS_TOKEN=...            # never commit this
```

The CDM location is `--cdm catalog.schema`, for example `main.omop`. Do not put
the token in the repo; `.secrets/` is gitignored but environment variables are
the intended route.

## Verify the install

```bash
MPLCONFIGDIR=./.mplconfig .venv/bin/python -m pytest -q          # 28 passed
.venv/bin/python scripts/fit_databricks.py \
  --cdm main.omop --n-patients 10000 --dry-run | head
```

The dry run prints the translated Spark SQL: a `with keep as materialized (...)`
CTE that selects 10k patients on the cluster, and a
`join keep using (person_id)` in every domain branch. It does not connect.

## First fit

```bash
MPLCONFIGDIR=./.mplconfig .venv/bin/python scripts/fit_databricks.py \
  --cdm <catalog.schema> --n-patients 10000 --k 20
```

Outputs `results/databricks_fit.pkl` (the fitted `PerDomainFit`) and
`results/databricks_fit.json` (summary: patients, records, k, propensity mean,
death fraction, parameter count, K, atoms).

## Increase the sample gradually

The sample is chosen on the cluster and is deterministic and nested:

```bash
--n-patients 10000 --seed 0
--n-patients 20000 --seed 0     # contains the 10k
--n-patients 50000 --seed 0     # contains the 20k
```

The cost is one sort of the person table. The event tables are joined to the
sample on the cluster and only the sampled rows are downloaded. Do not combine
`--n-patients` with `--sample-pct`; the latter is a rough percentage filter.

## Troubleshooting

**`pip install ohdsi-circepy @ git+...` fails.** Check git and network access
to GitHub. Try the branch instead of the commit:
`@feature/decomposable-cohorts`. If a cached wheel is stale, add
`--no-cache-dir`.

**`databricks-sql-connector` installs but Arrow is missing.** Install the
extra: `pip install "databricks-sql-connector[pyarrow]"`. Without it the script
falls back to `fetchall` (slower, still correct).

**The C kernel build fails.** Harmless. The pure-Python kernel is used; the
tests pass either way.

**Matplotlib complains about a cache.** Set
`MPLCONFIGDIR=./.mplconfig` for every command, as the examples do.

**I have no Docker/Postgres.** That is fine. The Databricks path does not use
them. The local OHDSI lab is only needed for the Synthea/MIMIC fixture
experiments, which are not part of this setup.

**Windows.** The setup script is bash. Use the PowerShell manual steps above;
the Python commands are otherwise identical.

## What lives where

| path | contents |
|---|---|
| `scripts/setup_work_laptop.sh` | this setup |
| `scripts/fit_databricks.py` | Databricks connection, cluster-side sampling, local fit |
| `docs/DATABRICKS.md` | connection and sampling details |
| `docs/FIDELITY_REPORT.md` | prevalence/incidence fidelity, if you run the fixture experiments |
| `results/` | fit pickle and JSON summary |
| `.secrets/` | local credentials, gitignored; not needed if you use env vars |

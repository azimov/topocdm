# Install

Full details, including the Databricks connection card and troubleshooting, are
in [docs/SETUP.md](docs/SETUP.md). This is the short version.

## One command (macOS / Linux)

```bash
bash scripts/setup_work_laptop.sh
```

It creates `.venv`, installs the dev and Databricks extras, installs circepy
from PR 60, builds the optional C kernel, and runs the tests.

## Manual (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
pip install -e ".[dev,databricks]"
pip install "ohdsi-circepy @ git+https://github.com/OHDSI/Circepy.git@7c32a12485fc788be9882c9ea9c7cf1cd89255d6"
```

## Manual (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip wheel setuptools
pip install -e ".[dev,databricks]"
pip install "ohdsi-circepy @ git+https://github.com/OHDSI/Circepy.git@7c32a12485fc788be9882c9ea9c7cf1cd89255d6"
```

## Connection card

Create `~/.config/ohdsi/databricks_connection.yml`:

```yaml
databricks:
  connection:
    server_hostname: "adb-....azuredatabricks.net"
    http_path: "/sql/1.0/warehouses/...."
    personal_access_token: "dapi...."
cmds:
  mycdm_version:
    cdm_schema: "catalog.schema"
    vocabulary_schema: "catalog.schema"
    results_schema: "scratch.schema"   # writable; the sample table goes here
```

## Verify

```bash
MPLCONFIGDIR=./.mplconfig .venv/bin/python -m pytest -q
.venv/bin/python scripts/fit_databricks.py --cmd mycdm_version --dry-run
```

## First fit

```bash
MPLCONFIGDIR=./.mplconfig .venv/bin/python scripts/fit_databricks.py \
  --cmd mycdm_version --n-patients 10000 --k 20
```

Increase the sample with the same seed: `--n-patients 20000`, `50000`. The
samples are nested, so 10k is a subset of 20k.

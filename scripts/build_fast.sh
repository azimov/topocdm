#!/usr/bin/env bash
# Build the optional C likelihood kernel in place.
#
# This wraps `python setup.py build_ext --inplace`, so the same command works on
# macOS, Linux and Windows (Git Bash/WSL). The extension is optional: if no
# compiler is available the build is skipped and the pure-Python kernel is used.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python3}"
"$PYTHON" setup.py build_ext --inplace
echo "C kernel built in topocdm/ (optional; the Python kernel works without it)"

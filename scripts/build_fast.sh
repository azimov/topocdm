#!/usr/bin/env bash
# Build the C likelihood kernel. Falls back silently if no compiler is found;
# the Python implementation is used in that case.
set -euo pipefail
CC=${CC:-clang}
SRC="$(dirname "$0")/../topocdm/_fast.c"
OUT="$(dirname "$0")/../topocdm/_fast.so"
if command -v "$CC" >/dev/null 2>&1; then
  "$CC" -O3 -march=native -shared -fPIC -o "$OUT" "$SRC"
  echo "built $OUT with $CC"
else
  echo "no C compiler found; using the Python kernel" >&2
fi

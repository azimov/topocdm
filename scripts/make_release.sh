#!/usr/bin/env bash
# Make a clean, SHA-named source archive (and optionally a git bundle) of HEAD.
#
#   bash scripts/make_release.sh [output.tar.gz] [--bundle]
#
# The tarball is produced by `git archive`, so it contains only tracked files:
# no .git, no .secrets/, no data/, no .venv/, no PDFs or pickles. It is a
# snapshot; it has no history and cannot be pulled from.
#
# --bundle also writes <output>.bundle, a single-file git repository with the
# full history. Clone it on the work laptop with:
#   git clone topocdm-<sha>.bundle topocdm
# A bundle is the better choice if you need to work on the code and bring
# changes back; a tarball is the better choice if you want no history.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

sha="$(git rev-parse --short HEAD)"
out="${1:-topocdm-$sha.tar.gz}"
want_bundle=0
if [ "${2:-}" = "--bundle" ] || [ "${1:-}" = "--bundle" ]; then
  want_bundle=1
fi
if [ "${1:-}" = "--bundle" ]; then
  out="topocdm-$sha.tar.gz"
fi

git archive --format=tar.gz --prefix="topocdm-$sha/" HEAD -o "$out"

bad="$(tar -tzf "$out" | grep -Ei '(^|/)(\.secrets/|data/|\.venv/|\.git/)|\.pdf$|\.pkl$' || true)"
if [ -n "$bad" ]; then
  echo "Refusing: the archive contains data/secret-looking paths:" >&2
  echo "$bad" >&2
  rm -f "$out"
  exit 1
fi

echo "wrote $out (HEAD $sha, tracked files only)"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$out"
else
  shasum -a 256 "$out"
fi

if [ "$want_bundle" = 1 ]; then
  bundle="${out%.tar.gz}.bundle"
  git bundle create "$bundle" --all
  echo "wrote $bundle (full history, clonable)"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$bundle"
  else
    shasum -a 256 "$bundle"
  fi
  echo "clone on the work laptop: git clone $bundle topocdm"
fi

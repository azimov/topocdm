#!/usr/bin/env bash
# Push the current branch to the private work mirror.
#
#   bash scripts/push_work.sh
#
# Requires a remote named "work" (see docs/WORK_ACCOUNT.md). Refuses to push
# if the tree is dirty or if any tracked file looks like data or a secret.
# Never makes anything public; it only pushes to the remote you configured.
set -euo pipefail

REMOTE="${WORK_REMOTE:-work}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "No '$REMOTE' remote. See docs/WORK_ACCOUNT.md to add the private work mirror." >&2
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "Working tree is dirty; commit or stash before pushing to work." >&2
  exit 1
fi

bad="$(git ls-files | grep -Ei '(^|/)(\.secrets/|data/)|\.pdf$|\.pkl$|\.env$' || true)"
if [ -n "$bad" ]; then
  echo "Refusing to push: tracked data/secret-looking files:" >&2
  echo "$bad" >&2
  exit 1
fi

branch="$(git branch --show-current)"
if [ -z "$branch" ]; then
  echo "Detached HEAD; checkout a branch first." >&2
  exit 1
fi

git push "$REMOTE" "$branch"
echo
echo "Pushed '$branch' to: $(git remote get-url "$REMOTE")"
echo "Keep this repository PRIVATE. Do not change its visibility."

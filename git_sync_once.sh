#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

if [[ -n "$(git status --porcelain)" ]]; then
  ts="$(date -u +"%Y-%m-%d %H:%M:%S UTC")"
  git add -A
  git commit -m "chore(sync): auto update ${ts}" || true
else
  echo "No changes to commit."
fi

git push origin main

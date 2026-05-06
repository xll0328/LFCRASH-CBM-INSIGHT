#!/usr/bin/env bash
set -euo pipefail

INTERVAL_SECONDS="${1:-300}"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"
PID_FILE="$REPO_DIR/.git_sync_loop.pid"
echo "$$" > "$PID_FILE"

echo "Starting auto sync loop in $REPO_DIR (interval=${INTERVAL_SECONDS}s)"
while true; do
  if [[ -n "$(git status --porcelain)" ]]; then
    ts="$(date -u +"%Y-%m-%d %H:%M:%S UTC")"
    git add -A
    git commit -m "chore(sync): auto update ${ts}" || true
    git push origin main
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] synced"
  fi
  sleep "$INTERVAL_SECONDS"
done

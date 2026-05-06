#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EMNLP_DIR="$ROOT/paper/emnlp2026"
FREEZE_ROOT="$EMNLP_DIR/frozen"
STAGE_STATUS="$ROOT/EMNLP_STAGE_STATUS.md"

resolve_freeze_dir() {
  if [[ $# -gt 0 && -n "${1:-}" ]]; then
    local target="$1"
    if [[ -d "$target" ]]; then
      printf '%s\n' "$(cd "$target" && pwd)"
    else
      printf '%s/%s\n' "$FREEZE_ROOT" "$target"
    fi
    return
  fi

  if [[ ! -f "$STAGE_STATUS" ]]; then
    echo "[verify] missing $STAGE_STATUS and no stamp was provided" >&2
    return 1
  fi

  local latest
  latest="$(grep -o "$FREEZE_ROOT/ARR[0-9TZ]*" "$STAGE_STATUS" | tail -n 1 || true)"
  if [[ -z "$latest" ]]; then
    echo "[verify] could not find latest ARR freeze in $STAGE_STATUS" >&2
    return 1
  fi
  printf '%s\n' "$latest"
}

FREEZE_DIR="$(resolve_freeze_dir "${1:-}")"
STAMP="$(basename "$FREEZE_DIR")"
MANIFEST_JSON="$FREEZE_DIR/freeze_manifest.json"
MANIFEST_MD="$FREEZE_DIR/freeze_manifest.md"
PKG_DIR="$FREEZE_DIR/package"
TAR_PATH="$FREEZE_ROOT/insight_emnlp_arr_freeze_${STAMP}.tar.gz"

[[ -d "$FREEZE_DIR" ]] || { echo "[verify] missing freeze dir $FREEZE_DIR" >&2; exit 1; }
[[ -d "$PKG_DIR" ]] || { echo "[verify] missing package dir $PKG_DIR" >&2; exit 1; }
[[ -f "$MANIFEST_JSON" ]] || { echo "[verify] missing $MANIFEST_JSON" >&2; exit 1; }
[[ -f "$MANIFEST_MD" ]] || { echo "[verify] missing $MANIFEST_MD" >&2; exit 1; }
[[ -f "$TAR_PATH" ]] || { echo "[verify] missing tarball $TAR_PATH" >&2; exit 1; }

tar -tzf "$TAR_PATH" >/dev/null

python3 - "$FREEZE_DIR" "$STAMP" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

freeze_dir = Path(sys.argv[1]).resolve()
stamp = sys.argv[2]
pkg_dir = freeze_dir / "package"
manifest_path = freeze_dir / "freeze_manifest.json"
manifest = json.loads(manifest_path.read_text())

errors = []

def fail(message: str) -> None:
    errors.append(message)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

if manifest.get("stamp") != stamp:
    fail(f"manifest stamp mismatch: {manifest.get('stamp')} != {stamp}")

if Path(manifest.get("package_root", "")).resolve() != pkg_dir:
    fail("manifest package_root does not match freeze package dir")

files = manifest.get("files")
if not isinstance(files, list) or not files:
    fail("manifest files list is missing or empty")
else:
    for entry in files:
        rel = entry.get("path")
        if not isinstance(rel, str):
            fail(f"manifest entry missing path: {entry!r}")
            continue
        path = pkg_dir / rel
        if not path.is_file():
            fail(f"listed file missing: {rel}")
            continue
        expected_size = entry.get("size_bytes")
        if expected_size is not None and path.stat().st_size != expected_size:
            fail(f"size mismatch for {rel}")
        expected_sha = entry.get("sha256")
        if expected_sha and sha256(path) != expected_sha:
            fail(f"sha256 mismatch for {rel}")

key_artifacts = manifest.get("key_artifacts", {})
if not isinstance(key_artifacts, dict) or not key_artifacts:
    fail("key_artifacts missing")
else:
    for name, rel in key_artifacts.items():
        if not isinstance(rel, str):
            fail(f"key artifact {name} has non-string path")
            continue
        if not (pkg_dir / rel).is_file():
            fail(f"key artifact missing: {name} -> {rel}")

stage_status = pkg_dir / "EMNLP_STAGE_STATUS.md"
if stage_status.is_file():
    text = stage_status.read_text(errors="replace")
    if stamp not in text:
        fail("package EMNLP_STAGE_STATUS.md does not mention freeze stamp")
else:
    fail("package EMNLP_STAGE_STATUS.md missing")

sanity_report = pkg_dir / "paper/emnlp2026/submission_sanity_report.txt"
if sanity_report.is_file():
    text = sanity_report.read_text(errors="replace")
    if "OK fatal_count=0" not in text:
        fail("submission_sanity_report.txt does not report OK fatal_count=0")
else:
    fail("submission_sanity_report.txt missing")

if errors:
    for error in errors:
        print(f"[verify] FAIL {error}", file=sys.stderr)
    raise SystemExit(1)

print(f"[verify] OK manifest/package checks for {stamp}")
PY

echo "[verify] OK tarball $TAR_PATH"

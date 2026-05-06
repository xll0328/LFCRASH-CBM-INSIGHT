#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EMNLP_DIR="$ROOT/paper/emnlp2026"
# Single source of truth for the appendix used by insight_emnlp.tex.
APPENDIX_SRC="$ROOT/paper/neurips2026/sec_appendix.tex"
FREEZE_ROOT="$EMNLP_DIR/frozen"
STAMP="${1:-ARR$(date -u +%Y%m%dT%H%M%SZ)}"
FREEZE_DIR="$FREEZE_ROOT/$STAMP"
PKG_DIR="$FREEZE_DIR/package"
TAR_PATH="$FREEZE_ROOT/insight_emnlp_arr_freeze_${STAMP}.tar.gz"
MANIFEST_JSON="$FREEZE_DIR/freeze_manifest.json"
MANIFEST_MD="$FREEZE_DIR/freeze_manifest.md"
STAGE_STATUS="$ROOT/EMNLP_STAGE_STATUS.md"
STAGE_STATUS_BACKUP=""
FREEZE_COMPLETED=0

rollback_stage_status() {
  local exit_code=$?
  if [[ "$FREEZE_COMPLETED" != "1" && -n "$STAGE_STATUS_BACKUP" && -f "$STAGE_STATUS_BACKUP" ]]; then
    cp "$STAGE_STATUS_BACKUP" "$STAGE_STATUS"
    echo "[freeze] restored $STAGE_STATUS after failed freeze" >&2
  fi
  if [[ -n "$STAGE_STATUS_BACKUP" && -f "$STAGE_STATUS_BACKUP" ]]; then
    rm -f "$STAGE_STATUS_BACKUP"
  fi
  exit "$exit_code"
}

trap rollback_stage_status EXIT

if [[ -f "$STAGE_STATUS" ]]; then
  STAGE_STATUS_BACKUP="$(mktemp)"
  cp "$STAGE_STATUS" "$STAGE_STATUS_BACKUP"
  python3 - <<'PY' "$STAGE_STATUS" "$FREEZE_DIR" "$TAR_PATH"
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

stage_path = Path(sys.argv[1])
freeze_dir = sys.argv[2]
tar_path = sys.argv[3]
text = stage_path.read_text()

today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
text = re.sub(r"^Date: \d{4}-\d{2}-\d{2}", f"Date: {today}", text, count=1, flags=re.MULTILINE)

patterns = [
    (
        r"(- Remote freeze directory:\n\s+`)[^`]+(`)",
        rf"\1{freeze_dir}\2",
        "remote freeze directory",
    ),
    (
        r"(- Remote tarball:\n\s+`)[^`]+(`)",
        rf"\1{tar_path}\2",
        "remote tarball",
    ),
]

for pattern, replacement, label in patterns:
    text, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise SystemExit(f"Could not update {label} in {stage_path}")

stage_path.write_text(text)
PY
fi

bash "$EMNLP_DIR/run_submission_sanity_checks.sh" >/dev/null

mkdir -p "$PKG_DIR/paper/emnlp2026" "$PKG_DIR/paper/figures" \
  "$PKG_DIR/paper/neurips2026" "$PKG_DIR/visualizations" \
  "$PKG_DIR/output/emnlp2026_support"

copy_file() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
}

copy_dir() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  cp -R "$src" "$dst"
}

# Core paper sources and compiled artifacts
for file in \
  ARR_LAST_MINUTE_CHECKLIST.md acl.sty acl_natbib.bst \
  arr_emnlp2026_workflow_checklist.txt audit_controlled_ontology_runs.py \
  claim_evidence_audit_report.md compile_emnlp.sh freeze_arr_submission.sh insight.bib \
  insight_emnlp.pdf insight_emnlp.tex insight_emnlp_first25.pdf \
  pdf_first_read_audit_report.md run_claim_evidence_audit.py \
  run_pdf_first_read_audit.py run_dad_mechanism_lightreg_block.sh \
  run_top_conference_quality_audit.py \
  reviewer_defense_coverage_report.md run_reviewer_defense_audit.py \
  run_emnlp_support_analyses.sh run_submission_sanity_checks.sh \
  summarize_dad_mechanism_lightreg_status.py watch_dad_mechanism_lightreg_status.py \
  verify_arr_freeze.sh \
  sec_conclusion_emnlp.tex \
  sec_experiments_emnlp.tex sec_intro_emnlp.tex sec_method_emnlp.tex \
  sec_related_emnlp.tex submission_package_notes.txt \
  submission_sanity_report.txt; do
  copy_file "$EMNLP_DIR/$file" "$PKG_DIR/paper/emnlp2026/$file"
done
copy_file "$APPENDIX_SRC" "$PKG_DIR/paper/emnlp2026/sec_appendix.tex"

# Referenced paper assets
copy_dir "$ROOT/paper/figures" "$PKG_DIR/paper/figures"
copy_file "$ROOT/paper/neurips2026/dad_case_contactsheet.pdf" \
  "$PKG_DIR/paper/neurips2026/dad_case_contactsheet.pdf"
for viz_asset in paper_strip.png timeline_concepts.png; do
  copy_file "$ROOT/visualizations/crash/$viz_asset" \
    "$PKG_DIR/visualizations/crash/$viz_asset"
done

# Reviewer/support notes
for doc in \
  EMNLP_SUPPORT_RESULTS.md EMNLP_CONTROLLED_ONTOLOGY_STATUS.md \
  EMNLP_FINAL_READINESS_AUDIT.md \
  EMNLP_INTERVENTION_STATUS.md EMNLP_REVIEW_RESPONSE_MAP.md \
  EMNLP_REVIEW_RESPONSE_PLAYBOOK.md \
  EMNLP_REVIEW_RESPONSE_TEMPLATES.md EMNLP_REVIEWER_QUICK_MAP.md \
  EMNLP_REVIEW_RESPONSE_TRACKER.md EMNLP_STAGE_STATUS.md \
  EMNLP_MASTER_EXECUTION_PLAN.md EMNLP_ORAL_PUSH_PLAN.md \
  EMNLP_ORAL_ACCEPT_CASE_ONEPAGER_20260427.md \
  EMNLP_ORAL_BEST_PAPER_GAP_LEDGER_20260427.md \
  EMNLP_ORAL_BEST_PAPER_EXECUTION_DASHBOARD_20260427.md \
  EMNLP_TOP_CONFERENCE_QUALITY_AUDIT_20260427.md \
  EMNLP_DAD_MECHANISM_HARDENING_PREDECLARED_PLAN_20260427.md \
  EMNLP_RERUN_PLAN.md EMNLP_AUTONOMY_HANDOFF.md; do
  if [[ -f "$ROOT/$doc" ]]; then
    copy_file "$ROOT/$doc" "$PKG_DIR/$doc"
  fi
done

for support_file in \
  controlled_ontology_status.json controlled_ontology_status.md \
  multiseed_ontology_status.json multiseed_ontology_status.md \
  a3d_headline_multiseed_status.json a3d_headline_multiseed_status.md \
  oral_readiness_audit.json oral_readiness_audit.md \
  top_conference_quality_audit.json \
  dad_curriculum_recovery_status.json dad_curriculum_recovery_status.md \
  dad_curriculum_sync_summary.json dad_curriculum_sync_summary.md \
  dad_core_ablation_summary.json dad_core_ablation_summary.md \
  dad_hardening_status.json dad_hardening_status.md \
  dad_mechanism_lightreg_status.json dad_mechanism_lightreg_status.md \
  dad_trigger_compare_extended_summary.json dad_trigger_compare_extended_summary.md \
  a3d_core_ablation_summary.json a3d_core_ablation_summary.md \
  human_ontology_audit_summary.json human_ontology_audit_summary.md \
  emnlp_status_snippets.tex emnlp_status_snippets.md \
  concept_verbalization_sensitivity_dad120.json \
  dad120_frame_manifest.json \
  topm_pseudolabel_sensitivity_dad120.json \
  concept_verbalization_sensitivity_dad500.json \
  dad500_frame_manifest.json \
  topm_pseudolabel_sensitivity_dad500.json; do
  if [[ -f "$ROOT/output/emnlp2026_support/$support_file" ]]; then
    copy_file "$ROOT/output/emnlp2026_support/$support_file" \
      "$PKG_DIR/output/emnlp2026_support/$support_file"
  fi
done

python3 - <<'PY' "$PKG_DIR" "$MANIFEST_JSON" "$MANIFEST_MD" "$STAMP"
import hashlib
import json
import os
import sys
from pathlib import Path

pkg_dir = Path(sys.argv[1])
manifest_json = Path(sys.argv[2])
manifest_md = Path(sys.argv[3])
stamp = sys.argv[4]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

files = []
for path in sorted(pkg_dir.rglob("*")):
    if path.is_file():
        rel = path.relative_to(pkg_dir).as_posix()
        files.append({
            "path": rel,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path) if rel.endswith((".pdf", ".tex", ".bib", ".md", ".json", ".txt", ".py", ".sh", ".sty", ".bst")) else None,
        })

manifest = {
    "stamp": stamp,
    "package_root": str(pkg_dir),
    "num_files": len(files),
    "files": files,
    "key_artifacts": {
        "full_pdf": "paper/emnlp2026/insight_emnlp.pdf",
        "first25_pdf": "paper/emnlp2026/insight_emnlp_first25.pdf",
    "entrypoint": "paper/emnlp2026/insight_emnlp.tex",
    "stage_status": "EMNLP_STAGE_STATUS.md",
    "autonomy_handoff": "EMNLP_AUTONOMY_HANDOFF.md",
    "master_execution_plan": "EMNLP_MASTER_EXECUTION_PLAN.md",
    "oral_accept_case_onepager": "EMNLP_ORAL_ACCEPT_CASE_ONEPAGER_20260427.md",
    "oral_best_paper_gap_ledger": "EMNLP_ORAL_BEST_PAPER_GAP_LEDGER_20260427.md",
    "oral_best_paper_execution_dashboard": "EMNLP_ORAL_BEST_PAPER_EXECUTION_DASHBOARD_20260427.md",
    "dad_mechanism_predeclared_plan": "EMNLP_DAD_MECHANISM_HARDENING_PREDECLARED_PLAN_20260427.md",
    "reviewer_quick_map": "EMNLP_REVIEWER_QUICK_MAP.md",
    "review_map": "EMNLP_REVIEW_RESPONSE_MAP.md",
    "review_response_playbook": "EMNLP_REVIEW_RESPONSE_PLAYBOOK.md",
    "review_response_templates": "EMNLP_REVIEW_RESPONSE_TEMPLATES.md",
    "review_tracker": "EMNLP_REVIEW_RESPONSE_TRACKER.md",
    "intervention_status": "EMNLP_INTERVENTION_STATUS.md",
    "controlled_ontology_status": "EMNLP_CONTROLLED_ONTOLOGY_STATUS.md",
    "final_readiness_audit": "EMNLP_FINAL_READINESS_AUDIT.md",
    "submission_sanity_report": "paper/emnlp2026/submission_sanity_report.txt",
    "claim_evidence_audit": "paper/emnlp2026/claim_evidence_audit_report.md",
    "pdf_first_read_audit": "paper/emnlp2026/pdf_first_read_audit_report.md",
    "reviewer_defense_coverage_audit": "paper/emnlp2026/reviewer_defense_coverage_report.md",
    "top_conference_quality_audit": "EMNLP_TOP_CONFERENCE_QUALITY_AUDIT_20260427.md",
    "top_conference_quality_audit_json": "output/emnlp2026_support/top_conference_quality_audit.json",
    "dad_mechanism_lightreg_launcher": "paper/emnlp2026/run_dad_mechanism_lightreg_block.sh",
    "freeze_verifier": "paper/emnlp2026/verify_arr_freeze.sh",
    "oral_readiness_audit": "output/emnlp2026_support/oral_readiness_audit.md",
    "multiseed_ontology_status": "output/emnlp2026_support/multiseed_ontology_status.md",
    "a3d_headline_multiseed_status": "output/emnlp2026_support/a3d_headline_multiseed_status.md",
    "dad_hardening_status": "output/emnlp2026_support/dad_hardening_status.md",
    "dad_mechanism_lightreg_status": "output/emnlp2026_support/dad_mechanism_lightreg_status.md",
    "human_ontology_audit": "output/emnlp2026_support/human_ontology_audit_summary.md",
    "dad500_topm_sensitivity": "output/emnlp2026_support/topm_pseudolabel_sensitivity_dad500.json",
    "dad500_verbalization_sensitivity": "output/emnlp2026_support/concept_verbalization_sensitivity_dad500.json",
    "status_snippets": "output/emnlp2026_support/emnlp_status_snippets.md",
    "motivation_strip": "visualizations/crash/paper_strip.png",
    "motivation_timeline": "visualizations/crash/timeline_concepts.png",
    },
}
manifest_json.write_text(json.dumps(manifest, indent=2))

lines = [
    "# ARR Freeze Manifest",
    "",
    f"- Stamp: `{stamp}`",
    f"- Package root: `{pkg_dir}`",
    f"- Number of files: `{len(files)}`",
    "",
    "## Key Artifacts",
]
for k, v in manifest["key_artifacts"].items():
    lines.append(f"- {k}: `{v}`")
lines += ["", "## Checksums"]
for rel in [
    "paper/emnlp2026/insight_emnlp.pdf",
    "paper/emnlp2026/insight_emnlp_first25.pdf",
    "paper/emnlp2026/insight_emnlp.tex",
    "paper/emnlp2026/insight.bib",
]:
    path = pkg_dir / rel
    lines.append(f"- `{rel}`: `{sha256(path)}`")
manifest_md.write_text("\n".join(lines) + "\n")
print(f"[wrote] {manifest_json}")
print(f"[wrote] {manifest_md}")
PY

mkdir -p "$FREEZE_ROOT"
LC_ALL=C tar -czf "$TAR_PATH" -C "$FREEZE_DIR" package freeze_manifest.json freeze_manifest.md

bash "$EMNLP_DIR/verify_arr_freeze.sh" "$STAMP"
FREEZE_COMPLETED=1
if [[ -n "$STAGE_STATUS_BACKUP" ]]; then
  rm -f "$STAGE_STATUS_BACKUP"
  STAGE_STATUS_BACKUP=""
fi

echo "[freeze] created $FREEZE_DIR"
echo "[freeze] tarball $TAR_PATH"

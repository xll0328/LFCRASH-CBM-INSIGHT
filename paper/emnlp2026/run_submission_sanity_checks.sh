#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EMNLP_DIR="$ROOT/paper/emnlp2026"
# Single source of truth for the appendix used by insight_emnlp.tex.
APPENDIX_SRC="$ROOT/paper/neurips2026/sec_appendix.tex"
REPORT="$EMNLP_DIR/submission_sanity_report.txt"
FULL_PDF="$EMNLP_DIR/insight_emnlp.pdf"
FIRST25_PDF="$EMNLP_DIR/insight_emnlp_first25.pdf"
CLAIM_AUDIT_REPORT="$EMNLP_DIR/claim_evidence_audit_report.md"
PDF_FIRST_READ_REPORT="$EMNLP_DIR/pdf_first_read_audit_report.md"
REVIEWER_DEFENSE_REPORT="$EMNLP_DIR/reviewer_defense_coverage_report.md"
TOP_CONFERENCE_QUALITY_REPORT="$ROOT/EMNLP_TOP_CONFERENCE_QUALITY_AUDIT_20260427.md"
MANUSCRIPT_SOURCES=(
  "$EMNLP_DIR/insight_emnlp.tex"
  "$EMNLP_DIR/sec_intro_emnlp.tex"
  "$EMNLP_DIR/sec_related_emnlp.tex"
  "$EMNLP_DIR/sec_method_emnlp.tex"
  "$EMNLP_DIR/sec_experiments_emnlp.tex"
  "$EMNLP_DIR/sec_conclusion_emnlp.tex"
  "$EMNLP_DIR/insight.bib"
  "$APPENDIX_SRC"
)

if command -v rg >/dev/null 2>&1; then
  SEARCH_BIN=(rg -n)
else
  SEARCH_BIN=(grep -R -n -E)
fi

KEY_ARTIFACTS=(
  "$ROOT/EMNLP_STAGE_STATUS.md"
  "$ROOT/EMNLP_MASTER_EXECUTION_PLAN.md"
  "$ROOT/EMNLP_ORAL_PUSH_PLAN.md"
  "$ROOT/EMNLP_ORAL_ACCEPT_CASE_ONEPAGER_20260427.md"
  "$ROOT/EMNLP_ORAL_BEST_PAPER_GAP_LEDGER_20260427.md"
  "$ROOT/EMNLP_ORAL_BEST_PAPER_EXECUTION_DASHBOARD_20260427.md"
  "$ROOT/EMNLP_DAD_MECHANISM_HARDENING_PREDECLARED_PLAN_20260427.md"
  "$ROOT/EMNLP_RERUN_PLAN.md"
  "$ROOT/EMNLP_REVIEW_RESPONSE_MAP.md"
  "$ROOT/EMNLP_REVIEW_RESPONSE_PLAYBOOK.md"
  "$ROOT/EMNLP_REVIEW_RESPONSE_TEMPLATES.md"
  "$ROOT/EMNLP_REVIEW_RESPONSE_TRACKER.md"
  "$ROOT/EMNLP_SUPPORT_RESULTS.md"
  "$ROOT/EMNLP_INTERVENTION_STATUS.md"
  "$ROOT/EMNLP_CONTROLLED_ONTOLOGY_STATUS.md"
  "$ROOT/EMNLP_FINAL_READINESS_AUDIT.md"
  "$ROOT/EMNLP_REVIEWER_QUICK_MAP.md"
  "$ROOT/output/emnlp2026_support/controlled_ontology_status.md"
  "$ROOT/output/emnlp2026_support/multiseed_ontology_status.md"
  "$ROOT/output/emnlp2026_support/a3d_headline_multiseed_status.md"
  "$ROOT/output/emnlp2026_support/dad_hardening_status.md"
  "$ROOT/output/emnlp2026_support/dad_mechanism_lightreg_status.md"
  "$ROOT/output/emnlp2026_support/top_conference_quality_audit.json"
  "$ROOT/output/emnlp2026_support/oral_readiness_audit.md"
  "$ROOT/output/emnlp2026_support/human_ontology_audit_summary.md"
  "$ROOT/output/emnlp2026_support/topm_pseudolabel_sensitivity_dad500.json"
  "$ROOT/output/emnlp2026_support/concept_verbalization_sensitivity_dad500.json"
  "$ROOT/output/emnlp2026_support/dad500_frame_manifest.json"
  "$EMNLP_DIR/run_claim_evidence_audit.py"
  "$CLAIM_AUDIT_REPORT"
  "$EMNLP_DIR/run_pdf_first_read_audit.py"
  "$PDF_FIRST_READ_REPORT"
  "$EMNLP_DIR/run_reviewer_defense_audit.py"
  "$REVIEWER_DEFENSE_REPORT"
  "$EMNLP_DIR/run_top_conference_quality_audit.py"
  "$TOP_CONFERENCE_QUALITY_REPORT"
  "$EMNLP_DIR/run_dad_mechanism_lightreg_block.sh"
  "$EMNLP_DIR/summarize_dad_mechanism_lightreg_status.py"
  "$EMNLP_DIR/watch_dad_mechanism_lightreg_status.py"
)

fatal_count=0

fatal() {
  echo "$1"
  fatal_count=$((fatal_count + 1))
}

{
  echo "INSIGHT submission sanity report"
  echo "generated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo

  echo "[1] Core files"
  for file in insight_emnlp.tex insight_emnlp.pdf insight_emnlp_first25.pdf insight.bib sec_intro_emnlp.tex sec_related_emnlp.tex sec_method_emnlp.tex sec_experiments_emnlp.tex sec_conclusion_emnlp.tex; do
    if [[ -f "$EMNLP_DIR/$file" ]]; then
      echo "OK $file"
    else
      fatal "MISSING $file"
    fi
  done
  if [[ -f "$APPENDIX_SRC" ]]; then
    echo "OK sec_appendix.tex ($APPENDIX_SRC)"
  else
    fatal "MISSING sec_appendix.tex"
  fi
  if [[ -f "$FULL_PDF" && -f "$FIRST25_PDF" ]]; then
    full_mtime=$(stat -c %Y "$FULL_PDF")
    first25_mtime=$(stat -c %Y "$FIRST25_PDF")
    if (( first25_mtime < full_mtime )); then
      fatal "STALE insight_emnlp_first25.pdf older than insight_emnlp.pdf"
    else
      echo "OK insight_emnlp_first25.pdf freshness"
    fi

    newest_source=""
    newest_source_mtime=0
    for source_file in "${MANUSCRIPT_SOURCES[@]}"; do
      if [[ ! -f "$source_file" ]]; then
        continue
      fi
      source_mtime=$(stat -c %Y "$source_file")
      if (( source_mtime > newest_source_mtime )); then
        newest_source_mtime="$source_mtime"
        newest_source="$source_file"
      fi
    done
    if (( newest_source_mtime > full_mtime )); then
      fatal "STALE insight_emnlp.pdf older than manuscript source $newest_source"
    else
      echo "OK insight_emnlp.pdf newer than manuscript sources"
    fi
  fi
  echo

  echo "[2] Compilation warnings"
  if [[ -f "$EMNLP_DIR/insight_emnlp.log" ]]; then
    compile_warnings="$(grep -n "undefined\\|Warning\\|Overfull\\|Underfull" "$EMNLP_DIR/insight_emnlp.log" | head -n 40 || true)"
    if [[ -n "$compile_warnings" ]]; then
      printf '%s\n' "$compile_warnings"
      if printf '%s\n' "$compile_warnings" | grep -Eq "undefined|Warning|Overfull"; then
        fatal "CHECK compilation log contains undefined references, warnings, or overfull boxes"
      fi
    else
      echo "OK no compile warnings matched"
    fi
  else
    fatal "MISSING insight_emnlp.log"
  fi
  echo

  echo "[3] Obvious identity / link scan"
  if [[ "${SEARCH_BIN[0]}" == "rg" ]]; then
    identity_matches="$("${SEARCH_BIN[@]}" \
      --glob '!paper/extracted/**' \
      --glob '!**/*.pdf' --glob '!**/*.png' --glob '!**/*.jpg' \
      '(https?://|github\\.com|gitlab\\.com|huggingface\\.co|dropbox\\.com|drive\\.google\\.com|@[A-Za-z0-9._%+-]+\\.[A-Za-z]{2,})' \
      "$EMNLP_DIR/insight_emnlp.tex" \
      "$EMNLP_DIR"/sec_*_emnlp.tex \
      "$APPENDIX_SRC" \
      "$ROOT"/EMNLP_*.md \
      "$ROOT/output/emnlp2026_support/" 2>/dev/null || true)"
  else
    identity_matches="$("${SEARCH_BIN[@]}" \
      '(https?://|github\.com|gitlab\.com|huggingface\.co|dropbox\.com|drive\.google\.com|@[A-Za-z0-9._%+-]+\.[A-Za-z]{2,})' \
      "$EMNLP_DIR/insight_emnlp.tex" \
      "$EMNLP_DIR"/sec_*_emnlp.tex \
      "$APPENDIX_SRC" \
      "$ROOT"/EMNLP_*.md \
      "$ROOT/output/emnlp2026_support/" 2>/dev/null || true)"
  fi
  if [[ -n "$identity_matches" ]]; then
    printf '%s\n' "$identity_matches"
    fatal "CHECK obvious identity/link scan produced matches"
  else
    echo "OK no obvious identity/link matches"
  fi
  echo

  echo "[4] PDF checks"
  command -v pdfinfo >/dev/null 2>&1 && pdfinfo "$FULL_PDF" || echo "pdfinfo not available"
  if ! python3 - "$FULL_PDF" "$FIRST25_PDF" <<'PY'; then
import sys
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("PDF Python check unavailable: missing pypdf/PyPDF2")
        raise SystemExit(0)

full_path = Path(sys.argv[1])
first25_path = Path(sys.argv[2])
if not full_path.exists() or not first25_path.exists():
    print("PDF Python check skipped: PDF missing")
    raise SystemExit(0)

full = PdfReader(str(full_path))
first25 = PdfReader(str(first25_path))
full_pages = len(full.pages)
first25_pages = len(first25.pages)
expected_first25 = min(25, full_pages)
has_issue = False
if first25_pages == expected_first25:
    print(f"OK PDF page counts full={full_pages} first25={first25_pages}")
else:
    print(f"CHECK PDF page counts full={full_pages} first25={first25_pages} expected_first25={expected_first25}")
    has_issue = True

metadata = full.metadata or {}
for key in ("/Author", "/Title", "/Subject", "/Keywords"):
    value = metadata.get(key)
    if value:
        print(f"CHECK PDF metadata {key}={value}")
        has_issue = True
    else:
        print(f"OK PDF metadata {key} empty")

raise SystemExit(1 if has_issue else 0)
PY
    fatal "CHECK PDF Python validation reported an issue"
  fi
  echo

  echo "[5] Claim/evidence text audit"
  if claim_audit_output="$(python3 "$EMNLP_DIR/run_claim_evidence_audit.py" 2>&1)"; then
    printf '%s\n' "$claim_audit_output"
    echo "OK claim/evidence audit passed"
  else
    printf '%s\n' "$claim_audit_output"
    fatal "CHECK claim/evidence audit reported blockers"
  fi
  echo

  echo "[6] PDF first-read audit"
  if pdf_first_read_output="$(python3 "$EMNLP_DIR/run_pdf_first_read_audit.py" 2>&1)"; then
    printf '%s\n' "$pdf_first_read_output"
    echo "OK PDF first-read audit passed"
  else
    printf '%s\n' "$pdf_first_read_output"
    fatal "CHECK PDF first-read audit reported blockers"
  fi
  echo

  echo "[7] Reviewer defense coverage audit"
  if reviewer_defense_output="$(python3 "$EMNLP_DIR/run_reviewer_defense_audit.py" 2>&1)"; then
    printf '%s\n' "$reviewer_defense_output"
    echo "OK reviewer defense coverage audit passed"
  else
    printf '%s\n' "$reviewer_defense_output"
    fatal "CHECK reviewer defense coverage audit reported blockers"
  fi
  echo

  echo "[8] Top-conference quality audit"
  if top_conference_output="$(python3 "$EMNLP_DIR/run_top_conference_quality_audit.py" 2>&1)"; then
    printf '%s\n' "$top_conference_output"
    echo "OK top-conference quality audit generated"
  else
    printf '%s\n' "$top_conference_output"
    fatal "CHECK top-conference quality audit failed"
  fi
  echo

  echo "[9] Key support artifacts"
  for file in "${KEY_ARTIFACTS[@]}"; do
    if [[ -f "$file" ]]; then
      echo "OK $file"
    else
      fatal "MISSING $file"
    fi
  done
  echo

  echo "[10] Stale project-state reference scan"
  if [[ "${SEARCH_BIN[0]}" == "rg" ]]; then
    stale_matches="$("${SEARCH_BIN[@]}" \
      'insight\.tex|ARR20260417|ARR20260426T130347Z|one-seed block|one-seed matched|not yet a broad|current 120-frame|missing cells|controlled ontology multi-seed block is incomplete|Oral-ready: no|oral-ready: no' \
      "$ROOT"/EMNLP_*.md \
      "$EMNLP_DIR"/*.md \
      "$EMNLP_DIR"/*.txt 2>/dev/null || true)"
  else
    stale_matches="$("${SEARCH_BIN[@]}" \
      'insight\.tex|ARR20260417|ARR20260426T130347Z|one-seed block|one-seed matched|not yet a broad|current 120-frame|missing cells|controlled ontology multi-seed block is incomplete|Oral-ready: no|oral-ready: no' \
      "$ROOT"/EMNLP_*.md \
      "$EMNLP_DIR"/*.md \
      "$EMNLP_DIR"/*.txt 2>/dev/null || true)"
  fi
  if [[ -n "$stale_matches" ]]; then
    printf '%s\n' "$stale_matches"
    fatal "CHECK stale project-state reference scan produced matches"
  else
    echo "OK no stale project-state references matched"
  fi
  echo

  echo "[11] Sanity verdict"
  if (( fatal_count == 0 )); then
    echo "OK fatal_count=0"
  else
    echo "FAIL fatal_count=$fatal_count"
  fi
} > "$REPORT"

echo "[sanity] wrote $REPORT"
sed -n '1,220p' "$REPORT"
if (( fatal_count > 0 )); then
  exit 1
fi

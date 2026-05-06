# NeurIPS 2026 Project Status

Date: 2026-05-01
Project: LFCRASH-CBM / INSIGHT

## Current Position

The active NeurIPS paper is `paper/neurips2026/insight_main.tex`, not the
legacy standalone draft at `paper/NEURIPS2026_full_draft.md`.

The paper-facing thesis is:

> Interpretable accident anticipation should be built on an intervenable
> semantic state space, not post-hoc explanation over a black-box predictor.

The current package is best understood as oral-track strengthening work:
strongest on semantic-state construction, ontology governance, A3D operating
point evidence, and protocol discipline; still not best-paper-ready because
DAD hard-case symmetry and fully policy-level timing/intervention evidence
remain bounded.

## Current Truth Set

- Canonical DAD headline line: 68.19% AP, 1.75s mTTA.
- Canonical A3D headline line: 93.40% AP, 4.90s mTTA, 4.89s TTA@R80, 0.9067 P@R80.
- DAD local-search and ontology-search runs are support-only unless explicitly
  promoted by a new ledger update.
- Actor-policy timing evidence remains mixed and must not be stated as solved.
- Intervention evidence supports partial structural intervenability, not a
  completed causal proof.

## What Was Advanced On 2026-05-01

- Added `run_neurips_sanity_checks.py` as a lightweight NeurIPS claim/evidence
  gate.
- Generated `neurips_sanity_report.md`; latest result is `OK fatal_count=0`
  with one expected warning about stale language in the legacy full draft.
- Marked `paper/NEURIPS2026_full_draft.md` as legacy/stale at the top of the file.
- Updated `appendix_asset_index.md` so future readers can find the sanity gate,
  sanity report, and current status file.
- Follow-up compile/sanity pass at 18:43 UTC: reran the NeurIPS sanity gate
  before and after compilation, rebuilt `insight_main.pdf`, fixed the stale
  appendix protocol-map reference from `fig:concept_case_study_main` to
  `fig:concept_case_study`, and wrapped the main-paper protocol map table to
  remove the compile-time overfull box.
- Latest compile log has no undefined references and no overfull boxes; only
  underfull layout warnings remain.
- Figure-facing follow-up at 18:58 UTC: replaced the introduction motivation
  figure's CRASH placeholder-frame asset with the real DAD dual-layer
  interpretability asset, then removed the duplicate single-case DAD figure from
  the experiments section so the paper now uses the real single-case figure in
  the introduction and the two-case figure in the interpretability analysis.
- Added `paper/neurips2026/gpt_image2_teaser/` with a high-information
  GPT Image 2 prompt, secure environment-key generation script, quality gate,
  and iteration log for producing a paper-facing graphical abstract candidate
  without storing API keys in the repository.
- GPT Image 2 follow-up at 19:27 UTC: generated two candidates through
  AIHubMix `gpt-image-2` using a direct no-proxy `curl` route. Promoted the
  high-quality 1536x1024 Candidate B to
  `paper/figures/insight_gpt_image2_teaser.{png,pdf}` and updated the
  introduction Figure `fig:motivation` to use it as a schematic overview with
  an explicitly bounded, non-evidence caption.
- Verification after teaser promotion: rebuilt `insight_main.pdf`, rendered the
  first two pages for inspection, reran the sanity gate with
  `OK fatal_count=0 warn_count=1`, and confirmed no API key was stored in the
  project artifacts.

## What Was Advanced On 2026-05-06

- Tightened main-paper scope language around the two live weaknesses:
  `paper/neurips2026/sec_experiments.tex` now states explicitly that current
  synchronized diagnostics do **not** establish DAD hard-case symmetry, and
  `paper/neurips2026/sec_conclusion.tex` adds DAD hard-case symmetry as an open
  limitation.
- Synchronized evidence-governance files with the same boundary:
  `paper/neurips2026/thesis_claims_evidence_matrix.md` now lists DAD hard-case
  symmetry as not-yet-proven / not-settled; and
  `paper/neurips2026/reviewer_proof_experiment_manifest.md` now requires a
  family-level hard-case symmetry audit table before any stronger symmetry claim.
- Updated `paper/neurips2026/claim_evidence_audit.json` with an explicit
  `hard_case_symmetry_scope` reviewer-sensitive guardrail.
- Added a refreshable hard-case symmetry scaffold:
  `paper/neurips2026/build_dad_hard_case_symmetry_audit.py` now generates
  `paper/neurips2026/dad_hard_case_symmetry_audit.md` from archived casebank
  assets, explicitly marking symmetry as open unless paired family-level
  success/failure evidence is filled.
- Upgraded the hard-case scaffold to use a primary-family counting rule and
  auto-suggest mixed early-vs-late pairs (currently 4 heuristic pair suggestions
  from `strong_top` assets), while keeping all pair labels explicitly marked as
  needing manual reviewer confirmation before any symmetry promotion.
- Added an explicit sentence in `paper/neurips2026/sec_experiments.tex` tying
  hard-case symmetry claims to that scaffold as a blocking audit gate.
- Rebuilt `paper/neurips2026/insight_main.pdf` and reran the NeurIPS sanity
  gate. Latest status at 05:56 UTC is `OK fatal_count=0 warn_count=1`
  (only expected legacy-draft warning remains).

## Immediate Safe Next Actions

1. After any NeurIPS paper edit, run:

```bash
python3 paper/neurips2026/run_neurips_sanity_checks.py
bash paper/neurips2026/compile_insight.sh
```

2. Keep all new claims synchronized with:

- `paper/neurips2026/submission_results_ledger.json`
- `paper/neurips2026/claim_evidence_audit.json`
- `paper/neurips2026/thesis_claims_evidence_matrix.md`
- `paper/neurips2026/reviewer_proof_experiment_manifest.md`

3. If the next task is best-paper strengthening, prioritize evidence that
reduces the two live weaknesses: DAD hard-case symmetry and policy-level timing
/ intervention validation.

4. If the next task is pure paper hygiene, the remaining low-risk target is
underfull layout polish in `insight_main.log`; this is secondary to the final
human read-through.

5. If the next task is further image-model iteration, prefer the direct
no-proxy `curl` route recorded in
`paper/neurips2026/gpt_image2_teaser/iteration_log.md`; the Python SDK route was
unreliable from this server during the 2026-05-01 session. Securely inject
`AIHUBMIX_API_KEY` in the shell environment and run:

```bash
python3 paper/neurips2026/gpt_image2_teaser/generate.py \
  --prompt paper/neurips2026/gpt_image2_teaser/prompt_v1.md \
  --n 2 --size 1536x1024 --quality high
```

Then score each candidate with
`paper/neurips2026/gpt_image2_teaser/quality_gate.md` before replacing any
paper figure.

## Stop Conditions

- Any fatal sanity-gate failure.
- Any proposal to use stale `97.36%`, `9.59s`, or speculative `75-80%+`
  convergence claims from the legacy draft.
- Any claim that actor-policy timing or semantic intervention is fully solved
  without new supporting artifacts.
- Any new GPU experiment, long sweep, or freeze cut without explicit approval.

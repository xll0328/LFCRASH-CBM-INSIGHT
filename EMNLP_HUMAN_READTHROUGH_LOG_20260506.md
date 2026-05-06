# EMNLP Human Read-Through Log (Phase B)

Date: 2026-05-06

## Scope

- `paper/emnlp2026/sec_intro_emnlp.tex`
- `paper/emnlp2026/sec_method_emnlp.tex`
- `paper/emnlp2026/sec_experiments_emnlp.tex`
- `paper/emnlp2026/sec_conclusion_emnlp.tex`

## Goal

Reduce reviewer over-interpretation risk while preserving all reported numbers and claim tiers.

## Applied Minimal Wording Updates

1. Replaced "universal policy-level causality" with
   "complete policy-level intervention control" in intro scope boundary.
2. Replaced "full policy-level causal validation" with
   "full policy-level intervention validation" in method rationale.
3. Replaced "policy-level causal timing account" with
   "policy-level intervention timing account" in experiments.
4. Replaced "complete causal timing proof" with
   "complete policy-level intervention timing proof" in experiments.
5. Replaced "complete policy-level causal intervention study" with
   "complete policy-level intervention validation study" in conclusion.

## Non-Changes (Deliberate)

- No metric values were changed.
- No evidence-tier labels were removed.
- No new claims were added.
- A3D flagship / DAD stress asymmetry remains unchanged.

## Post-Edit Gate

- `bash paper/emnlp2026/run_submission_sanity_checks.sh`
- Expected: `OK fatal_count=0`

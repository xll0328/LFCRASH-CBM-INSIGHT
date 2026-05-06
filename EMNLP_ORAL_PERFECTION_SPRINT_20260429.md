# EMNLP Oral Perfection Sprint

Date: 2026-04-29
Project: `LFCRASH-CBM`

This sprint is an expert-review plan for moving the current ARR-ready package
closer to EMNLP oral quality without violating the project guardrails. It is
grounded in the current manuscript, support artifacts, and automated audits; it
is not human upload approval.

## Current Expert Verdict

- ARR / regular acceptance: credible.
- Oral: possible, but not locked.
- Best paper: not credible from the current evidence alone.

The main oral gap is not a broken package. The package is technically clean.
The gap is reviewer interpretation: the paper must make the semantic-interface
object feel inevitable, keep DAD as honest stress evidence, and prevent support
analyses from looking like a scattered collection of runs.

## Top Oral Gaps

### Gap 1: First-read object clarity

- Risk: a reviewer reads the paper as accident anticipation with concept
  prompting, not as a governed semantic-interface paper.
- Current assets: Figure 1, framework figure, concept-pipeline figure,
  protocol map, visual evidence map.
- Sprint action: remove paper-facing meta language about oral/best-paper
  posture and replace it with claim-tier reading language.
- Done in this sprint: main-text wording now frames the object as a governed
  semantic interface and avoids explicit acceptance-target language.

### Gap 2: Evidence tier discipline

- Risk: the many recipe families look fragmented or cherry-picked.
- Current assets: protocol map, experiment portfolio figure, visual evidence
  map, sanity/claim audits.
- Sprint action: every result family must have one role: headline, controlled
  support, or stress evidence.
- Done in this sprint: wording around reader route, experiment breadth, data
  efficiency, and ablation captions was tightened to use claim-tier language.

### Gap 3: DAD fragility

- Risk: DAD instability is read as method failure rather than as the paper's
  explicit hard stress setting.
- Current assets: DAD stress summary, hardening status, low-regularization
  status, curriculum-recovery status.
- Sprint action: keep DAD bounded and avoid upgrading mixed mechanism evidence.
- Stop condition: any stronger DAD mechanism claim requires new aggregate
  evidence or explicit human decision.

### Gap 4: Actor-policy scope

- Risk: actor-policy timing is read as causal proof.
- Current assets: trigger-source comparison, extended trigger audit, conclusion
  limitations.
- Sprint action: keep actor-policy timing as downstream transfer/support
  evidence only.
- Stop condition: policy-level causality cannot be claimed from current
  artifacts.

### Gap 5: Related-work authority

- Risk: the paper is accepted as technically coherent but not seen as fully
  situated in ontology governance, concept bottlenecks, and safety-critical
  VLM evaluation.
- Current assets: 49 cited keys, no missing BibTeX keys, related-work section.
- Sprint action: target only missing, high-value citations if a concrete source
  is already available; do not add broad literature claims without reading.
- Stop condition: new literature search or citation claims require source
  verification.

### Gap 6: Human-validation boundary

- Risk: language-side audits are mistaken for exhaustive human concept labels.
- Current assets: DAD-500 pseudo-label audit, paraphrase sensitivity, 80-concept
  light review.
- Sprint action: keep "governed semantic artifact" wording; do not imply a
  completed frame-level human concept benchmark.

## Sprint Plan

### P0: Keep the package green

1. Recompile after any source edit.
2. Regenerate `insight_emnlp_first25.pdf`.
3. Run `bash paper/emnlp2026/run_submission_sanity_checks.sh`.
4. Run `python3 paper/emnlp2026/refresh_emnlp_status.py`.

### P1: Main-text oral polish

1. Remove paper-facing acceptance-target language from main text.
2. Replace "reviewer concern" and "oral-level" wording with claim-tier and
   evidence-tier wording.
3. Tighten Figure 1, protocol-map, data-efficiency, and ablation captions so
   each one states its evidential role.
4. Keep A3D as the clean flagship and DAD as the stress setting.

### P2: Defense package hardening

1. Keep `EMNLP_REVIEWER_QUICK_MAP.md` synchronized with the current claim
   hierarchy.
2. Keep rebuttal language grounded in artifact paths.
3. Add no new claims during rebuttal preparation unless the claim/evidence
   audit remains clean.

### P3: Evidence escalation only with approval

1. Do not start new GPU work from this sprint.
2. Continue monitoring `lf_arch_watch`.
3. Treat architecture-extension results as support-only until both 3-run
   families complete.
4. Any new DAD mechanism block requires explicit GPU authorization and a
   predeclared success/tie/failure reading.

## Immediate Changes Made

- Replaced main-text acceptance-target language with evidence-tier wording.
- Removed "Oral-first, best-paper-in-progress" from the introduction.
- Removed "Review-readiness summary for acceptance targets" from the
  conclusion.
- Reworded reviewer-facing meta phrases in experiments to claim-tier language.
- Kept all central claims unchanged.

## Remaining Human Stop Conditions

- Human PDF read-through of page 1, Figure 1, Table 1, key result tables, and
  appendix opening.
- Author list, ARR account / reviewer registration, venue commitment, and
  upload logistics.
- Any decision to cut a new freeze.
- Any decision to broaden evidence with new compute.


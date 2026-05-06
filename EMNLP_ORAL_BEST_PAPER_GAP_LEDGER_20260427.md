# EMNLP Oral / Best-Paper Gap Ledger

Date: 2026-04-27
Project: `LFCRASH-CBM`

This ledger records the current distance from ARR-ready submission to EMNLP
oral and best-paper competitiveness. It is grounded in the current repository
artifacts and should not be read as human upload approval.

Execution dashboard:

- `EMNLP_ORAL_BEST_PAPER_EXECUTION_DASHBOARD_20260427.md`

## Current Verdict

- ARR submission package: `ready pending human upload logistics`
- Oral competitiveness: `credible`
- Best-paper competitiveness: `not yet`
- Main reason: the semantic-interface story is coherent and well packaged, but
  best-paper-level confidence still needs stronger DAD mechanism clarity and
  broader cross-setting evidence.

## Evidence Already Strong Enough

### Semantic-interface claim

- Status: `ready`
- Evidence:
  - `output/emnlp2026_support/multiseed_ontology_status.md`
  - `EMNLP_CONTROLLED_ONTOLOGY_STATUS.md`
  - `paper/emnlp2026/claim_evidence_audit_report.md`
- Current support:
  - controlled ontology multi-seed coverage is `18/18`
  - the paper-facing claim is ontology choice changes the AP--mTTA operating
    point, not that one prompt gives a universal leaderboard win
- Reviewer posture:
  - safe as a main contribution
  - keep ontology construction framed as a governed semantic artifact

### A3D flagship evidence

- Status: `oral-ready`
- Evidence:
  - `output/emnlp2026_support/a3d_headline_multiseed_status.md`
  - `output/emnlp2026_support/oral_readiness_audit.md`
- Current support:
  - A3D headline aggregate is `94.16% +/- 0.95` AP and
    `4.62s +/- 0.42s` mTTA over `3/3` seeds
- Reviewer posture:
  - use A3D as the clean flagship result
  - do not let DAD diagnostics pull the whole paper into a weakness-first story

### Language-side support

- Status: `submission-ready`
- Evidence:
  - `EMNLP_SUPPORT_RESULTS.md`
  - `output/emnlp2026_support/topm_pseudolabel_sensitivity_dad500.json`
  - `output/emnlp2026_support/concept_verbalization_sensitivity_dad500.json`
  - `output/emnlp2026_support/human_ontology_audit_summary.md`
- Current support:
  - DAD language-side audit now uses `500` frames
  - concept verbalization sensitivity reports mean text cosine `0.9389`,
    frame-score correlation `0.8872`, and mean absolute score difference
    `0.0134`
  - human ontology audit covers `80` concepts across `9` families
- Reviewer posture:
  - this supports EMNLP fit and semantic artifact quality
  - do not describe it as exhaustive human validation

### Submission package integrity

- Status: `ready`
- Evidence:
  - `paper/emnlp2026/submission_sanity_report.txt`
  - `paper/emnlp2026/pdf_first_read_audit_report.md`
  - `paper/emnlp2026/frozen/ARR20260427T032620Z/freeze_manifest.md`
- Current support:
  - sanity reports `OK fatal_count=0`
  - claim/evidence audit reports `critical_blockers=0`
  - PDF first-read audit reports `critical_blockers=0`
- Reviewer posture:
  - package is technically coherent
  - human read-through and upload decisions remain outside automation

## Remaining Distance To Oral

Distance: `small to moderate`

The paper is already in a credible oral band because the central claim is
clear, the submission package is clean, and the main evidence is seed-backed.
The remaining oral risk is not a missing script or package artifact; it is
reviewer interpretation.

Highest-risk reviewer interpretations:

- "This is just concept prompting rather than a research object."
- "DAD results look fragile, so the method may not really work."
- "Actor-policy timing sounds causal but is not proven."
- "The ontology comparison might be a hidden tuning artifact."

Current response assets:

- `EMNLP_REVIEWER_QUICK_MAP.md`
- `EMNLP_REVIEW_RESPONSE_MAP.md`
- `EMNLP_REVIEW_RESPONSE_PLAYBOOK.md`
- `EMNLP_REVIEW_RESPONSE_TEMPLATES.md`
- `EMNLP_FINAL_READINESS_AUDIT.md`

Next oral-focused actions that do not require GPU:

1. Do the final human PDF read-through on page 1, Figure 1, Table 1, and the
   appendix opening.
2. During the read-through, mark any sentence that sounds like DAD causality,
   actor-policy proof, or universal ontology superiority.
3. Use the reviewer quick map as the response anchor rather than inventing new
   claims during rebuttal.

## Remaining Distance To Best Paper

Distance: `large but diagnosable`

Best-paper readiness would require the paper to look not only coherent and
strong, but unusually decisive. The current package is not there yet because
the stress-test side remains mixed.

### Gap 1: DAD mechanism clarity

- Current status: `mixed evidence`
- Evidence:
  - `output/emnlp2026_support/dad_hardening_status.md`
  - `output/emnlp2026_support/dad_mechanism_lightreg_lowreg_status.md`
- Current facts:
  - canonical DAD line remains `68.19%` AP and `1.75s` mTTA
  - matched DAD full support block is complete at `3/3` runs
  - full aggregate is `63.19% +/- 1.21` AP and `2.17s +/- 0.05s` mTTA
  - low-regularization completion is `64.42% +/- 0.97` AP and `2.35s +/- 0.20` mTTA (`3/3`, useful-tie band)
  - full-vs-ablation deltas are mixed rather than a clean mechanism win
- Best-paper blocker:
  - DAD supports the paper as a hard stress test, but does not yet make the
    mechanism feel inevitable
- Allowed next compute only with explicit GPU authorization:
  - one targeted DAD mechanism-hardening block
  - no repeat of completed ontology multi-seed, A3D headline, DAD full-support,
    or DAD-500 language audits unless an artifact is corrupted

### Gap 2: Actor-policy maturity

- Current status: `support-only`
- Evidence:
  - `output/emnlp2026_support/dad_hardening_status.md`
  - `output/emnlp2026_support/oral_readiness_audit.md`
- Current facts:
  - DAD trigger-source audit favors classifier triggers over actor triggers
  - actor branch should not be promoted to a causal timing claim
- Best-paper blocker:
  - the paper cannot currently claim a mature policy-level timing mechanism
- Allowed next action:
  - keep actor-policy wording bounded
  - only revisit with a scoped pilot and explicit compute approval
  - current low-regularization follow-up is complete; any best-paper move should build on this boundary.

### Gap 3: Breadth beyond current datasets

- Current status: `not in current submission scope`
- Current facts:
  - A3D is clean and DAD is fragile
  - the core story is cross-dataset enough for ARR, but not broad enough to
    feel definitive at best-paper level
- Best-paper blocker:
  - a best-paper case would benefit from another independent stress setting or
    a stronger generality argument
- Allowed next action:
  - record this as post-submission research direction
  - do not add speculative claims to the current ARR paper

## Decision Gates

### Gate A: Submission lock

Pass condition:

- latest freeze remains `ARR20260427T032620Z`
- sanity remains `OK fatal_count=0`
- final human read-through finds no claim-overreach sentence

If passed:

- do not reopen central story before upload

### Gate B: Oral defense readiness

Pass condition:

- reviewer quick map can answer EMNLP fit, ontology modeling, DAD fragility,
  actor-policy scope, and intervention scope without adding new claims

If failed:

- fix response materials first
- do not run new experiments as a substitute for clear framing

### Gate C: Best-paper escalation

Pass condition:

- DAD mechanism-hardening plan is explicitly authorized for GPU use
- the plan has a predeclared success / failure reading
- paper claims will be updated only from aggregate evidence

If failed:

- keep the paper in acceptance/oral mode
- do not chase broad experiments before submission

## Concrete Next Actions

### No-compute path

1. Human read-through and upload logistics.
2. Freeze-path confirmation against `EMNLP_STAGE_STATUS.md`.
3. Rebuttal drill using `EMNLP_REVIEW_RESPONSE_PLAYBOOK.md`.
4. Figure/table visual inspection at normal PDF zoom.

### Compute-authorized path

1. Define exactly one DAD mechanism-hardening block.
2. Write its success/failure interpretation before launch.
3. Run only after explicit GPU approval.
4. Refresh `output/emnlp2026_support/dad_hardening_status.md`.
5. Update paper language only if aggregate evidence changes the claim tier.
6. If the first light-regularization block remains mixed, launch the
   follow-up low-regularization probe (`run_dad_mechanism_lightreg_lowreg_block.sh`)
   under explicit approval.

## Forbidden Moves

- Do not claim best-paper readiness from packaging quality alone.
- Do not treat AI-assisted audits as human validation.
- Do not promote DAD from stress test to flagship.
- Do not promote actor-policy timing to a causal claim.
- Do not rerun completed support blocks to hunt for a more convenient number.
- Do not change the central paper story before upload unless the human
  read-through finds a concrete error.

## Bottom Line

The project is close to a strong ARR / EMNLP submission and plausibly in the
oral conversation. The remaining gap to best paper is substantial because the
paper would need a cleaner DAD mechanism story or broader independent evidence,
not merely a cleaner package. The correct immediate strategy is to lock the
current submission, avoid overclaiming, and reserve any new compute for a
single predeclared DAD mechanism-hardening block after explicit approval.

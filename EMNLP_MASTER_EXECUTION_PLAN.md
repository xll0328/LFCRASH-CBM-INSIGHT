# EMNLP 2026 Master Execution Plan

Date: 2026-04-26

## Mission

Maximize the paper's probability of:

- safe ARR / EMNLP submission
- strong accept recommendation
- credible oral-level upside

This plan is written from the current repository state, not from an idealized
future state.

## Current audited state

- Submission-ready for ARR / EMNLP: yes
- Oral-ready: yes
- Best-paper-ready: no
- Strongest current story: a language-grounded risk concept interface whose
  ontology choice changes the AP--mTTA operating point
- Strongest current dataset: A3D
- Most fragile dataset: DAD
- Current actor-policy status: supporting evidence only, not a main claim

## What is already solid

- Paper framing is aligned around the semantic interface claim.
- Freeze packaging exists and the latest recommended freeze is
  `paper/emnlp2026/frozen/ARR20260426T131028Z`.
- Submission sanity report is clean enough for upload, with only small layout
  warnings remaining.
- Controlled ontology comparison is complete at both single-run and `18/18`
  multi-seed support levels.
- A3D headline stability is complete at `3/3` seeds.
- Language-side support analyses now use the current 500-frame DAD audit and
  are scoped appropriately.
- DAD hardening boards correctly record that the matched full support block is
  complete; the remaining DAD issue is mechanism fragility, not missing
  coverage.
- Timing / intervention wording is already bounded conservatively.

## What currently blocks best-paper-level confidence

### 1. DAD is still visibly fragile

- The canonical DAD line is useful and competitive.
- The clean-seed, synchronized-epoch, and full-vs-ablation diagnostics show
  non-trivial variance and mixed mechanism evidence.
- This is acceptable for submission, but it must be framed with discipline.

### 2. Actor-policy evidence remains support-only

- Prediction timing is measurable.
- The archived actor branch remains too flat for a main policy-level causal
  timing claim.

### 3. Human logistics remain outside the repository

- Author list, ARR reviewer registration, venue commitment, and final manual
  PDF read-through still need human completion.

## Priority order

## P0: Keep submission-safe status

- Do not reopen the paper's central claim.
- Do not rewrite the model story around actor-policy timing.
- Do not mix search runs, support runs, and headline runs.
- Do not overwrite frozen artifacts unless a new freeze is intentionally cut.

## P1: Highest-value oral-upside work

- Preserve the completed multi-seed ontology, A3D headline, and DAD diagnostic
  evidence in the frozen package.
- Keep response materials aligned with the current support boards.
- Make final manual read-through and upload logistics the default next action.

## P2: Next-best evidence upgrades

- If compute is explicitly authorized, run one targeted DAD mechanism-hardening
  block that directly addresses the mixed full-vs-ablation story.
- Do not repeat completed ontology multi-seed, A3D headline, DAD full-support,
  or DAD-500 language-side audits unless a concrete artifact is corrupted.

## P3: Best-paper upside only after P1 is stable

- Polish the opening page until the contribution reads immediately.
- Tighten tables and figure captions so the semantic-interface claim lands
  without explanation overhead.
- Improve appendix readability and reviewer-navigation quality.
- Prepare rebuttal-ready evidence maps and response templates.

## Writing rules

- The main claim is semantic-interface science, not raw leaderboard chasing.
- A3D is the clean flagship dataset.
- DAD should be written as honest, bounded, and still meaningful.
- Actor-policy timing is support evidence unless a new experiment materially
  changes the picture.
- Ontology construction must read as a governed research artifact, not prompt
  engineering.

## Immediate next actions for takeover

1. Run submission sanity after any manuscript or support-document edit.
2. Cut a new freeze whenever reviewer-facing docs or package scripts change.
3. Keep the latest freeze path in `EMNLP_STAGE_STATUS.md`.
4. Treat any new GPU work as opt-in and scoped to DAD mechanism hardening.

## What not to spend time on right now

- Crash-only score chasing
- architecture rewrites
- aggressive actor-branch repositioning
- broad ablation refreshes with no direct paper impact

## Operational rule

Every new block must answer one of two questions:

- Does this make acceptance safer?
- Does this materially improve oral-level confidence?

If the answer is neither, it is not a priority before submission.

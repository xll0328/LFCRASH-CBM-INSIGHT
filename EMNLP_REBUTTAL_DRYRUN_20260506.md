# EMNLP Rebuttal Dry-Run (5 High-Risk Attacks)

Date: 2026-05-06  
Scope: fast internal rehearsal aligned with current evidence tiers

## 1) "This is prompt engineering, not NLP research."

Short response:
The contribution is not prompt-only behavior. The paper treats ontology construction as a governed, reproducible modeling component and evaluates ontology variants under matched launchers; changing the ontology shifts AP--mTTA operating points on both DAD and A3D.

Evidence:
- `paper/emnlp2026/sec_method_emnlp.tex`
- `paper/emnlp2026/sec_experiments_emnlp.tex` (`tab:concept_sets`)
- `output/emnlp2026_support/multiseed_ontology_status.md`

## 2) "Your DAD evidence is unstable, so the method may not work."

Short response:
DAD is explicitly framed as stress-test evidence, not the flagship mechanism proof. We report canonical and support diagnostics separately and keep claims bounded; the clean flagship evidence is A3D.

Evidence:
- `paper/emnlp2026/sec_experiments_emnlp.tex` (`tab:dad`, DAD stress paragraphs)
- `output/emnlp2026_support/dad_hardening_status.md`
- `output/emnlp2026_support/a3d_headline_multiseed_status.md`

## 3) "You claim policy-level causality without proof."

Short response:
We do not make that claim. Actor-policy timing is support-only, and intervention is reported as partial structural intervenability with explicit limits.

Evidence:
- `paper/emnlp2026/sec_experiments_emnlp.tex` (`tab:trigger_compare`, `tab:interp_quant`)
- `paper/neurips2026/sec_appendix.tex`
- `paper/emnlp2026/claim_evidence_audit_report.md`

## 4) "Why EMNLP and not just AV benchmarking?"

Short response:
The paper's central object is a language-grounded semantic interface: canonical concept naming, merge policy, family balancing, paraphrase robustness, and auditability are methodological units, not post-hoc add-ons.

Evidence:
- `paper/emnlp2026/sec_intro_emnlp.tex`
- `paper/emnlp2026/sec_method_emnlp.tex`
- `paper/figures/insight_fig_concept_pipeline.pdf`
- `output/emnlp2026_support/concept_verbalization_sensitivity_dad500.json`

## 5) "Results are fragmented across many blocks."

Short response:
The evidence is intentionally tiered to prevent pooled cherry-picking. The paper now tags major artifacts as headline / controlled support / stress and provides a protocol map for deterministic reading order.

Evidence:
- `paper/emnlp2026/sec_experiments_emnlp.tex` (`tab:protocol_map_main`, caption tier tags)
- `paper/figures/insight_fig9_experiment_portfolio.pdf`
- `paper/figures/insight_fig5_safety_utility.pdf`

## Rule For Live Rebuttal

If a response cannot cite one concrete artifact path and one bounded claim sentence, do not use it.

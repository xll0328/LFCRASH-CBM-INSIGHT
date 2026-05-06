# EMNLP Virtual Review Action Matrix

Date: 2026-05-06  
Scope: map virtual pre-review weaknesses to concrete fixes and evidence

## Critical Issues -> Actions

| Reviewer concern | Action taken | Artifact path | Current status |
|---|---|---|---|
| Over-scaffolded narrative and defensive tone | Compressed reading-guidance prose; kept one protocol map and one evidence taxonomy | `paper/emnlp2026/sec_intro_emnlp.tex`, `paper/emnlp2026/sec_experiments_emnlp.tex` | Done |
| Claim feels broader than evidence | Tightened wording to matched training protocols and bounded stress claims | `paper/emnlp2026/sec_intro_emnlp.tex`, `paper/emnlp2026/sec_conclusion_emnlp.tex` | Done |
| Singleton-heavy ontology narrative | Added seed-aggregated ontology effect-size paragraph with CIs | `paper/emnlp2026/sec_experiments_emnlp.tex`, `output/emnlp2026_support/ontology_effect_size_summary.md` | Done |
| Inspectability vs faithfulness blurred | Added explicit partial-faithfulness boundary language and semantic-validity table | `paper/emnlp2026/sec_experiments_emnlp.tex` (`tab:semantic_validity`) | Done |
| Ontology protocol under-specified | Added operational curation rules (filter/merge/balance/review) in method | `paper/emnlp2026/sec_method_emnlp.tex` | Done |
| Fairness of baseline comparison unclear | Added comparability note for backbone/protocol mismatch | `paper/emnlp2026/sec_experiments_emnlp.tex` | Done |
| Missing recent related work | Added verified 2025 references currently checkable | `paper/emnlp2026/insight.bib`, `paper/emnlp2026/sec_related_emnlp.tex` | Done |
| Ontology-source effect confounded with concept count | Added dedicated size-matched control track (30/80 budgets), deterministic historical subset builder, live queued runs, explicit source-vs-count caveat text in main experiments, and auto effect-size summarizer for direct paper integration | `paper/emnlp2026/run_ontology_size_matched_controls.sh`, `paper/emnlp2026/build_historical_size_matched_subsets.py`, `paper/emnlp2026/audit_ontology_size_matched_controls.py`, `paper/emnlp2026/summarize_ontology_size_matched_effects.py`, `output/emnlp2026_support/ontology_size_matched_effects.md`, `paper/emnlp2026/sec_experiments_emnlp.tex` | Running |
| Seed-block completion overestimated by intermediate snapshots | Switched audit logic to completion-safe accounting (result file + process finished); running jobs are excluded from completed aggregates | `paper/emnlp2026/audit_dad_ontology_seed_extension.py`, `paper/emnlp2026/audit_ontology_size_matched_controls.py` | Done |
| No real-time trend visibility before run completion | Added running-preview metrics and a one-command status refresh chain for DAD extension + size-matched controls + effect summary | `paper/emnlp2026/refresh_ontology_control_status.sh`, `output/emnlp2026_support/*_status.md`, `output/emnlp2026_support/ontology_size_matched_effects.md` | Done |
| Table readability and visual consistency under dense evidence blocks | Standardized table typography + soft macaron header styling and refreshed core figure palettes (framework / concept pipeline / safety-utility / ontology coverage-evolution-case-study); fixed underscore-safe appendix path rendering to keep compile clean | `paper/emnlp2026/insight_emnlp.tex`, `paper/emnlp2026/sec_experiments_emnlp.tex`, `paper/neurips2026/fig_framework.py`, `paper/neurips2026/make_concept_pipeline_fig.py`, `paper/neurips2026/make_concept_family_coverage_fig.py`, `paper/neurips2026/make_concept_evolution_fig.py`, `paper/neurips2026/make_concept_case_study_fig.py`, `paper/neurips2026/insight_viz.py`, `paper/neurips2026/sec_appendix.tex` | Done |
| Appendix hard-case table still looked too engineering-oriented | Replaced underscore-style family identifiers with reader-facing labels while preserving counts and audit semantics | `paper/neurips2026/sec_appendix.tex` | Done |
| Eq. (reward) readability and notation precision | Replaced long case labels with compact indicator-form equation and added symbol legend sentence | `paper/emnlp2026/sec_method_emnlp.tex` | Done |
| 611/81/80 vs 837 ontology-count ambiguity | Added explicit count disambiguation text (governed release ontology vs historical full benchmark vocabulary) in Method + Experiments | `paper/emnlp2026/sec_method_emnlp.tex`, `paper/emnlp2026/sec_experiments_emnlp.tex` | Done |

## Stability Escalation Track (Compute)

| Target | Planned extension | Launcher | Runtime status |
|---|---|---|---|
| `perfect_v1` DAD stability | +5 seeds (`7,11,2718,314,2026`) | `paper/emnlp2026/run_dad_ontology_seed_extension.sh --execute --perfect-only --gpus 4,0` | Running |
| `historical_full` DAD stability | +5 seeds (`7,11,2718,314,2026`) | `paper/emnlp2026/run_dad_ontology_seed_extension.sh --execute --all-concepts --gpus 2,3 --sequential-workers` | Running (queued workers) |
| `risk_core_v1` DAD stability | +5 seeds (`7,11,2718,314,2026`) | same as above | Running (queued workers) |

Status monitor:
- `python3 paper/emnlp2026/audit_dad_ontology_seed_extension.py`
- `output/emnlp2026_support/dad_ontology_seed_extension_status.md`

## Integration Rules (when new seeds finish)

1. Refresh `dad_ontology_seed_extension_status.*`.
2. Recompute ontology effect-size summary.
3. Refresh size-matched control summaries and update the count/source confound paragraph.
4. Update only stability/ontology-control paragraphs/tables; do not reopen story framing.
5. Re-run submission sanity gate.
6. Sync stage/dashboard/todo docs.

## Stop Conditions

- Do not upgrade claim tier until all planned extension seeds finish.
- Do not turn support diagnostics into policy-level causal claims.
- Do not merge queued partial results into headline tables.

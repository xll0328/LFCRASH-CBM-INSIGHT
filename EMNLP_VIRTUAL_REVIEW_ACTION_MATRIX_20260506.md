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
3. Update only stability paragraphs/tables; do not reopen story framing.
4. Re-run submission sanity gate.
5. Sync stage/dashboard/todo docs.

## Stop Conditions

- Do not upgrade claim tier until all planned extension seeds finish.
- Do not turn support diagnostics into policy-level causal claims.
- Do not merge queued partial results into headline tables.

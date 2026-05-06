# Controlled Ontology Comparison Matrix for DAD and A3D

## Purpose
This matrix standardizes ontology comparison under a shared training recipe so that ontology quality can be treated as scientific evidence rather than as anecdotal preprocessing.

## Shared training entrypoint
- Launcher: `run_controlled_ontology_experiment.sh`
- Backend: `train_multi.py`
- Manifest: `output/concept_sets/neurips2026_controlled_ontology_manifest.json`

## Ontology sources

| Concept set | Size | File | Scientific role |
|---|---:|---|---|
| Historical full | 837 | `/data/sony/LFCRASH/000_all_concept_set.txt` | high-coverage historical reference |
| Risk-core v1 | 30 | `output/concept_sets/risk_core_concept_set_v1.txt` | compact human-prior baseline |
| Perfect v1 | 80 | `output/concept_sets/perfect_concept_set_v1.txt` | paper-ready discovered ontology |

## Controlled matrix

### DAD
| Run name | Dataset | Concept set | Tag | Purpose |
|---|---|---|---|---|
| dad_shared_historical_full | dad | historical_full | dad_shared_historical_full | high-coverage reference |
| dad_shared_risk_core_v1 | dad | risk_core_v1 | dad_shared_risk_core_v1 | compact manual baseline |
| dad_shared_perfect_v1 | dad | perfect_v1 | dad_shared_perfect_v1 | discovered ontology comparison |

### A3D
| Run name | Dataset | Concept set | Tag | Purpose |
|---|---|---|---|---|
| a3d_shared_historical_full | a3d | historical_full | a3d_shared_historical_full | high-coverage reference |
| a3d_shared_risk_core_v1 | a3d | risk_core_v1 | a3d_shared_risk_core_v1 | compact manual baseline |
| a3d_shared_perfect_v1 | a3d | perfect_v1 | a3d_shared_perfect_v1 | discovered ontology comparison |

## Required fixed settings
For a valid controlled ontology block:
1. same dataset split
2. same optimizer family
3. same scheduler family
4. same epoch budget
5. same batch size
6. same evaluation cadence
7. ontology source is the only intended variable

## Evaluation outputs
Each run must report:
- AP
- mTTA
- TTA@R80
- P@R80
- concept count
- parameter count
- training recipe ID
- seed

## Secondary diagnostics
In addition to headline metrics, record:
- concept-family balance
- concept sparsity/coherence summary
- intervention readiness note
- whether ontology is paper-facing / canonicalized / human reviewed

## Questions this matrix answers
1. Does ontology quality affect the AP–mTTA frontier?
2. Does a compact ontology retain competitiveness while improving auditability?
3. Does the polished ontology provide a better trade-off than both the broad historical vocabulary and the tiny manual prior?

## Paper usage policy
- Do not mix this block with canonical DAD curriculum numbers.
- Present this block as controlled ontology science, not as the main leaderboard table.
- Use it to support the claim that ontology construction is part of the method.

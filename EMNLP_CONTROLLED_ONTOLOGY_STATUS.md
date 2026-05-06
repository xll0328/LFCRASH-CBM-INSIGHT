# EMNLP Controlled Ontology Status

The controlled ontology block is no longer a planning item. All six recommended
shared-launcher runs are already present and paper-usable.

Current status:

- `dad_shared_historical_full`: AP `64.91%`, mTTA `1.88s`
- `dad_shared_risk_core_v1`: AP `65.25%`, mTTA `2.15s`
- `dad_shared_perfect_v1`: AP `64.15%`, mTTA `2.30s`
- `a3d_shared_historical_full`: AP `93.88%`, mTTA `9.41s`
- `a3d_shared_risk_core_v1`: AP `94.36%`, mTTA `9.57s`
- `a3d_shared_perfect_v1`: AP `96.54%`, mTTA `8.69s`

What this means for the paper:

- DAD already supports the intended story cleanly:
  compact manual ontology gives the best AP, while the polished ontology gives
  the best mTTA.
- A3D also supports the intended story:
  the polished ontology gives the best AP, while the compact manual ontology
  gives the best timing metrics.
- The empirical question is therefore answered:
  ontology choice changes the AP--mTTA operating point under a matched recipe.

Practical implication:

- do not spend immediate compute on rerunning these six lines unless a specific
  artifact is corrupted or a seed refresh becomes reviewer-critical
- spend the next effort budget on presentation, appendix discipline, and only
  then on optional multi-seed extensions

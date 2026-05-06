#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

manifest = {
    'description': 'First-round concept-set comparison experiments',
    'concept_sets': {
        'old_full_837': '/data/sony/LFCRASH/000_all_concept_set.txt',
        'risk_core_v1': '/data/sony/LFCRASH/LFCRASH-CBM/output/concept_sets/risk_core_concept_set_v1.txt',
        'new_discovered_v1': '/data/sony/LFCRASH/LFCRASH-CBM/output/concept_remake_v1/all_concepts_discovered.txt'
    },
    'recommended_runs': [
        {
            'name': 'dad_ac_old_full_837',
            'script': 'train_dad_ac.py',
            'args': ['--tag', 'dad_ac_old_full_837', '--concept_file', '/data/sony/LFCRASH/000_all_concept_set.txt']
        },
        {
            'name': 'dad_ac_risk_core_v1',
            'script': 'train_dad_ac.py',
            'args': ['--tag', 'dad_ac_risk_core_v1', '--num_concepts', '30', '--concept_file', '/data/sony/LFCRASH/LFCRASH-CBM/output/concept_sets/risk_core_concept_set_v1.txt']
        },
        {
            'name': 'dad_ac_new_discovered_v1',
            'script': 'train_dad_ac.py',
            'args': ['--tag', 'dad_ac_new_discovered_v1', '--concept_file', '/data/sony/LFCRASH/LFCRASH-CBM/output/concept_remake_v1/all_concepts_discovered.txt']
        },
        {
            'name': 'a3d_ac_risk_core_v1',
            'script': 'train_multi.py',
            'args': ['--dataset', 'a3d', '--tag', 'a3d_ac_risk_core_v1', '--num_concepts', '30', '--concept_file', '/data/sony/LFCRASH/LFCRASH-CBM/output/concept_sets/risk_core_concept_set_v1.txt']
        }
    ]
}

out = ROOT / 'output' / 'concept_sets' / 'first_round_experiments.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2))
print(out)

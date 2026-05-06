#!/bin/bash
set -e
PY=python3
ROOT="/data/sony/LFCRASH/LFCRASH-CBM"

$PY "$ROOT/train_dad_ac.py" --gpu 0 --tag "dad_ac_old_full_837" --concept_file "/data/sony/LFCRASH/000_all_concept_set.txt"
$PY "$ROOT/train_dad_ac.py" --gpu 1 --tag "dad_ac_risk_core_v1" --num_concepts 30 --concept_file "$ROOT/output/concept_sets/risk_core_concept_set_v1.txt"
# Enable after concept discovery completes:
# $PY "$ROOT/train_dad_ac.py" --gpu 2 --tag "dad_ac_new_discovered_v1" --concept_file "$ROOT/output/concept_remake_v1/all_concepts_discovered.txt"

$PY "$ROOT/train_multi.py" --dataset a3d --gpu 3 --tag "a3d_ac_risk_core_v1" --num_concepts 30 --concept_file "$ROOT/output/concept_sets/risk_core_concept_set_v1.txt"

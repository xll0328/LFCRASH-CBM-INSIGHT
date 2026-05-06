#!/usr/bin/env python3
import json
import re
import subprocess
from pathlib import Path
from statistics import mean, pstdev


EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s+\|")
ENHANCED_EVAL_RE = re.compile(
    r"AP=([0-9.]+), mTTA=([0-9.]+), TTA_R80=([0-9.]+), P_R80=([0-9.]+)"
)
MULTI_EPOCH_RE = re.compile(r"Ep\s+(\d+)/(\d+)")
MULTI_EVAL_RE = re.compile(r"EVAL \| AP=([0-9.]+) mTTA=([0-9.]+) TTA_R80=([0-9.]+)")
TAG_RE = re.compile(r"--tag\s+([^\s]+)")
ONTOLOGY_QUEUE_RE = re.compile(
    r"run_controlled_ontology_experiment\.sh\s+(\w+)\s+([A-Za-z0-9_]+)\s+(\d+)\s+([A-Za-z0-9_]+).*?--seed\s+(\d+)"
)


def load_json(path: Path):
    return json.loads(path.read_text())


def stats(values):
    if not values:
        return None
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": mean(values), "std": pstdev(values)}


def parse_train_enhanced_log(log_path: Path):
    if not log_path.exists():
        return None

    last_epoch = None
    started = False
    best = None
    completed = False

    for line in log_path.read_text().splitlines():
        if "LFCRASH-CBM v4 Enhanced" in line:
            started = True

        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            last_epoch = int(epoch_match.group(1))
            started = True

        eval_match = ENHANCED_EVAL_RE.search(line)
        if eval_match:
            row = {
                "AP": float(eval_match.group(1)),
                "mTTA": float(eval_match.group(2)),
                "TTA_R80": float(eval_match.group(3)),
                "P_R80": float(eval_match.group(4)),
                "epoch": last_epoch,
            }
            if best is None or row["AP"] > best["AP"]:
                best = row

        if "Training Complete!" in line:
            completed = True

    if best is None and not started:
        return None

    result = {"log_path": str(log_path), "last_epoch_seen": last_epoch, "completed_in_log": completed}
    if best is None:
        return result

    best["log_path"] = str(log_path)
    best["last_epoch_seen"] = last_epoch
    best["completed_in_log"] = completed
    return best


def parse_train_multi_log(log_path: Path):
    if not log_path.exists():
        return None

    last_epoch = None
    total_epochs = None
    best = None
    started = False
    completed = False

    for line in log_path.read_text().splitlines():
        if "=== CG-CRASH v4" in line:
            started = True

        epoch_match = MULTI_EPOCH_RE.search(line)
        if epoch_match:
            last_epoch = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2))
            started = True

        eval_match = MULTI_EVAL_RE.search(line)
        if eval_match:
            row = {
                "AP": float(eval_match.group(1)),
                "mTTA": float(eval_match.group(2)),
                "TTA_R80": float(eval_match.group(3)),
                "epoch": last_epoch,
            }
            if best is None or row["AP"] > best["AP"]:
                best = row

        if "=== DONE ===" in line:
            completed = True

    if not started and best is None:
        return None

    result = {
        "log_path": str(log_path),
        "last_epoch_seen": last_epoch,
        "total_epochs": total_epochs,
        "completed_in_log": completed,
    }
    if best is not None:
        result.update(best)
    return result


def discover_ontology_queue_targets(root: Path):
    queue_root = root / "output" / "emnlp2026_support"
    queued = {}

    for queue_dir in sorted(queue_root.glob("ontology_rerun_queue_*"), reverse=True):
        for script_path in sorted(queue_dir.glob("gpu_*.sh")):
            for line_no, line in enumerate(script_path.read_text().splitlines(), start=1):
                match = ONTOLOGY_QUEUE_RE.search(line)
                if match is None:
                    continue

                dataset, concept_set, gpu, tag, seed = match.groups()
                seed_int = int(seed)
                base_tag = tag
                suffix = f"_s{seed_int}"
                if tag.endswith(suffix):
                    base_tag = tag[: -len(suffix)]

                key = (dataset, base_tag, seed_int)
                queued.setdefault(
                    key,
                    {
                        "dataset": dataset,
                        "concept_set": concept_set,
                        "base_tag": base_tag,
                        "tag": tag,
                        "seed": seed_int,
                        "gpu": int(gpu),
                        "queue_dir": str(queue_dir),
                        "queue_script": str(script_path),
                        "line_no": line_no,
                    },
                )

    return queued


def discover_active_train_multi_tags(root: Path):
    result = subprocess.run(
        ["ps", "-eo", "args="],
        capture_output=True,
        text=True,
        check=False,
    )
    active = set()
    if result.returncode != 0:
        return active

    root_str = str(root)
    for line in result.stdout.splitlines():
        if "train_multi.py" not in line or root_str not in line:
            continue
        match = TAG_RE.search(line)
        if match is not None:
            active.add(match.group(1))
    return active

#!/usr/bin/env python3
"""Generate a compact experiment-portfolio figure for submission review.

The figure gives a reviewer-first view of:
1) controlled-ontology coverage density,
2) experiment protocol completion depth,
3) model family / stress-point frontier snapshot.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUPPORT_DIR = ROOT / "output" / "emnlp2026_support"
FIG_DIR = ROOT / "paper" / "figures"
RUN_DAD_DIR = ROOT / "output" / "dad_ac"
RUN_A3D_DIR = ROOT / "output" / "a3d_ac"


@dataclass
class Point:
    label: str
    family: str
    ap: float
    mtta: float
    marker: str = "o"
    size: float = 70
    alpha: float = 0.9
    edge: str = "white"


def pct(v: float) -> str:
    return f"{100.0 * v:.2f}%"


def safe_load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_multiseed_ontology() -> list[dict]:
    data = safe_load_json(SUPPORT_DIR / "multiseed_ontology_status.json")
    if not data:
        return []
    return data.get("rows", [])


def load_a3d_headline():
    return safe_load_json(SUPPORT_DIR / "a3d_headline_multiseed_status.json")


def load_dad_curriculum():
    return safe_load_json(SUPPORT_DIR / "dad_curriculum_recovery_status.json")


def load_dad_hardening():
    return safe_load_json(SUPPORT_DIR / "dad_mechanism_lightreg_status.json")


def load_dad_lowreg():
    return safe_load_json(SUPPORT_DIR / "dad_mechanism_lightreg_lowreg_status.json")


def load_run_result(path: Path):
    payload = safe_load_json(path / "results.json")
    if not payload:
        return None
    ap = payload.get("AP")
    mtta = payload.get("mTTA")
    if ap is None or mtta is None:
        return None
    return payload


def collect_model_family_points():
    points: list[Point] = []

    # A3D headline aggregate.
    a3d = load_a3d_headline()
    if a3d and a3d.get("aggregate_completed"):
        agg = a3d["aggregate_completed"]
        points.append(
            Point(
                label="A3D headline (3-seed)",
                family="A3D headline", 
                ap=100.0 * agg["AP"]["mean"],
                mtta=agg["mTTA"]["mean"],
                marker="s",
            )
        )

    # DAD headline (single canonical checkpoint).
    dad = load_dad_curriculum()
    if dad and dad.get("canonical"):
        c = dad["canonical"]
        points.append(
            Point(
                label="DAD canonical",
                family="DAD headline",
                ap=100.0 * c["AP"],
                mtta=c["mTTA"],
                marker="s",
            )
        )

    # DAD recovery family aggregate.
    if dad and dad.get("combined_v2_family"):
        agg = dad["combined_v2_family"].get("aggregate", {})
        if agg:
            points.append(
                Point(
                    label="DAD curriculum family (10 runs)",
                    family="DAD stress",
                    ap=100.0 * agg["AP"]["mean"],
                    mtta=agg["mTTA"]["mean"],
                    marker="^",
                    alpha=0.8,
                )
            )

    # DAD mechanism hardening blocks.
    hard = load_dad_hardening()
    if hard and hard.get("completed_aggregate"):
        agg = hard["completed_aggregate"]
        points.append(
            Point(
                label="DAD hardening (3 runs)",
                family="DAD stress",
                ap=100.0 * agg["AP"]["mean"],
                mtta=agg["mTTA"]["mean"],
                marker="D",
                alpha=0.7,
            )
        )

    low = load_dad_lowreg()
    if low and low.get("completed_aggregate"):
        agg = low["completed_aggregate"]
        points.append(
            Point(
                label="DAD low-reg follow-up (3 runs)",
                family="DAD stress",
                ap=100.0 * agg["AP"]["mean"],
                mtta=agg["mTTA"]["mean"],
                marker="v",
                alpha=0.7,
            )
        )

    for path in sorted(RUN_A3D_DIR.glob("a3d_ac_perfect_v1_h384*")):
        if not path.is_dir():
            continue
        tag = path.name
        if not re.fullmatch(r"a3d_ac_perfect_v1_h384_q\d+", tag):
            continue
        res = load_run_result(path)
        if not res:
            continue
        points.append(
            Point(
                label=f"A3D {tag}",
                family="A3D h_dim=384",
                ap=100.0 * res["AP"],
                mtta=res["mTTA"],
                marker="x",
                size=55,
                alpha=0.6,
            )
        )

    # DAD model-family extension: RWKV branches.
    for path in sorted(RUN_DAD_DIR.glob("dad_ac_perfect_v1_rwkv*")):
        if not path.is_dir():
            continue
        tag = path.name
        if not re.fullmatch(r"dad_ac_perfect_v1_rwkv_q\d+", tag):
            continue
        res = load_run_result(path)
        if not res:
            continue
        points.append(
            Point(
                label=tag,
                family="DAD RWKV",
                ap=100.0 * res["AP"],
                mtta=res["mTTA"],
                marker="P",
                size=55,
                alpha=0.7,
            )
        )

    return points


def build_coverage_counts(rows: list[dict]):
    dataset_order = ["dad", "a3d"]
    concept_order = ["historical_full", "risk_core_v1", "perfect_v1"]
    map_name = {
        "historical_full": "Historical Full",
        "risk_core_v1": "Risk-core",
        "perfect_v1": "Perfect v1",
    }

    completed = np.zeros((len(dataset_order), len(concept_order)), dtype=float)
    expected = np.zeros_like(completed)

    idx = {(ds, cs): i for i, ds in enumerate(dataset_order) for cs in concept_order}
    for r in rows:
        ds, cs = r.get("dataset"), r.get("concept_set")
        if (ds, cs) not in idx:
            continue
        i, j = dataset_order.index(ds), concept_order.index(cs)
        completed[i, j] = r.get("num_completed", 0)
        expected[i, j] = r.get("num_expected", 0)

    ratios = np.divide(completed, np.maximum(expected, 1.0))

    return {
        "dataset_order": dataset_order,
        "concept_order": concept_order,
        "map_name": map_name,
        "completed": completed,
        "expected": expected,
        "ratios": ratios,
    }


def protocol_bars():
    rows = load_multiseed_ontology()
    a3d = load_a3d_headline()
    dad = load_dad_curriculum()
    hard = load_dad_hardening()
    low = load_dad_lowreg()

    dad_rows = [r for r in rows if r.get("dataset") == "dad"]
    a3d_rows = [r for r in rows if r.get("dataset") == "a3d"]

    done = []
    target = []
    labels = []

    done.append(sum(r.get("num_completed", 0) for r in dad_rows))
    target.append(sum(r.get("num_expected", 0) for r in dad_rows))
    labels.append("DAD ontology cells")

    done.append(sum(r.get("num_completed", 0) for r in a3d_rows))
    target.append(sum(r.get("num_expected", 0) for r in a3d_rows))
    labels.append("A3D ontology cells")

    if a3d:
        done.append(a3d.get("num_completed", 0))
        target.append(a3d.get("num_expected", 0))
    else:
        done.append(0); target.append(0)
    labels.append("A3D headline seeds")

    if dad and dad.get("canonical"):
        done.append(1)
        target.append(1)
    else:
        done.append(0); target.append(1)
    labels.append("DAD canonical")

    if dad and dad.get("combined_v2_family"):
        done.append(dad.get("combined_v2_family", {}).get("num_completed", dad["combined_v2_family"].get("aggregate", {}).get("AP", {}).get("n", 0)))
        target.append(dad.get("combined_v2_family", {}).get("aggregate", {}).get("AP", {}).get("n", done[-1]))
    else:
        done.append(0); target.append(0)
    labels.append("DAD stress-family runs")

    if hard:
        done.append(hard.get("num_completed", 0)); target.append(hard.get("num_started", 3))
    else:
        done.append(0); target.append(3)
    labels.append("DAD mechanism hardening")

    if low:
        done.append(low.get("num_completed", low.get("num_started", 0))); target.append(3)
    else:
        done.append(0); target.append(3)
    labels.append("DAD low-reg follow-up")

    return labels, np.array(done), np.array(target)


def gather_frontier_points() -> list[Point]:
    points = collect_model_family_points()

    # Add ontology-row aggregates for compact comparison.
    for row in load_multiseed_ontology():
        agg = row.get("aggregate")
        if not agg:
            continue
        if row.get("dataset") == "dad":
            family = "DAD ontology"
        else:
            family = "A3D ontology"
        concept_label = row.get("concept_set")
        points.append(
            Point(
                label=f"{row.get('dataset').upper()} {concept_label}",
                family=family,
                ap=100.0 * agg["AP"]["mean"],
                mtta=agg["mTTA"]["mean"],
                marker="o",
                size=45,
                alpha=0.5,
            )
        )

    return points


def draw_figure() -> None:
    rows = load_multiseed_ontology()
    coverage = build_coverage_counts(rows)
    protocol_labels, done, total = protocol_bars()
    frontier_points = gather_frontier_points()

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.0], height_ratios=[1.0, 1.15], hspace=0.30, wspace=0.20)

    # Panel A: ontology coverage matrix.
    ax0 = fig.add_subplot(gs[0, 0])
    cmap = plt.cm.GnBu
    im = ax0.imshow(coverage["ratios"], vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax0.set_title("A) Controlled ontology coverage", fontsize=13)
    ax0.set_xlabel("Ontology variant")
    ax0.set_ylabel("Dataset")
    ax0.set_xticks(range(len(coverage["concept_order"])))
    ax0.set_yticks(range(len(coverage["dataset_order"])))
    ax0.set_xticklabels([coverage["map_name"][k] for k in coverage["concept_order"]], rotation=20, ha="right")
    ax0.set_yticklabels([d.upper() for d in coverage["dataset_order"]])

    for i in range(coverage["ratios"].shape[0]):
        for j in range(coverage["ratios"].shape[1]):
            c = coverage["completed"][i, j]
            e = coverage["expected"][i, j]
            txt = "0/0" if e == 0 else f"{int(c):.0f}/{int(e):.0f}"
            ax0.text(
                j, i, txt,
                ha="center", va="center",
                color="white" if coverage["ratios"][i, j] > 0.60 else "#222222",
                fontsize=9,
                weight="bold",
            )

    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("Completion ratio")

    # Panel B: protocol inventory bars.
    ax1 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(protocol_labels))
    ratios = np.divide(done, np.maximum(total, 1))
    ax1.bar(x, total, color="#d8e6fa", edgecolor="#0f62fe", alpha=0.8, label="Target")
    ax1.bar(x, done, color="#0f62fe", alpha=0.75, label="Completed")
    ax1.set_xticks(x)
    ax1.set_xticklabels(protocol_labels, rotation=32, ha="right")
    ax1.set_ylim(0, max(total) * 1.25 if len(total) > 0 else 1)
    ax1.set_ylabel("Run count")
    ax1.set_title("B) Protocol inventory", fontsize=13)
    for i, (d, t) in enumerate(zip(done, total)):
        ax1.text(i, max(d + 0.15, 0.4), f"{int(d)}/{int(t)}", ha="center", va="bottom", fontsize=8)
    ax1.legend(loc="upper right", fontsize=8)

    # Panel C: family frontier.
    ax2 = fig.add_subplot(gs[1, :])
    families = sorted(set(p.family for p in frontier_points))
    palette = {
        "A3D headline": "#0f62fe",
        "DAD headline": "#da1e28",
        "DAD stress": "#198038",
        "DAD ontology": "#8a3ffc",
        "A3D ontology": "#ff832b",
        "DAD RWKV": "#b42318",
        "A3D h_dim=384": "#3f3f3f",
    }

    by_family = {k: [] for k in families}
    for p in frontier_points:
        by_family.setdefault(p.family, []).append(p)

    for family, pts in sorted(by_family.items()):
        xs = [p.mtta for p in pts]
        ys = [p.ap for p in pts]
        ms = [p.size for p in pts]
        col = palette.get(family, "#161616")
        scatter_kwargs = dict(
            s=ms,
            marker=pts[0].marker,
            color=col,
            alpha=pts[0].alpha,
            label=family,
        )
        if pts[0].marker not in {"x", "+"}:
            scatter_kwargs["edgecolors"] = pts[0].edge
            scatter_kwargs["linewidths"] = 0.7
        ax2.scatter(xs, ys, **scatter_kwargs)
        for p in pts:
            ax2.annotate(
                p.label,
                (p.mtta, p.ap),
                textcoords="offset points",
                xytext=(2, 4),
                fontsize=7,
                alpha=0.8,
            )

    ax2.set_title("C) Experiment frontier snapshot (AP vs mTTA)", fontsize=13)
    ax2.set_xlabel("mTTA (s)")
    ax2.set_ylabel("AP (%)")
    ax2.grid(alpha=0.25)
    ax2.legend(loc="best", fontsize=8, ncol=2)

    fig.suptitle("Experiment Portfolio Map: breadth, depth, and model-family frontiers", fontsize=15)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = FIG_DIR / f"insight_fig9_experiment_portfolio.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[written] {FIG_DIR / 'insight_fig9_experiment_portfolio.pdf'}")
    print(f"[written] {FIG_DIR / 'insight_fig9_experiment_portfolio.png'}")


def main() -> int:
    if not SUPPORT_DIR.exists():
        print(f"[warn] support dir not found: {SUPPORT_DIR}")
        return 1
    draw_figure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

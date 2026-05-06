#!/usr/bin/env python3
"""Build the INSIGHT semantic-interface framework figure."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


rcParams["font.family"] = "DejaVu Sans"
rcParams["font.size"] = 9

ROOT = Path("/data/sony/LFCRASH/LFCRASH-CBM")
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#202124"
MUTED = "#5f6368"
LINE = "#4b5563"
BLUE = "#2F6B9A"
RED = "#B94135"
GREEN = "#2E7D52"
ORANGE = "#C46A1A"
PURPLE = "#6B4FA3"
TEAL = "#167C80"
BG = "#FFFFFF"
BAND = "#F7F8FA"


def rounded_box(ax, x, y, w, h, title, body, fc, ec, title_color=None, lw=1.5):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.66,
        title,
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=title_color or ec,
        zorder=4,
    )
    ax.text(
        x + w / 2,
        y + h * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=7.2,
        color=MUTED,
        linespacing=1.15,
        zorder=4,
    )
    return patch


def arrow(ax, start, end, color=LINE, lw=1.55, rad=0.0, dashed=False):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        connectionstyle=f"arc3,rad={rad}",
        linewidth=lw,
        color=color,
        linestyle="--" if dashed else "-",
        zorder=5,
    )
    ax.add_patch(patch)
    return patch


def label(ax, x, y, text, color=INK, size=8.2, weight="normal", ha="center"):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=size,
        color=color,
        fontweight=weight,
        zorder=6,
        bbox=dict(facecolor=BG, edgecolor="none", alpha=0.72, pad=0.6),
    )


def chip(ax, x, y, text, color):
    rounded_box(ax, x, y, 1.22, 0.36, text, "", "#FFFFFF", color, title_color=color, lw=1.15)


def main() -> int:
    fig, ax = plt.subplots(figsize=(15.2, 6.2), facecolor=BG)
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # Background lanes.
    lanes = [
        (0.25, 4.35, 14.7, 1.45, "#F5F8FC", "Language-side ontology construction"),
        (0.25, 2.35, 14.7, 1.55, "#FFF7F2", "Trainable semantic interface"),
        (0.25, 0.45, 14.7, 1.35, "#F5FAF7", "Anticipation decision and audit trail"),
    ]
    for x, y, w, h, fc, title in lanes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                facecolor=fc,
                edgecolor="#E4E7EB",
                linewidth=0.9,
                zorder=0,
            )
        )
        ax.text(x + 0.25, y + h - 0.2, title, ha="left", va="top", fontsize=8.3, color=MUTED)

    ax.text(
        7.6,
        6.02,
        "INSIGHT centers the language-grounded semantic interface",
        ha="center",
        va="center",
        fontsize=13.2,
        fontweight="bold",
        color=INK,
    )

    # Top lane: ontology pipeline.
    top_y = 4.72
    rounded_box(ax, 0.65, top_y, 1.55, 0.72, "Risk frames", "pre-crash\nsampling", "#FFFFFF", BLUE)
    rounded_box(ax, 2.65, top_y, 1.85, 0.72, "VLM proposals", "risk-first\nphrases", "#FFFFFF", PURPLE)
    rounded_box(ax, 4.95, top_y, 1.9, 0.72, "Canonicalize", "merge aliases\nremove clutter", "#FFFFFF", ORANGE)
    rounded_box(ax, 7.3, top_y, 1.85, 0.72, "Balance", "semantic\nfamilies", "#FFFFFF", TEAL)
    rounded_box(ax, 9.6, top_y, 1.75, 0.72, "Review", "provenance +\nlight audit", "#FFFFFF", GREEN)
    rounded_box(ax, 11.9, 4.58, 2.35, 1.0, "Paper-facing ontology", "80 canonical risk concepts\nfamilies, names, provenance", "#FFFFFF", RED, lw=2.0)

    for sx, ex in [(2.2, 2.65), (4.5, 4.95), (6.85, 7.3), (9.15, 9.6), (11.35, 11.9)]:
        arrow(ax, (sx, 5.08), (ex, 5.08), color=LINE)

    # Middle lane: model interface.
    mid_y = 2.78
    rounded_box(ax, 0.65, mid_y, 1.65, 0.76, "Video state", "object features\nx_t", "#FFFFFF", BLUE)
    rounded_box(ax, 2.75, mid_y, 1.75, 0.76, "OFA", "object-focused\naggregation", "#FFFFFF", BLUE)
    rounded_box(ax, 5.0, 2.56, 2.5, 1.16, "Semantic interface", "CBM concept state c_t\ntrainable, named, inspectable", "#FFF0ED", RED, lw=2.4)
    rounded_box(ax, 8.1, mid_y, 1.55, 0.76, "CRS", "global concept\nrisk ranking", "#FFFFFF", ORANGE)
    rounded_box(ax, 10.05, mid_y, 1.65, 0.76, "CGTA", "concept-delta\ntemporal attention", "#FFFFFF", GREEN)
    rounded_box(ax, 12.1, mid_y, 1.7, 0.76, "TSD", "future CLIP\nbootstrap", "#FFFFFF", PURPLE)

    for sx, ex in [(2.3, 2.75), (4.5, 5.0), (7.5, 8.1), (9.65, 10.05), (11.7, 12.1)]:
        arrow(ax, (sx, 3.16), (ex, 3.16), color=LINE)

    arrow(ax, (13.05, 4.58), (6.25, 3.72), color=RED, lw=1.6, rad=0.12)
    label(ax, 9.7, 4.05, "ontology becomes the bottleneck vocabulary", RED, 8.2, "bold")

    # Bottom lane: decision and audit.
    low_y = 0.82
    rounded_box(ax, 4.7, low_y, 2.1, 0.76, "Concept-augmented state", "[h_t || c_t]", "#FFFFFF", RED, lw=1.9)
    rounded_box(ax, 7.35, low_y, 1.75, 0.76, "CAAC", "alert timing\nconsumer", "#FFFFFF", GREEN, lw=1.9)
    rounded_box(ax, 9.65, low_y, 1.85, 0.76, "Outputs", "risk score +\nwarning time", "#FFFFFF", GREEN)
    rounded_box(ax, 12.05, low_y, 2.25, 0.76, "Audit trail", "concept names,\nweights, deltas, cases", "#FFFFFF", TEAL)

    arrow(ax, (6.25, 2.56), (5.75, 1.58), color=RED, lw=1.65)
    arrow(ax, (6.8, 1.2), (7.35, 1.2), color=LINE)
    arrow(ax, (9.1, 1.2), (9.65, 1.2), color=LINE)
    arrow(ax, (11.5, 1.2), (12.05, 1.2), color=LINE)
    arrow(ax, (8.85, 2.78), (12.75, 1.58), color=ORANGE, lw=1.25, rad=-0.12, dashed=True)
    arrow(ax, (10.88, 2.78), (13.05, 1.58), color=GREEN, lw=1.25, rad=-0.08, dashed=True)

    label(ax, 3.1, 2.08, "not a post-hoc caption:\nthe concept vector is in the forward path", RED, 8.1, "bold")
    label(ax, 10.9, 2.06, "audit signals remain attached\nto the same named ontology", TEAL, 8.1, "bold")

    # Small concept chips to make the interface concrete.
    chip(ax, 5.1, 2.05, "merge conflict", RED)
    chip(ax, 6.45, 2.05, "low visibility", RED)
    chip(ax, 7.8, 2.05, "VRU crossing", RED)

    # Legend.
    legend_items = [
        ("language artifact", RED),
        ("temporal model", GREEN),
        ("support signal", PURPLE),
        ("audit output", TEAL),
    ]
    x0 = 0.7
    for i, (text, color) in enumerate(legend_items):
        ax.plot([x0 + i * 2.15, x0 + i * 2.15 + 0.35], [0.18, 0.18], color=color, lw=3)
        ax.text(x0 + i * 2.15 + 0.45, 0.18, text, va="center", ha="left", fontsize=7.8, color=MUTED)

    plt.tight_layout(pad=0.2)
    png = OUT / "insight_framework.png"
    pdf = OUT / "insight_framework.pdf"
    plt.savefig(png, dpi=240, bbox_inches="tight", facecolor=BG)
    plt.savefig(pdf, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

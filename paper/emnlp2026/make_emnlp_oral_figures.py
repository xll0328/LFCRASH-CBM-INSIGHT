#!/usr/bin/env python3
"""Build paper-facing EMNLP figures for the oral-readiness candidate.

The framework figure may use a GPT Image 2 bitmap as a subdued visual base, but
all scientific labels and geometry are drawn deterministically here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageEnhance


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper" / "figures"
OUT_DIR = Path(__file__).resolve().parent / "gpt_image2_framework" / "outputs"

W, H = 2400, 1220


COLORS = {
    "ink": "#1f2937",
    "muted": "#667085",
    "line": "#344054",
    "grid": "#d0d5dd",
    "bg": "#fbfcfd",
    "blue": "#1d70b8",
    "purple": "#7357b8",
    "amber": "#c45f0b",
    "red": "#c9341f",
    "teal": "#008d75",
    "green": "#0b7a53",
}


def latest_image2_base() -> Path | None:
    candidates = sorted(OUT_DIR.glob("*/insight_teaser_00.png"))
    return candidates[-1] if candidates else None


def prep_background(base_path: Path | None) -> Image.Image:
    bg = Image.new("RGB", (W, H), COLORS["bg"])
    if not base_path:
        return bg

    img = Image.open(base_path).convert("RGB")
    scale = max(W / img.width, H / img.height)
    resized = img.resize((int(img.width * scale), int(img.height * scale)))
    left = (resized.width - W) // 2
    top = (resized.height - H) // 2
    crop = resized.crop((left, top, left + W, top + H))
    crop = ImageEnhance.Color(crop).enhance(0.55)
    crop = ImageEnhance.Contrast(crop).enhance(0.72)
    crop = ImageEnhance.Brightness(crop).enhance(1.08)
    return Image.blend(bg, crop, 0.16)


def card(ax, xy, wh, edge, title, lines, fill="#ffffff", lw=2.6, title_size=8.6):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=lw,
        edgecolor=edge,
        facecolor=fill,
        alpha=0.96,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + 0.18 * h,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=edge,
    )
    for i, line in enumerate(lines):
        ax.text(
            x + w / 2,
            y + h * (0.47 + 0.18 * i),
            line,
            ha="center",
            va="center",
            fontsize=7.2,
            color=COLORS["ink"],
        )
    return patch


def arrow(ax, start, end, color=COLORS["line"], lw=2.8, style="-", rad=0.0, ms=22):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        linestyle=style,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch


def small_label(ax, x, y, text, color=COLORS["muted"], size=6.5, weight="regular"):
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, fontweight=weight)


def build_framework() -> None:
    base = latest_image2_base()
    background = prep_background(base)

    fig, ax = plt.subplots(figsize=(7.2, 2.85), dpi=300)
    ax.imshow(background)
    ax.set_xlim(45, 2225)
    ax.set_ylim(825, 75)
    ax.axis("off")

    # Normalized coordinate helpers.
    def xy(nx, ny):
        return nx * W, ny * H

    # Main stage cards.
    y0, h = 120, 220
    w = 390
    xs = [88, 610, 1132, 1654]
    cards = [
        (xs[0], y0, COLORS["blue"], "video evidence", ["pre-crash frames", "object features o_t"]),
        (xs[1], y0, COLORS["purple"], "risk language", ["risk-first proposals", "canonical ontology", "merge provenance"]),
        (xs[2], y0, COLORS["red"], "CBM state", ["concept vector c_t", "CRS + CGTA + TSD"]),
        (xs[3], y0, COLORS["teal"], "outputs", ["risk score", "alert timing", "audit cases"]),
    ]
    for x, y, c, title, lines in cards:
        card(ax, (x, y), (w, h), c, title, lines, title_size=8.6)

    # Arrows between main stages.
    for i in range(3):
        arrow(ax, (xs[i] + w + 24, y0 + h / 2), (xs[i + 1] - 24, y0 + h / 2))

    # Evidence thumbnails in the first card.
    thumb_y = y0 + 145
    for i in range(4):
        tx = xs[0] + 46 + i * 66
        ax.add_patch(Rectangle((tx, thumb_y), 54, 36, facecolor="#e8eef3", edgecolor="#9aa6b2", linewidth=1.2))
        ax.plot([tx + 5, tx + 49], [thumb_y + 28, thumb_y + 28], color="#667085", linewidth=1.1)
        ax.plot([tx + 10, tx + 27, tx + 48], [thumb_y + 26, thumb_y + 14, thumb_y + 25], color="#7a8f9f", linewidth=1.5)
        ax.add_patch(Rectangle((tx + 18, thumb_y + 21), 17, 7, facecolor="#c45f0b", alpha=0.65, edgecolor="none"))

    # Bottleneck bars.
    bar_x, bar_y = xs[2] + 68, y0 + 135
    bar_vals = [0.88, 0.72, 0.55, 0.36]
    for i, val in enumerate(bar_vals):
        by = bar_y + i * 30
        ax.add_patch(Rectangle((bar_x, by), 210, 13, facecolor="#edf1f5", edgecolor="none"))
        ax.add_patch(Rectangle((bar_x, by), 210 * val, 13, facecolor="#d9700d", alpha=0.85, edgecolor="none"))

    # Output panel details.
    out_x = xs[3] + 84
    ax.plot([out_x, out_x + 210], [y0 + 170, y0 + 132], color=COLORS["teal"], linewidth=3)
    ax.plot([out_x, out_x + 210], [y0 + 198, y0 + 196], color="#98a2b3", linewidth=2)
    ax.plot([xs[3] + 264, xs[3] + 264], [y0 + 126, y0 + 216], color=COLORS["red"], linewidth=3)
    small_label(ax, xs[3] + 264, y0 + 228, "alert", color=COLORS["red"], size=6.2, weight="bold")

    # Central semantic-interface band.
    band_y = 390
    ax.add_patch(FancyBboxPatch((330, band_y), 1640, 120, boxstyle="round,pad=0.02,rounding_size=18",
                                facecolor="#fff7ed", edgecolor="#d9700d", linewidth=2.4, alpha=0.96))
    ax.text(1150, band_y + 42, "shared concept interface for prediction, timing, and audit",
            ha="center", va="center", fontsize=9.6, fontweight="bold", color="#b54708")
    ax.text(1150, band_y + 84, "ontology construction is evaluated as a modeling choice",
            ha="center", va="center", fontsize=7.2, color=COLORS["ink"])
    arrow(ax, (755, y0 + h + 20), (755, band_y - 8), color="#d9700d", lw=2.5, style="--", ms=18)
    arrow(ax, (1258, y0 + h + 20), (1258, band_y - 8), color="#d9700d", lw=2.5, style="--", ms=18)
    arrow(ax, (1725, y0 + h + 20), (1725, band_y - 8), color=COLORS["teal"], lw=2.5, style="--", ms=18)

    # Bottom audit/evidence cards.
    y2, h2, w2 = 590, 190, 410
    lower = [
        (230, y2, COLORS["purple"], "governance", ["family balance", "merge provenance", "review notes"]),
        (760, y2, COLORS["red"], "ontology tests", ["historical full", "risk-core manual", "discovered ontology"]),
        (1290, y2, COLORS["teal"], "audit evidence", ["concept rankings", "case traces", "alert trajectory"]),
    ]
    for x, y, c, title, lines in lower:
        card(ax, (x, y), (w2, h2), c, title, lines, fill="#ffffff", lw=2.2, title_size=8.0)
    arrow(ax, (800, band_y + 124), (438, y2 - 20), color=COLORS["purple"], lw=2.1, style="--", rad=0.18, ms=17)
    arrow(ax, (1115, band_y + 124), (965, y2 - 20), color=COLORS["red"], lw=2.1, style="--", rad=0.0, ms=17)
    arrow(ax, (1610, band_y + 124), (1495, y2 - 20), color=COLORS["teal"], lw=2.1, style="--", rad=-0.12, ms=17)

    png = FIG_DIR / "emnlp_fig2_framework_image2_overlay.png"
    pdf = FIG_DIR / "emnlp_fig2_framework_image2_overlay.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"wrote {png.relative_to(ROOT)}")
    print(f"wrote {pdf.relative_to(ROOT)}")
    if base:
        promoted = FIG_DIR / "emnlp_gpt_image2_framework_base.png"
        Image.open(base).save(promoted)
        print(f"base {base.relative_to(ROOT)} -> {promoted.relative_to(ROOT)}")
    else:
        print("base none")


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    build_framework()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

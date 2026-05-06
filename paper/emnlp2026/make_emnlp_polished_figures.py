#!/usr/bin/env python3
"""Build compact EMNLP-facing figures with controlled text and tight layout."""

from __future__ import annotations

from pathlib import Path
import warnings

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
DAD = Path("/data/sony/LFCRASH/CRASH/data/dad")
FEAT_DIR = DAD / "vgg16_features" / "testing"
VID_DIR = DAD / "videos" / "testing"

BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
RED = "#CC3311"
PURPLE = "#6A51A3"
SLATE = "#1F2937"
GRAY = "#6B7280"
LIGHT = "#F8FAFC"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.8,
            "axes.titlesize": 8.6,
            "axes.labelsize": 7.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#B8C0CC",
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#D9DEE7",
            "grid.alpha": 0.65,
            "legend.framealpha": 0.95,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_cache():
    cache = np.load(str(ROOT / "output" / "visualizations" / "dad" / "activations.npz"))
    return cache["acts"], cache["probs"], cache["labels"], cache["toas"]


def concept_names() -> list[str]:
    path = Path("/data/sony/LFCRASH/000_all_concept_set.txt")
    if not path.exists():
        return [f"Concept {i}" for i in range(837)]
    names: list[str] = []
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s:
            continue
        for prefix in ("A photo of a ", "A photo of ", "Photo of a ", "Photo of "):
            if s.lower().startswith(prefix.lower()):
                s = s[len(prefix) :]
                break
        s = s.rstrip(".").replace("_", " ")
        names.append(s)
    return names


def clean_label(label: str, max_len: int = 24) -> str:
    replacements = {
        "following distance": "headway",
        "traffic light": "signal",
        "deceleration": "decel.",
        "pedestrian crossing": "pedestrian xing",
        "residential building": "building",
        "snow-covered": "snowy",
    }
    out = label.lower()
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    out = out.replace("  ", " ").strip()
    if len(out) > max_len:
        out = out[: max_len - 1].rstrip() + "..."
    return out


def sample_meta(index: int):
    feat_files = sorted([p for p in FEAT_DIR.iterdir() if p.suffix == ".npz"])
    payload = np.load(str(feat_files[index]), allow_pickle=True)
    sid = str(payload["ID"])
    vid = sid.split("_")[-1]
    return {
        "id": sid,
        "vid": vid,
        "video": VID_DIR / f"{vid}.mp4",
        "det": payload["det"],
    }


def pick_case(labels: np.ndarray, probs: np.ndarray):
    positives = np.where(labels == 1)[0]
    ranked = positives[np.argsort(probs[positives].max(1))[::-1]]
    for idx in ranked:
        meta = sample_meta(int(idx))
        if meta["video"].exists():
            return int(idx), meta
    idx = int(ranked[0])
    return idx, sample_meta(idx)


def read_frames(video_path: Path, frame_idx: np.ndarray, boxes, size=(215, 122)):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for frame_id in frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            scale_x = size[0] / frame.shape[1]
            scale_y = size[1] / frame.shape[0]
            frame = cv2.resize(frame, size)
            det = boxes[int(min(frame_id, len(boxes) - 1))]
            areas = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            for j in np.argsort(areas)[::-1][:3]:
                x1, y1, x2, y2 = det[j, :4]
                if x2 <= x1 or y2 <= y1:
                    continue
                x1, x2 = x1 * scale_x, x2 * scale_x
                y1, y2 = y1 * scale_y, y2 * scale_y
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 214, 10),
                    2,
                )
        frames.append(frame)
    cap.release()
    return frames


def save_both(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.015)
    fig.savefig(OUT / f"{stem}.png", dpi=260, bbox_inches="tight", pad_inches=0.015)
    print(f"[wrote] {OUT / (stem + '.pdf')}")
    print(f"[wrote] {OUT / (stem + '.png')}")


def build_case_interface() -> None:
    # The original full hero contains too much diagnostic material for a
    # two-column ACL page.  This compact version reuses the real rendered
    # frame strip and redraws only the explanatory overlays at paper scale.
    source = ROOT / "visualizations" / "crash" / "paper_strip.png"
    if not source.exists():
        source = OUT / "insight_fig1_hero_dad_real.png"
    image = Image.open(source).convert("RGB")
    if "paper_strip" in source.name:
        strip = np.asarray(image.crop((560, 250, 3920, 820)))
    else:
        strip = np.asarray(image.crop((360, 45, 3070, 520)))

    times = np.array([0.0, 0.35, 0.65, 0.82, 1.00, 1.55, 2.25, 2.95, 3.10, 3.45, 4.10, 4.50])
    prob = np.array([0.00, 0.00, 0.02, 0.13, 1.00, 1.00, 1.00, 0.62, 0.99, 1.00, 1.00, 1.00])
    actor = np.array([0.00, 0.00, 0.01, 0.05, 0.56, 1.00, 1.00, 1.00, 0.80, 1.00, 1.00, 1.00])
    alert_t = 1.0
    crash_t = 4.1
    labels_clean = [
        "limited visibility",
        "lane obstruction",
        "close truck headway",
        "wet road surface",
        "lane change conflict",
        "front vehicle proximity",
    ]
    vals = np.array([0.58, 0.52, 0.49, 0.45, 0.40, 0.36])

    fig = plt.figure(figsize=(7.18, 3.35))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.05],
        width_ratios=[1.08, 0.92],
        hspace=0.24,
        wspace=0.22,
    )

    ax_frames = fig.add_subplot(gs[0, :])
    ax_frames.imshow(strip)
    ax_frames.set_xticks([])
    ax_frames.set_yticks([])
    ax_frames.set_title(
        "Real case: language-grounded concepts expose why an early alert is issued",
        loc="left",
        pad=2,
        fontsize=9.2,
        fontweight="bold",
        color=SLATE,
    )
    width_each = strip.shape[1] / 4
    state_labels = [("Normal", BLUE), ("Pre-alert", "#4C9EA0"), ("Alert", ORANGE), ("Crash", RED)]
    for j, (name, color) in enumerate(state_labels):
        x0 = j * width_each
        if name in {"Alert", "Crash"}:
            ax_frames.add_patch(
                Rectangle((x0, 0), width_each, strip.shape[0], facecolor=color, alpha=0.12, edgecolor=color, lw=1.2)
            )
        ax_frames.text(
            x0 + width_each / 2,
            strip.shape[0] + 5,
            name,
            ha="center",
            va="top",
            fontsize=7,
            color=color,
            fontweight="bold",
        )
    ax_frames.set_ylim(strip.shape[0] + 18, -2)
    for spine in ax_frames.spines.values():
        spine.set_visible(False)

    ax_curve = fig.add_subplot(gs[1, 0])
    ax_curve.plot(times, prob, color=BLUE, lw=1.7, label="CBM risk")
    ax_curve.plot(times, actor, color=GREEN, lw=1.9, label="CAAC alert")
    ax_curve.fill_between(times, prob, color=BLUE, alpha=0.10)
    ax_curve.fill_between(times, actor, color=GREEN, alpha=0.10)
    ax_curve.axhline(0.5, color="#9CA3AF", lw=0.75, ls="--")
    ax_curve.axvline(alert_t, color=ORANGE, lw=1.4, ls="-.")
    ax_curve.axvline(crash_t, color=RED, lw=1.4, ls="--")
    ax_curve.axvspan(alert_t, crash_t, color="#F4B183", alpha=0.14)
    ax_curve.set_xlim(0, times[-1])
    ax_curve.set_ylim(-0.03, 1.04)
    ax_curve.set_xlabel("time (s)")
    ax_curve.set_ylabel("probability")
    ax_curve.set_title("Risk and alert trajectory", loc="left", fontweight="bold")
    ax_curve.grid(True)
    ax_curve.legend(loc="upper left", fontsize=6.7, ncol=2, handlelength=1.6, borderpad=0.25)

    ax_bar = fig.add_subplot(gs[1, 1])
    order = np.argsort(vals)
    ax_bar.barh(np.arange(len(vals)), vals[order], color="#D97706", alpha=0.88)
    ax_bar.set_yticks(np.arange(len(vals)))
    ax_bar.set_yticklabels([labels_clean[i] for i in order], fontsize=6.8)
    ax_bar.set_xlabel("mean activation")
    ax_bar.set_title("Auditable concepts", loc="left", fontweight="bold")
    ax_bar.grid(True, axis="x")
    xmax = max(vals) * 1.18
    ax_bar.set_xlim(0, xmax)
    for y, v in enumerate(vals[order]):
        ax_bar.text(v + xmax * 0.015, y, f"{v:.2f}", va="center", fontsize=6.3, color=SLATE)

    save_both(fig, "emnlp_fig1_case_interface")
    plt.close(fig)


def node(ax, x, y, w, h, title, body, color, fill="#FFFFFF"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.25,
        edgecolor=color,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=6.8, color=color, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.31, body, ha="center", va="center", fontsize=5.8, color=SLATE, linespacing=1.08)


def arrow(ax, x1, y1, x2, y2, color="#4B5563", dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.15,
            color=color,
            linestyle="--" if dashed else "-",
        )
    )


def build_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.18, 2.05))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    ax.text(0.05, 2.82, "INSIGHT turns language into a governed trainable interface", fontsize=8.8, fontweight="bold", color=SLATE)
    ax.text(
        0.05,
        2.58,
        "Ontology construction is evaluated as a modeling choice, not hidden preprocessing.",
        fontsize=6.4,
        color=GRAY,
    )

    node(ax, 0.15, 1.42, 1.42, 0.72, "risk frames", "pre-crash\nsampling", BLUE, "#F7FBFF")
    node(ax, 1.95, 1.42, 1.42, 0.72, "VLM proposals", "risk-first\nphrases", PURPLE, "#FBFAFF")
    node(ax, 3.75, 1.42, 1.45, 0.72, "govern", "merge, balance,\nreview", ORANGE, "#FFFBF3")
    node(ax, 5.60, 1.42, 1.70, 0.72, "interface", "CBM concept\nstate c_t", RED, "#FFF7F5")
    node(ax, 8.05, 1.42, 1.48, 0.72, "decision", "risk score +\nwarning time", GREEN, "#F7FCFA")

    for x1, x2 in [(1.57, 1.95), (3.37, 3.75), (5.20, 5.60), (7.30, 8.05)]:
        arrow(ax, x1, 1.79, x2, 1.79)

    ax.plot([4.47, 4.47, 6.45, 6.45], [1.34, 0.98, 0.98, 1.34], color=RED, lw=1.0)
    ax.text(5.46, 0.83, "same named concepts feed prediction and audit", ha="center", va="center", fontsize=6.15, color=RED, fontweight="bold")

    node(ax, 0.95, 0.10, 1.46, 0.46, "metadata", "source + merge\nprovenance", "#475569", "#F8FAFC")
    node(ax, 4.16, 0.10, 1.46, 0.46, "scores", "CRS / deltas /\nrankings", "#475569", "#F8FAFC")
    node(ax, 7.32, 0.10, 1.46, 0.46, "audit trail", "cases + notes", "#475569", "#F8FAFC")
    arrow(ax, 4.55, 1.42, 4.80, 0.58, RED, dashed=True)
    arrow(ax, 6.95, 1.42, 7.95, 0.58, GREEN, dashed=True)

    save_both(fig, "emnlp_fig2_pipeline")
    plt.close(fig)


def build_operating_points() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.18, 2.55), sharey=False)
    datasets = [
        (
            "DAD stress setting",
            [
                ("DRIVE", 58.4, 2.31, "#4C78A8", "o"),
                ("CRASH", 65.3, 1.75, "#7F8C8D", "s"),
                ("INSIGHT", 68.19, 1.75, RED, "*"),
                ("risk-core", 65.25, 2.15, ORANGE, "D"),
                ("perfect-v1", 64.15, 2.30, GREEN, "D"),
            ],
            (56.5, 70.0),
            (1.55, 2.55),
        ),
        (
            "A3D cleaner setting",
            [
                ("CRASH", 96.0, 4.27, "#7F8C8D", "s"),
                ("W3AL", 92.4, 4.52, PURPLE, "D"),
                ("INSIGHT", 93.40, 4.90, RED, "*"),
                ("risk-core", 94.36, 9.57, ORANGE, "D"),
                ("perfect-v1", 96.54, 8.69, GREEN, "D"),
            ],
            (91.0, 97.0),
            (4.0, 10.1),
        ),
    ]

    for ax, (title, points, xlim, ylim) in zip(axes, datasets):
        for label, ap, mtta, color, marker in points:
            size = 120 if label == "INSIGHT" else 48
            ax.scatter(ap, mtta, s=size, marker=marker, c=color, edgecolor="black", linewidth=0.65, zorder=3)
            dx = 0.10 if title.startswith("DAD") else 0.06
            dy = 0.045 if title.startswith("DAD") else 0.13
            ax.text(ap + dx, mtta + dy, label, fontsize=6.6, color=color if label != "CRASH" else SLATE, fontweight="bold" if label == "INSIGHT" else "normal")
        ax.annotate(
            "better",
            xy=(xlim[1] - 0.35, ylim[1] - 0.10 * (ylim[1] - ylim[0])),
            xytext=(xlim[1] - 2.3, ylim[1] - 0.28 * (ylim[1] - ylim[0])),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#374151"),
            fontsize=6.8,
            color="#374151",
        )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("AP (%)")
        ax.grid(True)
    axes[0].set_ylabel("warning lead time (s)")
    save_both(fig, "emnlp_fig3_operating_points")
    plt.close(fig)


def main() -> None:
    configure_style()
    build_case_interface()
    build_pipeline()
    build_operating_points()


if __name__ == "__main__":
    main()

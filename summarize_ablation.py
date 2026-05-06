#!/usr/bin/env python3
"""
整理 phase2_ablation 所有结果，以 ablation flag 为准（忽略 tag 字段错位问题），
输出论文可用的消融汇总表。
"""
import json
from pathlib import Path

OUTPUT_DIR = Path("output/phase2_ablation")

# 每个子目录对应的「期望」配置（从目录名推断，不信任 tag 字段）
DIR_TO_CONFIG = {
    "a3d_full":       {"dataset": "a3d",   "no_cbm": False, "no_align": False, "no_sparse": False, "no_recon": False},
    "a3d_no_align":   {"dataset": "a3d",   "no_cbm": False, "no_align": True,  "no_sparse": False, "no_recon": False},
    "a3d_no_cbm":     {"dataset": "a3d",   "no_cbm": True,  "no_align": False, "no_sparse": False, "no_recon": False},
    "a3d_no_recon":   {"dataset": "a3d",   "no_cbm": False, "no_align": False, "no_sparse": False, "no_recon": True},
    "a3d_no_sparse":  {"dataset": "a3d",   "no_cbm": False, "no_align": False, "no_sparse": True,  "no_recon": False},
    "crash_full":     {"dataset": "crash", "no_cbm": False, "no_align": False, "no_sparse": False, "no_recon": False},
    "crash_no_align": {"dataset": "crash", "no_cbm": False, "no_align": True,  "no_sparse": False, "no_recon": False},
    "crash_no_cbm":   {"dataset": "crash", "no_cbm": True,  "no_align": False, "no_sparse": False, "no_recon": False},
    "crash_no_recon": {"dataset": "crash", "no_cbm": False, "no_align": False, "no_sparse": False, "no_recon": True},
    "crash_no_sparse":{"dataset": "crash", "no_cbm": False, "no_align": False, "no_sparse": True,  "no_recon": False},
    "dad_full":       {"dataset": "dad",   "no_cbm": False, "no_align": False, "no_sparse": False, "no_recon": False},
    "dad_no_align":   {"dataset": "dad",   "no_cbm": False, "no_align": True,  "no_sparse": False, "no_recon": False},
    "dad_no_cbm":     {"dataset": "dad",   "no_cbm": True,  "no_align": False, "no_sparse": False, "no_recon": False},
    "dad_no_recon":   {"dataset": "dad",   "no_cbm": False, "no_align": False, "no_sparse": False, "no_recon": True},
    "dad_no_sparse":  {"dataset": "dad",   "no_cbm": False, "no_align": False, "no_sparse": True,  "no_recon": False},
}

CONFIG_LABEL = {
    (False, False, False, False): "Full",
    (True,  False, False, False): "w/o CBM",
    (False, True,  False, False): "w/o Align",
    (False, False, True,  False): "w/o Sparse",
    (False, False, False, True):  "w/o Recon",
}

results = []
missing = []

for dirname, expected in DIR_TO_CONFIG.items():
    rfile = OUTPUT_DIR / dirname / "results.json"
    if not rfile.exists() or rfile.stat().st_size == 0:
        missing.append(dirname)
        continue

    # 有些文件有多个 JSON 对象拼接，取最后一个有效的
    raw = rfile.read_text().strip()
    # 分割多个 JSON 对象
    objs = []
    depth = 0
    start = None
    for i, ch in enumerate(raw):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objs.append(json.loads(raw[start:i+1]))
                except json.JSONDecodeError:
                    pass
                start = None

    if not objs:
        missing.append(dirname)
        continue

    # 取最后一个对象（最新写入的）
    data = objs[-1]

    # 验证 ablation flag 是否与目录名期望一致
    ablation = data.get("ablation", {})
    actual_flags = (
        ablation.get("no_cbm", False),
        ablation.get("no_align", False),
        ablation.get("no_sparse", False),
        ablation.get("no_recon", False),
    )
    expected_flags = (
        expected["no_cbm"],
        expected["no_align"],
        expected["no_sparse"],
        expected["no_recon"],
    )

    flag_match = actual_flags == expected_flags
    label = CONFIG_LABEL.get(actual_flags, str(actual_flags))

    results.append({
        "dir": dirname,
        "dataset": data.get("dataset", expected["dataset"]),
        "label": label,
        "AP": data.get("AP", data.get("best_ap", float("nan"))),
        "mTTA": data.get("mTTA", float("nan")),
        "TTA_R80": data.get("TTA_R80", float("nan")),
        "P_R80": data.get("P_R80", float("nan")),
        "best_epoch": data.get("best_epoch", "?"),
        "flag_match": flag_match,
        "actual_flags": actual_flags,
        "expected_flags": expected_flags,
    })

# 按 dataset 和 label 排序
ORDER = {"Full": 0, "w/o CBM": 1, "w/o Align": 2, "w/o Sparse": 3, "w/o Recon": 4}
results.sort(key=lambda x: (x["dataset"], ORDER.get(x["label"], 99)))

print("=" * 90)
print(f"{'Dataset':<8} {'Config':<12} {'AP':>7} {'mTTA':>7} {'TTA_R80':>9} {'P_R80':>7} {'Ep':>4}  {'Flag OK'}")
print("=" * 90)

current_ds = None
for r in results:
    if r["dataset"] != current_ds:
        if current_ds is not None:
            print("-" * 90)
        current_ds = r["dataset"]
    flag_ok = "✓" if r["flag_match"] else f"✗ actual={r['actual_flags']}"
    print(f"{r['dataset']:<8} {r['label']:<12} {r['AP']:>7.4f} {r['mTTA']:>7.4f} {r['TTA_R80']:>9.4f} {r['P_R80']:>7.4f} {r['best_epoch']:>4}  {flag_ok}")

print("=" * 90)

if missing:
    print(f"\n⚠ 缺失/空文件 ({len(missing)})：{', '.join(missing)}")
else:
    print("\n✓ 所有结果文件齐全")

# 输出 LaTeX 表格
print("\n--- LaTeX 消融表 ---")
print(r"\begin{table}[h]")
print(r"\centering")
print(r"\begin{tabular}{llcccc}")
print(r"\toprule")
print(r"Dataset & Configuration & AP & mTTA & TTA@R80 & P@R80 \\")
print(r"\midrule")
current_ds = None
for r in results:
    if r["dataset"] != current_ds:
        if current_ds is not None:
            print(r"\midrule")
        current_ds = r["dataset"]
    ds_str = r["dataset"].upper() if r["label"] == "Full" else ""
    bold = r["label"] == "Full"
    def fmt(v, b):
        return f"\\textbf{{{v:.4f}}}" if b else f"{v:.4f}"
    print(f"{ds_str} & {r['label']} & {fmt(r['AP'],bold)} & {fmt(r['mTTA'],bold)} & {fmt(r['TTA_R80'],bold)} & {fmt(r['P_R80'],bold)} \\\\")
print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\caption{Ablation study on LFCRASH-CBM. Full model vs.~component removal.}")
print(r"\label{tab:ablation}")
print(r"\end{table}")

"""
Result table generation.

Modes (--mode):
  cmas   : c-MAS val (480 samples) — fusion method comparison table
  final  : Final GT vs c-MAS comparison table

Data in these tables is hardcoded from experiment results.
Edit the data dicts below to update values.

Usage:
    python make_table.py --mode cmas
    python make_table.py --mode final
    python make_table.py --mode cmas --no_plot   # text table only
"""

import argparse
import numpy as np

VIS_DIR = "view_attn_out/vis"

COMBOS   = ["C1\n[0°]", "C2\n[0°,±30°]", "C3\n[0°,±45°]", "C4\n[0°,±90°]", "C8\n[All 7]"]
METHODS  = ["Raw Avg", "Logit-Avg", "Logit-Concat", "Feat-Avg", "Feat-Concat"]
COLORS   = ['#b0b0b0', '#4c8dc2', '#1f6fa8', '#2ca02c', '#17becf']

# ── Experiment results (edit here to update) ──────────────────────────────
# 실험 2: GT 데이터 기반 최종 통합 결과
GT_DATA = {
    "Raw Avg":      [50.77, 51.83, 51.87, 50.98, 51.90],
    "Logit-Avg":    [68.84, 71.04, 71.81, 71.72, 72.52],
    "Logit-Concat": [68.54, 71.64, 72.60, 73.45, 73.87],
    "Feat-Avg":     [73.47, 75.43, 76.12, 76.18, 76.82],
    "Feat-Concat":  [73.23, 75.93, 76.44, 77.27, 77.84],
}

# 실험 2: c-MAS 데이터 기반 최종 통합 결과 (GT-trained models, zero-shot)
CMAS_DATA = {
    "Raw Avg":      [40.21, 39.79, 40.00, 37.71, 40.21],
    "Logit-Avg":    [57.92, 57.50, 56.67, 50.42, 53.75],
    "Logit-Concat": [56.46, 56.67, 53.96, 52.71, 53.33],
    "Feat-Avg":     [59.58, 60.42, 59.38, 50.00, 55.83],
    "Feat-Concat":  [60.21, 56.88, 55.42, 51.04, 53.96],
}

# 추가 실험 1: GT-trained MLP + c-MAS views (zero-shot inference)
GT0_CMAS_DATA = {
    "Raw Avg":      [49.38, 45.21, 45.00, 43.12, 41.88],
    "Logit-Avg":    [70.21, 67.50, 66.67, 57.50, 58.13],
    "Logit-Concat": [70.21, 56.67, 58.75, 54.58, 53.96],
    "Feat-Avg":     [74.79, 74.38, 72.08, 62.50, 60.62],
    "Feat-Concat":  [73.75, 62.71, 63.75, 62.92, 58.33],
}

# 추가 실험 2: GT+cMAS mixed training
MIXED_DATA = {
    "Logit-Avg":    [44.17, 28.33, 28.12, 22.29, 14.79],
    "Logit-Concat": [43.12, 33.75, 31.46, 26.88, 32.71],
    "Feat-Avg":     [68.12, 64.58, 62.29, 58.13, 50.62],
    "Feat-Concat":  [69.17, 60.62, 55.83, 55.21, 39.58],
}

COMBO_LABELS_SHORT = ["C1 [0°]", "C2 [0°,±30°]", "C3 [0°,±45°]", "C4 [0°,±90°]", "C8 [All 7]"]


# ── Text tables ───────────────────────────────────────────────────────────
def print_table(data, title, methods=None):
    if methods is None:
        methods = list(data.keys())
    print(f"\n{title}")
    print("=" * 78)
    print(f"  {'Combo':<14} " + "  ".join(f"{m:>13}" for m in methods))
    print("=" * 78)
    for i, lbl in enumerate(COMBO_LABELS_SHORT):
        row = [data[m][i] for m in methods]
        best = max(row)
        cells = [f"**{v:6.2f}%**" if v == best else f"  {v:6.2f}%  " for v in row]
        print(f"  {lbl:<14} " + "  ".join(cells))
    print("=" * 78)


def run_cmas(no_plot=False):
    print_table(CMAS_DATA, "c-MAS val (480 samples) — Top-1 Accuracy (%)")

    if no_plot:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os
    os.makedirs(VIS_DIR, exist_ok=True)

    x = np.arange(len(COMBOS))
    w = 0.75 / len(METHODS)
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (method, color) in enumerate(zip(METHODS, COLORS)):
        offset = (i - len(METHODS)/2 + 0.5) * w
        vals   = CMAS_DATA[method]
        bars   = ax.bar(x + offset, vals, w, label=method, color=color,
                        edgecolor='white', linewidth=0.5)
        for ci, (bar, v) in enumerate(zip(bars, vals)):
            best_in_combo = max(CMAS_DATA[m][ci] for m in METHODS)
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=7.5,
                    fontweight='bold' if v == best_in_combo else 'normal')
            if v == best_in_combo:
                ax.bar(x[ci] + offset, v, w, color=color, edgecolor='gold', linewidth=2.0)
    ax.set_xticks(x); ax.set_xticklabels(COMBOS, fontsize=10)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=11)
    ax.set_ylim(30, 70)
    ax.set_title('c-MAS val (480 samples) — Fusion Method Comparison\n'
                 '(GT-trained models, zero-shot on c-MAS)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.axhline(40.21, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = f"{VIS_DIR}/cmas_combo_table.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"\nSaved: {out}")


def run_final(no_plot=False):
    # GT vs c-MAS side-by-side
    print("\n실험 2: GT 데이터 기반 결과")
    print_table(GT_DATA, "GT Projection (16,487 val samples)")
    print("\n실험 2: c-MAS 데이터 기반 결과 (zero-shot)")
    print_table(CMAS_DATA, "c-MAS Generated (480 val samples)")

    methods_diff = ["Raw Avg", "Logit-Avg", "Logit-Concat", "Feat-Avg", "Feat-Concat"]
    print("\n성능 갭 (c-MAS - GT)")
    print("=" * 78)
    print(f"  {'Combo':<14} " + "  ".join(f"{m:>13}" for m in methods_diff))
    print("=" * 78)
    for i, lbl in enumerate(COMBO_LABELS_SHORT):
        gaps = [CMAS_DATA[m][i] - GT_DATA[m][i] for m in methods_diff]
        cells = [f"  {g:+6.2f}%  " for g in gaps]
        print(f"  {lbl:<14} " + "  ".join(cells))
    print("=" * 78)

    print("\n추가 실험 1: GT 0° + c-MAS 뷰 (GT-trained, zero-shot)")
    print_table(GT0_CMAS_DATA, "GT 0° + c-MAS Views")
    print("\n추가 실험 2: GT+cMAS 혼합 학습")
    print_table(MIXED_DATA, "Mixed Training",
                methods=["Logit-Avg", "Logit-Concat", "Feat-Avg", "Feat-Concat"])

    if no_plot:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import os
    os.makedirs(VIS_DIR, exist_ok=True)

    # Side-by-side GT vs c-MAS bar chart (best-combo per method)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle("GT Projection vs c-MAS Generated\n(NTU60 xsub_val, GT-trained models)",
                 fontsize=13, fontweight='bold')

    for ax, data, title, n in zip(
        axes,
        [GT_DATA, CMAS_DATA],
        ["GT Kinect-25 Projection\n(16,487 val samples)",
         "c-MAS Generated\n(480 val samples, zero-shot)"],
        [16487, 480]
    ):
        for ci, lbl in enumerate(COMBO_LABELS_SHORT):
            x = np.arange(len(METHODS))
            vals = [data[m][ci] for m in METHODS]
            bars = ax.bar(x + ci * (len(METHODS) + 1), vals, color=COLORS,
                          width=0.7, edgecolor='white')
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Top-1 Accuracy (%)')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    patches = [mpatches.Patch(color=c, label=m) for c, m in zip(COLORS, METHODS)]
    fig.legend(handles=patches, loc='lower center', ncol=5,
               bbox_to_anchor=(0.5, -0.04), fontsize=9)
    plt.tight_layout()
    out = f"{VIS_DIR}/final_comparison.png"
    plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
    print(f"\nSaved: {out}")


# ── Entry point ───────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['cmas', 'final'],
                        help='cmas | final')
    parser.add_argument('--no_plot', action='store_true',
                        help='Print text table only, skip saving figure')
    args = parser.parse_args()

    {'cmas': run_cmas, 'final': run_final}[args.mode](no_plot=args.no_plot)


if __name__ == '__main__':
    main()

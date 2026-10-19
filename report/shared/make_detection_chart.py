#!/usr/bin/env python
"""Generate figures/detection_{ood,id}.png from exported_probes/*_detection.csv.

One dot per (probe, held-out set) with its bootstrap 95% CI; chance at 0.5.
Colors match safety_probes.viz.PROBE_COLORS so probe identity is consistent
with the ROC and steering figures; marker shapes are a secondary encoding for
colorblind/print safety. Regenerate after any pipeline re-run:

    python report/shared/make_detection_chart.py

Writes figures/ next to this script; copy the PNGs into report/submitted/figures/ or
report/arxiv-v1-upload/figures/ to use them.
"""
import csv, pathlib, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPORT = pathlib.Path(__file__).resolve().parents[2] / "exported_probes"
OUT = pathlib.Path(__file__).resolve().parent / "figures" / "detection_ood.png"  # dir anchor

PROBES = ["MM (last)", "MM (mean)", "LR (last)", "LR (mean)", "LR-attention"]
COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]  # viz.PROBE_COLORS
MARKERS = ["o", "s", "D", "^", "v"]

# (csv slug, eval-name in csv, x label), per output chart
CHARTS = {
    "detection_ood.png": [
        ("deception", "AI Liar (OOD)", "deception\nAI Liar"),
        ("deception", "Roleplaying (OOD)", "deception\nroleplaying"),
        ("high-stakes", "anthropic_hh_balanced (OOD)", "high-stakes\nAnthropic-HH"),
        ("sycophancy", "math (OOD)", "sycophancy\nmath"),
    ],
    "detection_id.png": [
        ("deception", "test (ID)", "deception\nID test"),
        ("high-stakes", "test (ID)", "high-stakes\nID test"),
        ("sycophancy", "test (ID)", "sycophancy\nID test"),
    ],
}

def load(slug):
    rows = list(csv.DictReader((EXPORT / f"{slug}_detection.csv").open()))
    out = {}
    for r in rows:
        m = re.match(r"([\d.]+) \[([\d.]+),([\d.]+)\]", r["AUROC"])
        out[(r["probe"], r["eval"])] = tuple(float(g) for g in m.groups())
    return out

plt.style.use("seaborn-v0_8-darkgrid")
for fname, SETS in CHARTS.items():
    data = {slug: load(slug) for slug in {s for s, _, _ in SETS}}
    fig, ax = plt.subplots(figsize=(6.4, 3.1) if len(SETS) == 4 else (5.0, 3.1))
    w = 0.13  # offset between probes inside a group
    for j, (probe, col, mk) in enumerate(zip(PROBES, COLORS, MARKERS)):
        xs, ys, lo, hi = [], [], [], []
        for i, (slug, ev, _) in enumerate(SETS):
            a, l, h = data[slug][(probe, ev)]
            xs.append(i + (j - 2) * w); ys.append(a); lo.append(a - l); hi.append(h - a)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, color=col, ms=5.5, lw=0,
                    elinewidth=1.3, capsize=2.4, label=probe,
                    markeredgecolor="white", markeredgewidth=0.6)
    ax.axhline(0.5, color="#888", lw=1.1, ls="--", zorder=0)
    ax.annotate("chance", xy=(-0.44, 0.507), fontsize=8.5, color="#666")
    ax.set_xticks(range(len(SETS)))
    ax.set_xticklabels([lbl for _, _, lbl in SETS], fontsize=9.5)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.45, 1.02)
    ax.set_xlim(-0.5, len(SETS) - 0.5)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.92, ncol=2)
    fig.tight_layout()
    out = OUT.parent / fname
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")

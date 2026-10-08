"""
make_figures.py -- figures for the report (CPU only, needs matplotlib).

  training.png   train answer-letter NLL and validation accuracy per step,
                 answer-only vs rationale run; selected checkpoint ringed
  test_tiers.png test accuracy (generative strict, canonical order) per edit
                 tier: baseline vs both fine-tuned runs

  python make_figures.py --figdir figures \
      --run "answer-only=out/ans-s1=logs/ans-s1/train/*.out" \
      --run "rationale=out/rat-s1=logs/rat-s1/train/*.out" \
      --split data/test.jsonl \
      --result "Qwen2.5-VL-7B (base)=runs/baseline/base_q257b.jsonl" \
      --result "+ SFT, answer only=runs/ans-s1/ans-s1-best_cyc4.jsonl" \
      --result "+ SFT, rationale=runs/rat-s1/rat-s1-best_cyc4.jsonl"
"""
import argparse
import glob
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from make_tables import TIERS, load, scores  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BASE = "#a3a29c"                                   # neutral: the baseline
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]         # categorical slots 1-3
STEP = re.compile(r"^step\s+(\d+)/(\d+)\s+ep\s+([\d.]+)\s+loss\s+([\d.]+)"
                  r"(?:\s+letter nll\s+([\d.]+)\s+acc\s+([\d.]+))?")


def train_rows(pattern):
    rows = {}
    for path in sorted(glob.glob(pattern)):
        for line in open(path, errors="replace"):
            m = STEP.match(line.strip())
            if m and m.group(5):
                rows[int(m.group(1))] = float(m.group(5))   # resumed step: last wins
    return sorted(rows.items())


def val_rows(out_dir):
    by = {}
    for l in open(Path(out_dir) / "val_log.jsonl"):
        if l.strip():
            r = json.loads(l)
            by[r["step"]] = r["val_forced_acc"]
    return sorted(by.items())


def style(ax, title, ylabel, xlabel="optimizer step"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def ref_line(ax, y, text):
    ax.axhline(y, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.annotate(text, (1.0, y), xycoords=("axes fraction", "data"),
                xytext=(-4, 3), textcoords="offset points", ha="right",
                va="bottom", color=INK2, fontsize=8)


def fig_training(runs, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    style(a, "Train: NLL of the answer letter (per 10 steps)", "NLL")
    style(b, "Validation accuracy (99 items, letter-balanced)", "accuracy")
    offsets = [(-14, 22), (0, 10)]            # keep the two labels clear of the lines
    for (label, out_dir, logs), color, off in zip(runs, SERIES, offsets):
        tr = train_rows(logs)
        a.plot(*zip(*tr), color=color, linewidth=2, label=label)
        val = val_rows(out_dir)
        b.plot(*zip(*val), color=color, linewidth=2, marker="o", markersize=4,
               label=label)
        bs, bv = max(val, key=lambda x: (x[1], -x[0]))
        b.plot([bs], [bv], marker="o", markersize=13, markerfacecolor="none",
               markeredgecolor=color, markeredgewidth=2)
        b.annotate("selected: step %d, %.3f" % (bs, bv), (bs, bv),
                   xytext=off, textcoords="offset points",
                   ha="right" if off[0] else "center",
                   color=INK, fontsize=8)
    ref_line(a, math.log(4), "chance ln 4 = 1.386")
    ref_line(b, 0.25, "chance 0.25")
    b.set_ylim(0.15, 0.52)
    for ax in (a, b):
        for e in (175, 350):
            ax.axvline(e, color=GRID, linewidth=1.5)
        leg = ax.legend(loc="upper right" if ax is a else "lower right",
                        frameon=False, fontsize=8)
        for t in leg.get_texts():
            t.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def fig_tiers(split, results, path):
    items = {}
    for line in open(split):
        if line.strip():
            it = json.loads(line)
            items[it["qid"]] = it
    cols = ["overall"] + TIERS
    n = {t: sum(it["tier"] == t for it in items.values()) for t in TIERS}
    n["overall"] = len(items)
    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=SURFACE)
    style(ax, "Test accuracy by edit tier (generative, original A-D order, n=%d)"
          % len(items), "accuracy (%)", xlabel="")
    width = 0.8 / len(results)
    colors = [BASE] + SERIES
    for k, ((label, p), color) in enumerate(zip(results, colors)):
        sc = scores(load(p, items), items)
        vals = []
        for c in cols:
            v = [x for q, s in sc.items() if s.get("gen0")
                 and (c == "overall" or items[q]["tier"] == c) for x in s["gen0"]]
            vals.append(100.0 * sum(v) / len(v))
        xs = [i + (k - (len(results) - 1) / 2) * width for i in range(len(cols))]
        ax.bar(xs, vals, width=width - 0.02, color=color, label=label,
               edgecolor=SURFACE, linewidth=1)
        ax.annotate("%.1f" % vals[0], (xs[0], vals[0]), xytext=(0, 2),
                    textcoords="offset points", ha="center", color=INK, fontsize=8)
    ref_line(ax, 25, "chance 25%")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["%s\n(n=%d)" % (c, n[c]) for c in cols])
    ax.set_ylim(0, 70)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--run", action="append", default=[],
                    help="LABEL=OUT_DIR=LOG_GLOB")
    ap.add_argument("--split", default="data/test.jsonl")
    ap.add_argument("--result", action="append", default=[], help="LABEL=PATH")
    args = ap.parse_args()
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    if args.run:
        fig_training([r.split("=", 2) for r in args.run], figdir / "training.png")
    if args.result:
        fig_tiers(args.split, [r.split("=", 1) for r in args.result],
                  figdir / "test_tiers.png")
    print("wrote", sorted(str(p) for p in figdir.glob("*.png")))


if __name__ == "__main__":
    main()

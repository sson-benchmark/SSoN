"""
make_tables.py -- per-tier accuracy tables, several models side by side.

Each run is LABEL=PATH to an eval_local.py result file. Records are restricted
to the items of --split and the last record per qid wins.

By default only the canonical view (original A-D order: the protocol's single
call per item) is used. --all-views adds the 3 cyclic rotations.

Tables:
  gen0  generative strict, canonical view                 (headline)
  gen4  generative strict, mean over 4 cyclic views       (--all-views, only
        when every view was generated)
  fc4   forced choice over the views used
  dis4  distractor pick rate: share of answers that name the planted
        distractor panel (lower is better; uniform guessing = 25%)
Canonical 95% CIs are Wilson intervals over binary item outcomes. Four-view
diagnostics require complete metric coverage of all four views and use the
item bootstrap from report.py. Items, rather than individual views, are
resampled together. Distractor rates require valid picks on all included views.

  python make_tables.py --split data/test.jsonl \
      "Qwen2.5-VL-7B=results/runs/baseline/base_q257b.jsonl" \
      "+SFT answer=results/runs/ans-s1/ans-s1-best_cyc4.jsonl" \
      "+SFT rationale=results/runs/rat-s1/rat-s1-best_cyc4.jsonl"
"""
import argparse
import json
import math
from collections import defaultdict

from common import LETTERS, parse_strict
from report import cluster_boot

TIERS = ["move", "rotate", "mirror", "swap", "delete", "detail"]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load(path, items, all_views=False):
    last = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            last[r["qid"]] = r
    views = defaultdict(dict)                       # src qid -> view -> record
    for r in last.values():
        src = r.get("src_qid", r["qid"])
        if src in items and (all_views or r.get("view", 0) == 0):
            views[src][r.get("view", 0)] = r
    return views


def gen_pick(r):
    if r.get("response") is not None:
        return parse_strict(r["response"])
    return r.get("parsed_answer")


def scores(views, items):
    """per src item: dict of metric -> list of 0/1 over the views it has"""
    out = {}
    for src, vs in views.items():
        it = items[src]
        dis = (it.get("sup") or {}).get("distractor")
        s = defaultdict(list)
        all_gen = all(r.get("response") is not None for r in vs.values())
        four_views = set(vs) == {0, 1, 2, 3}
        for v, r in vs.items():
            g = gen_pick(r)
            if r.get("response") is not None:
                if all_gen and four_views:
                    s["gen4"].append(g == r["gold"])
                if v == 0:
                    s["gen0"].append(g == r["gold"])
            fc = r.get("forced_answer")
            if fc:
                s["fc4"].append(fc == r["gold"])
            pick = g if all_gen else fc
            if dis is not None and pick in LETTERS:
                perm = r.get("perm") or [0, 1, 2, 3]
                s["dis4"].append(perm[LETTERS.index(pick)] == dis)
        out[src] = s
    return out


def cell(vals):
    if not vals:
        return None
    return 100.0 * sum(vals) / len(vals)


def metric_summary(sc, items, metric, all_views=False):
    """Mean and CI with equal weight for every fully measured source item."""
    multi = metric == "gen4" or (all_views and metric != "gen0")
    expected = 4 if multi else 1
    values, by_tier = [], defaultdict(list)
    for src in sorted(sc):
        outcomes = sc[src].get(metric, [])
        if len(outcomes) != expected:
            continue
        if any(v not in (False, True, 0, 1) for v in outcomes):
            raise ValueError(f"Nonbinary {metric} outcome for item {src}")
        mean = sum(outcomes) / expected
        values.append(mean)
        by_tier[items[src]["tier"]].append(mean)
    if not values:
        return None
    ci = cluster_boot(values) if multi else wilson(sum(values), len(values))
    return {"overall": cell(values), "by_tier": dict(by_tier),
            "ci": ci, "n": len(values),
            "ci_method": "item bootstrap" if multi else "Wilson"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/test.jsonl")
    ap.add_argument("--all-views", action="store_true",
                    help="also use the 3 cyclic rotations (diagnostic)")
    ap.add_argument("runs", nargs="+", help="LABEL=PATH")
    args = ap.parse_args()

    items = {}
    for line in open(args.split):
        if line.strip():
            it = json.loads(line)
            items[it["qid"]] = it
    tiers = [t for t in TIERS if any(it["tier"] == t for it in items.values())]
    tiers += sorted({it["tier"] for it in items.values()} - set(tiers))
    n_tier = {t: sum(it["tier"] == t for it in items.values()) for t in tiers}

    runs = []
    for spec in args.runs:
        label, path = spec.split("=", 1)
        sc = scores(load(path, items, args.all_views), items)
        runs.append((label, path, sc))

    names = {"gen0": "Protocol separate_v1: generative strict, original A-D order  (HEADLINE)",
             "gen4": "Diagnostic: generative strict, mean over 4 cyclic views",
             "fc4": "Diagnostic: forced choice (A/B/C/D logits)",
             "dis4": "Diagnostic: distractor pick rate  (lower is better, random = 25%)"}
    wl = max(12, max(len(r[0]) for r in runs) + 2)
    print("split %s   n=%d items   views: %s   chance = 25.00" % (
        args.split, len(items),
        "canonical + 3 cyclic rotations" if args.all_views
        else "canonical only (original A-D order)"))
    for label, path, sc in runs:
        print("  %-*s %s   (%d items scored)" % (wl, label, path, len(sc)))
    for m, title in names.items():
        if not any(any(s.get(m) for s in sc.values()) for _, _, sc in runs):
            continue
        cols = ["Overall"] + ["%s (%d)" % (t, n_tier[t]) for t in tiers]
        w = max(11, max(len(c) for c in cols) + 2)
        print("\n" + title)
        multi = m == "gen4" or (args.all_views and m != "gen0")
        ci_method = "item bootstrap" if multi else "Wilson"
        head = "%-*s" % (wl, "Model") + "".join("%*s" % (w, c) for c in cols) \
            + "   95% %s CI, n items" % ci_method
        print(head)
        print("-" * len(head))
        for label, _, sc in runs:
            summary = metric_summary(sc, items, m, args.all_views)
            if summary is None:
                continue
            row = "%-*s" % (wl, label) + "%*.2f" % (w, summary["overall"])
            for t in tiers:
                v = cell(summary["by_tier"].get(t, []))
                row += "%*s" % (w, "-" if v is None else "%.2f" % v)
            lo, hi = summary["ci"]
            row += "   [%.1f, %.1f]  n=%d" % (100 * lo, 100 * hi, summary["n"])
            print(row)


if __name__ == "__main__":
    main()

"""
report.py -- before/after comparison of two eval_local.py result files.

  python report.py --before runs/baseline/base_q257b.jsonl \
      --after runs/rat-s1/rat-s1-best_cyc4.jsonl --by dataset tier letter

For each run: generative strict accuracy (headline; unparsed = wrong), parse
and truncation rate, format-tolerant diagnostics (fenced / lenient), forced
choice, Wilson 95% CIs and per-breakdown tables, all on the canonical view.
If the runs contain the 4 cyclic views, the augmented accuracy is added with a
95% CI from an item-level bootstrap (the 4 views of an item are not
independent), plus all-4-views-correct and a 4-view majority vote.
Paired: exact McNemar test on the canonical view, and the bootstrap CI of the
per-item difference on the augmented set.
"""
import argparse
import json
import math
import random
from collections import Counter, defaultdict

from common import LETTERS, parse_fenced, parse_lenient, parse_strict, read_jsonl


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar_exact(b, c):
    """b = before-right/after-wrong, c = before-wrong/after-right. Two-sided."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load(path):
    recs = {}
    for r in read_jsonl(path):                      # last write wins, dedup by qid
        recs[r["qid"]] = r
    if not recs:
        raise SystemExit(f"{path}: no records")
    return recs


def split_views(recs):
    """(canonical, {src_qid: {view: rec}}). Records from a run without --views
    have no `view` field and are all canonical."""
    canon, groups = {}, defaultdict(dict)
    for q, r in recs.items():
        v = r.get("view", 0)
        if v == 0:
            canon[q] = r
        groups[r.get("src_qid", q)][v] = r
    return canon, groups


def pred_of(r, metric):
    return r.get("forced_answer") if metric == "forced" else strict_of(r)


def has_metric(r, metric):
    return bool(r.get("forced_answer")) if metric == "forced" \
        else r.get("response") is not None


def cluster_boot(per_item, n_boot=2000, seed=0):
    """95% CI of the mean of per-item accuracies, resampling ITEMS."""
    if not per_item:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(per_item)
    means = sorted(sum(per_item[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def view_stats(groups, metric):
    """Per-item stats over the 4 cyclic views; items missing a view are skipped
    so every item carries the same weight and the letter prior stays flat."""
    out = {}
    for src, vs in groups.items():
        if sorted(vs) != [0, 1, 2, 3] or not all(has_metric(r, metric)
                                                  for r in vs.values()):
            continue
        ok = [pred_of(vs[k], metric) == vs[k]["gold"] for k in range(4)]
        votes = Counter()
        for r in vs.values():
            p = pred_of(r, metric)
            if p in LETTERS:
                votes[r["perm"][LETTERS.index(p)]] += 1      # original panel index
        gold_idx = vs[0]["perm"][LETTERS.index(vs[0]["gold"])]
        top = votes.most_common()
        vote_ok = bool(top) and top[0][0] == gold_idx and \
            (len(top) == 1 or top[1][1] < top[0][1])             # ties count as wrong
        out[src] = {"acc": sum(ok) / 4, "all4": all(ok), "vote": vote_ok,
                    "preds": [pred_of(vs[k], metric) for k in range(4)]}
    return out


def summarize_views(groups, metric, label):
    st = view_stats(groups, metric)
    if not st:
        return None
    n = len(st)
    acc = sum(s["acc"] for s in st.values()) / n
    lo, hi = cluster_boot([s["acc"] for s in st.values()])
    pred = Counter(p for s in st.values() for p in s["preds"])
    print(f"  {label:19s} {acc:.3f}  [{lo:.3f}, {hi:.3f}]   "
          f"<- augmented, {n} items x 4 views, constant predictor = 0.250")
    print(f"    all 4 views right {sum(s['all4'] for s in st.values()) / n:.3f}"
          f"   4-view vote {sum(s['vote'] for s in st.values()) / n:.3f}"
          f"   pred letters {dict(sorted(pred.items(), key=lambda x: str(x[0])))}")
    return st


def strict_of(r):
    return parse_strict(r["response"]) if r.get("response") is not None \
        else r.get("parsed_answer")


def summarize(name, all_recs, by):
    recs, groups = split_views(all_recs)
    n = len(recs)
    has_gen = any(r.get("response") is not None for r in recs.values())
    has_fc = any(r.get("forced_answer") for r in recs.values())
    print(f"\n{'=' * 72}\n== {name}   (canonical n={n}, records={len(all_recs)})")
    print("  -- canonical test set (original panel order) --")

    ok_strict, ok_forced = {}, {}
    if has_gen:
        strict = parsed = fenced = lenient = trunc = 0
        for q, r in recs.items():
            s = strict_of(r)
            ok_strict[q] = s == r["gold"]
            strict += s == r["gold"]
            parsed += s is not None
            fenced += parse_fenced(r["response"]) == r["gold"]
            lenient += parse_lenient(r["response"]) == r["gold"]
            trunc += bool(r.get("truncated"))
        lo, hi = wilson(strict, n)
        print(f"  generative strict   {strict / n:.3f}  [{lo:.3f}, {hi:.3f}]   <- headline")
        print(f"  parse rate          {parsed / n:.3f}")
        print(f"  truncated at cap    {trunc / n:.3f}")
        print(f"  + fenced            {fenced / n:.3f}   <- format-corrected")
        print(f"  + lenient           {lenient / n:.3f}   <- upper bound, diagnostic only")
    if has_fc:
        f_ok = sum(1 for r in recs.values() if r["forced_answer"] == r["gold"])
        for q, r in recs.items():
            ok_forced[q] = r.get("forced_answer") == r["gold"]
        lo, hi = wilson(f_ok, n)
        print(f"  forced choice       {f_ok / n:.3f}  [{lo:.3f}, {hi:.3f}]   <- formatting removed")

    pred_key = "parsed" if has_gen else "forced"
    pred = Counter((strict_of(r) if has_gen else r.get("forced_answer"))
                   for r in recs.values())
    gold = Counter(r["gold"] for r in recs.values())
    print(f"  pred letters ({pred_key}) {dict(sorted(pred.items(), key=lambda x: str(x[0])))}")
    print(f"  gold letters        {dict(sorted(gold.items()))}")
    print(f"  majority-letter floor {max(gold.values()) / n:.3f}   (chance = 0.250)")
    top = Counter({k: v for k, v in pred.items() if k in LETTERS}).most_common(1)
    top_letter, top_n = top[0] if top else (None, 0)
    if top_n / n > 0.6:
        print(f"  !! {top_n / n:.0%} of predictions are {top_letter!r}: on the "
              f"canonical set this accuracy is mostly that letter's share of the "
              f"gold ({gold.get(top_letter, 0) / n:.3f}), not ability. Use the "
              f"augmented number.")

    views = {}
    if any(r.get("view", 0) for r in all_recs.values()):
        print("  -- augmented test set (4 cyclic views per item) --")
        if has_gen:
            views["strict"] = summarize_views(groups, "strict", "generative strict")
        if has_fc:
            views["forced"] = summarize_views(groups, "forced", "forced choice")

    for key in by:
        buckets = defaultdict(lambda: [0, 0, 0])
        for q, r in recs.items():
            k = r["gold"] if key == "letter" else (r.get(key) or "NA")
            b = buckets[str(k)]
            b[2] += 1
            b[0] += ok_strict.get(q, False)
            b[1] += ok_forced.get(q, False)
        print(f"  per {key}:")
        head = f"    {'':24s} {'n':>4s}"
        if has_gen:
            head += f" {'strict':>8s}"
        if has_fc:
            head += f" {'forced':>8s}"
        print(head)
        for k, (s, f, m) in sorted(buckets.items()):
            line = f"    {k:24s} {m:4d}"
            if has_gen:
                line += f" {s / m:8.3f}"
            if has_fc:
                line += f" {f / m:8.3f}"
            print(line)
    return ok_strict, ok_forced, views


def paired_views(label, st_b, st_a, n_boot=2000, seed=0):
    """Delta of augmented accuracy, paired by item, CI by item bootstrap."""
    if not st_b or not st_a:
        return
    common = sorted(set(st_b) & set(st_a))
    if not common:
        return
    d = [st_a[q]["acc"] - st_b[q]["acc"] for q in common]
    lo, hi = cluster_boot(d, n_boot, seed)
    print(f"\n  {label}, augmented 4-view  (n={len(common)} items)")
    print(f"    delta = {sum(d) / len(d):+.3f}   95% item-bootstrap CI "
          f"[{lo:+.3f}, {hi:+.3f}]"
          f"{'   <- excludes 0' if lo > 0 or hi < 0 else '   <- includes 0'}")


def paired(label, ok_b, ok_a):
    common = sorted(set(ok_b) & set(ok_a))
    if not common:
        return
    n01 = sum(1 for q in common if ok_b[q] and not ok_a[q])
    n10 = sum(1 for q in common if not ok_b[q] and ok_a[q])
    p = mcnemar_exact(n01, n10)
    delta = (sum(ok_a[q] for q in common) - sum(ok_b[q] for q in common)) / len(common)
    print(f"\n  {label}  (n={len(common)})")
    print(f"    before-only correct {n01}, after-only correct {n10}")
    print(f"    delta = {delta:+.3f}")
    print(f"    exact McNemar p = {p:.2e}"
          f"{'   <- significant' if p < 0.05 else '   <- NOT significant'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--by", nargs="*", default=["dataset", "tier"],
                    choices=["dataset", "tier", "band", "magnitude", "letter"])
    args = ap.parse_args()

    b, a = load(args.before), load(args.after)
    ok_bs, ok_bf, v_b = summarize("BEFORE  " + args.before, b, args.by)
    ok_as, ok_af, v_a = summarize("AFTER   " + args.after, a, args.by)

    print(f"\n{'=' * 72}\n== PAIRED")
    if ok_bs and ok_as:
        paired("generative strict", ok_bs, ok_as)
    if ok_bf and ok_af:
        paired("forced choice", ok_bf, ok_af)
    for m in ("strict", "forced"):
        paired_views({"strict": "generative strict",
                      "forced": "forced choice"}[m], v_b.get(m), v_a.get(m))


if __name__ == "__main__":
    main()

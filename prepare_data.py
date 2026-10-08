"""
prepare_data.py -- grouped, stratified train/val/test split with pre-declared
acceptance criteria, plus 4x panel-order augmentation of the train set.

  1. GROUPS (label-free): two items share a group if they share a panel image
     (sha256), a declared scene id, or a family id. Groups never straddle splits.
  2. STRATA: dataset x tier; cells with < 10 items fold into "<dataset>|other".
  3. StratifiedGroupKFold(10, shuffle, random_state=seed):
     folds {0,1} -> test, fold {2} -> val, folds 3-9 -> train.
  4. ACCEPTANCE CRITERIA (CRITERIA below, fixed before any seed was drawn).
     Seeds are tried in increasing order; the FIRST seed that satisfies every
     criterion is used. Every rejected seed and its violations are written to
     split_search_log.json.

Outputs (in --out):
  train.jsonl, val.jsonl, test.jsonl   canonical panel order
  train_aug.jsonl                      4 panel orders per train item, odd image
                                       at A, B, C, D exactly once (shuffled)
  split_report.json, split_search_log.json

  python prepare_data.py --manifest data/manifest.jsonl --out data --seed-search 0-30
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from common import LETTERS, build_target

CRITERIA = {
    "min_train_per_dataset": 30,
    "min_val_per_dataset": 8,
    "min_test_per_dataset": 12,
    "min_cell_items": 10,           # (dataset, tier) cells below this are exempt
    "min_test_per_cell": 1,         # every large cell is represented in test ...
    "min_train_per_cell": 3,        # ... and seen in train
    "min_test_per_tier": 5,
    "min_train_per_tier": 30,
    "test_n_range": (180, 220),
    "val_n_range": (80, 120),
    "max_dataset_shift": 0.06,      # |train share - test share| per dataset
    "max_tier_shift": 0.08,         # |train share - test share| per tier
}


# --------------------------------------------------------------------------
# Grouping and strata
# --------------------------------------------------------------------------
class Union:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def build_groups(items):
    u = Union(len(items))
    links = defaultdict(list)
    for i, it in enumerate(items):
        for h in it.get("image_sha256") or []:
            links[("sha", h)].append(i)
        if it.get("scene"):
            links[("scene", it["scene"])].append(i)
        if it.get("family"):
            links[("family", it["family"])].append(i)
    for members in links.values():
        for x in members[1:]:
            u.union(members[0], x)
    return [u.find(i) for i in range(len(items))]


def strata(items, min_cell):
    cell = Counter((it["dataset"], it["tier"]) for it in items)
    return [f"{it['dataset']}|{it['tier']}"
            if cell[(it["dataset"], it["tier"])] >= min_cell
            else f"{it['dataset']}|other" for it in items], cell


# --------------------------------------------------------------------------
# Acceptance criteria -> list of violations (empty = accepted)
# --------------------------------------------------------------------------
def violations(items, tr, va, te, cell, C):
    bad = []
    datasets = sorted({it["dataset"] for it in items})
    tiers = sorted({it["tier"] for it in items})

    def count(idx, **kw):
        return sum(1 for i in idx if all(items[i][k] == v for k, v in kw.items()))

    for d in datasets:
        for idx, key, name in ((tr, "min_train_per_dataset", "train"),
                               (va, "min_val_per_dataset", "val"),
                               (te, "min_test_per_dataset", "test")):
            if count(idx, dataset=d) < C[key]:
                bad.append(f"{name}<{C[key]} for dataset {d}")
    for (d, t), n in sorted(cell.items()):
        if n < C["min_cell_items"]:
            continue
        if count(te, dataset=d, tier=t) < C["min_test_per_cell"]:
            bad.append(f"test has no {d}|{t}")
        if count(tr, dataset=d, tier=t) < C["min_train_per_cell"]:
            bad.append(f"train<{C['min_train_per_cell']} for cell {d}|{t}")
    for t in tiers:
        if count(te, tier=t) < C["min_test_per_tier"]:
            bad.append(f"test<{C['min_test_per_tier']} for tier {t}")
        if count(tr, tier=t) < C["min_train_per_tier"]:
            bad.append(f"train<{C['min_train_per_tier']} for tier {t}")
    for idx, key, name in ((te, "test_n_range", "test"), (va, "val_n_range", "val")):
        lo, hi = C[key]
        if not lo <= len(idx) <= hi:
            bad.append(f"{name} n={len(idx)} outside {C[key]}")
    for key, lim, values in (("dataset", C["max_dataset_shift"], datasets),
                             ("tier", C["max_tier_shift"], tiers)):
        for k in values:
            a = count(tr, **{key: k}) / max(len(tr), 1)
            b = count(te, **{key: k}) / max(len(te), 1)
            if abs(a - b) > lim:
                bad.append(f"{key} shift {k}: {a - b:+.3f} (limit {lim})")
    return bad


# --------------------------------------------------------------------------
# Panel-order augmentation (train only)
# --------------------------------------------------------------------------
def permute(item, order):
    """order[j] = original panel index shown at slot j."""
    out = dict(item)
    for k in ("images", "rel_images", "image_sha256"):
        if item.get(k):
            out[k] = [item[k][i] for i in order]
    out["answer"] = LETTERS[order.index(LETTERS.index(item["answer"]))]
    out["src_qid"] = item["qid"]
    out["perm"] = list(order)
    out["qid"] = f"{item['qid']}#p{''.join(str(i) for i in order)}"
    return out


def augment(item, rng):
    """4 permutations placing the odd image at A, B, C, D exactly once; the
    three consistent panels are randomly ordered around it."""
    odd = LETTERS.index(item["answer"])
    others = [i for i in range(4) if i != odd]
    out = []
    for slot in range(4):
        rest = others[:]
        rng.shuffle(rest)
        out.append(permute(item, tuple(rest[:slot] + [odd] + rest[slot:])))
    return out


def parse_range(s):
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.jsonl")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed-search", default="0-30")
    ap.add_argument("--aug-seed", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--test-folds", type=int, default=2)
    ap.add_argument("--val-folds", type=int, default=1)
    args = ap.parse_args()

    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    items = [json.loads(l) for l in open(args.manifest) if l.strip()]
    n = len(items)
    gid = np.array(build_groups(items))
    y, cell = strata(items, CRITERIA["min_cell_items"])
    y = np.array(y)
    print(f"{n} items, {len(set(gid))} groups, {len(set(y))} strata")

    search_log, chosen = [], None
    for seed in parse_range(args.seed_search):
        sgk = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True,
                                   random_state=seed)
        folds = [f for _, f in sgk.split(np.zeros(n), y, gid)]
        te = sorted({i for k in range(args.test_folds) for i in folds[k].tolist()})
        va = sorted({i for k in range(args.test_folds, args.test_folds + args.val_folds)
                     for i in folds[k].tolist()})
        tr = sorted(set(range(n)) - set(te) - set(va))
        for a, b in ((tr, va), (tr, te), (va, te)):
            assert not ({gid[i] for i in a} & {gid[i] for i in b})
        bad = violations(items, tr, va, te, cell, CRITERIA)
        search_log.append({"seed": seed, "n": [len(tr), len(va), len(te)],
                           "accepted": not bad, "violations": bad})
        print(f"seed {seed:3d}: {len(tr)}/{len(va)}/{len(te)}  "
              + ("ACCEPTED" if not bad else f"rejected -- {bad[0]}"))
        if not bad:
            chosen = (seed, tr, va, te)
            break

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "split_search_log.json").write_text(json.dumps(
        {"criteria": CRITERIA, "seeds_tried": search_log}, indent=2))
    if chosen is None:
        raise SystemExit("no seed passed the acceptance criteria")

    seed, tr, va, te = chosen
    rng = random.Random(args.aug_seed)
    splits = {"train": [items[i] for i in tr], "val": [items[i] for i in va],
              "test": [items[i] for i in te]}
    splits["train_aug"] = [a for it in splits["train"] for a in augment(it, rng)]
    rng.shuffle(splits["train_aug"])

    for a in splits["train_aug"]:        # permuted rationale names the permuted answer
        assert build_target(a, "rationale").endswith('"%s"}' % a["answer"])

    for name, data in splits.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    datasets = sorted({it["dataset"] for it in items})
    tiers = sorted({it["tier"] for it in items})
    share = lambda idx, k, v: sum(1 for i in idx if items[i][k] == v) / max(len(idx), 1)
    dist = lambda idx, key: dict(sorted(Counter(items[i][key] for i in idx).items()))
    report = {
        "accepted_seed": seed,
        "aug_seed": args.aug_seed,
        "criteria": CRITERIA,
        "seeds_rejected": [s["seed"] for s in search_log if not s["accepted"]],
        "n_groups": int(len(set(gid))),
        "counts": {k: len(v) for k, v in splits.items()},
        "per_dataset": {k: dist(idx, "dataset")
                        for k, idx in (("train", tr), ("val", va), ("test", te))},
        "per_tier": {k: dist(idx, "tier")
                     for k, idx in (("train", tr), ("val", va), ("test", te))},
        "per_letter": {k: dict(sorted(Counter(d["answer"] for d in v).items()))
                       for k, v in splits.items()},
        "shift_train_minus_test": {
            "dataset": {d: round(share(tr, "dataset", d) - share(te, "dataset", d), 4)
                        for d in datasets},
            "tier": {t: round(share(tr, "tier", t) - share(te, "tier", t), 4)
                     for t in tiers}},
        "test_majority_letter_baseline": round(
            max(Counter(items[i]["answer"] for i in te).values()) / max(len(te), 1), 4),
    }
    (out / "split_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["counts"]), "->", out)


if __name__ == "__main__":
    main()

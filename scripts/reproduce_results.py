#!/usr/bin/env python3
"""Recompute the released headline scores without images, models or a GPU."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import parse_strict
from report import cluster_boot, mcnemar_exact, wilson

RUNS = {
    "Base": "baseline/base_q257b.jsonl",
    "Answer SFT": "ans-s1/ans-s1-best_cyc4.jsonl",
    "Rationale and answer SFT": "rat-s1/rat-s1-best_cyc4.jsonl",
}
EXPECTED = {"Base": 44, "Answer SFT": 67, "Rationale and answer SFT": 84}


def reproduce(results_dir, splits_dir):
    lists = {}
    for name, count in (("train", 700), ("val", 99), ("test", 201)):
        ids = (splits_dir / (name + "_qids.txt")).read_text().splitlines()
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError("Unexpected or duplicate official " + name + " IDs")
        lists[name] = set(ids)
    if any(lists[a] & lists[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Official splits overlap")
    summary = {"test_items": 201, "runs": {}}
    canonical_by_run = {}
    for label, name in RUNS.items():
        path = results_dir / name
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        groups = {}
        for record in records:
            qid, view = record.get("src_qid", record["qid"]), record.get("view", 0)
            if qid not in lists["test"] or view not in range(4):
                raise ValueError("Unexpected test record in " + name)
            if view in groups.setdefault(qid, {}):
                raise ValueError("Duplicate question and view in " + name)
            if record.get("response") is not None:
                parsed = parse_strict(record["response"])
                if parsed != record.get("parsed_answer") or (parsed == record["gold"]) != record.get("correct"):
                    raise ValueError("Stored answer disagrees with response parsing in " + name)
            groups[qid][view] = record
        if set(groups) != lists["test"] or any(set(vs) != set(range(4)) for vs in groups.values()):
            raise ValueError("Incomplete four-order test coverage in " + name)
        canonical = {q: parse_strict(vs[0]["response"]) == vs[0]["gold"] for q, vs in groups.items()}
        canonical_by_run[label] = canonical
        n = sum(canonical.values())
        if n != EXPECTED[label]:
            raise ValueError("Released headline result changed for " + label)
        by_tier = {}
        for q, vs in groups.items():
            counts = by_tier.setdefault(vs[0]["tier"], {"correct": 0, "total": 0})
            counts["correct"] += int(canonical[q])
            counts["total"] += 1
        result = {"correct": n, "total": 201, "accuracy": n / 201,
                  "wilson_95": list(wilson(n, 201)), "by_tier": by_tier,
                  "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if all(r.get("response") is not None for vs in groups.values() for r in vs.values()):
            per_item = [sum(parse_strict(vs[k]["response"]) == vs[k]["gold"] for k in range(4)) / 4
                        for q, vs in sorted(groups.items())]
            result.update(cyclic4_accuracy=sum(per_item) / 201,
                          cyclic4_item_bootstrap_95=list(cluster_boot(per_item)),
                          all_four_correct=sum(x == 1 for x in per_item))
        summary["runs"][label] = result
    before, after = canonical_by_run["Answer SFT"], canonical_by_run["Rationale and answer SFT"]
    b = sum(before[q] and not after[q] for q in before)
    c = sum(after[q] and not before[q] for q in before)
    summary["answer_vs_rationale"] = {"answer_only_correct": b, "rationale_only_correct": c,
                                      "mcnemar_exact_p": mcnemar_exact(b, c)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/runs")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = reproduce(args.results_dir, args.splits_dir)
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
    for name, values in result["runs"].items():
        print(f"{name:26s} {values['correct']:3d}/201  {100 * values['accuracy']:5.2f}%")
    print("Official split and archived response checks passed.")
    print("Paired McNemar p = %.6f" % result["answer_vs_rationale"]["mcnemar_exact_p"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print("Saved", args.output)


if __name__ == "__main__":
    main()

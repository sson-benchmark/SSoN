"""
analyze_rationale.py -- are the rationale model's explanations correct, and is
its accuracy gain over the answer-only model concentrated where they are?

Every rationale line has a ground truth in item["sup"], so it is scored
automatically (CPU only):
  edit type    exact match with the true edit type
  object       head noun of "Edited object:" matches the true object
               (object_exact: whole normalised string)
  consistency  the view named in "Reason: In view X" equals the final answer

Then accuracy of BOTH fine-tuned models on the same items, split by whether
the rationale model's type/object were right. If the rationale merely
co-occurred with easy items, the answer-only model would also do better on
the "rationale right" items.

  python analyze_rationale.py --split data/test.jsonl \
      --rat runs/rat-s1/rat-s1-best_cyc4.jsonl --ans runs/ans-s1/ans-s1-best_cyc4.jsonl
"""
import argparse
import json
import re

from common import parse_strict


def field(resp, key):
    return next((l.split(":", 1)[1].strip() for l in resp.splitlines()
                 if l.lower().startswith(key.lower() + ":")), None)


def norm(s):
    return re.sub(r"[^a-z ]", "", (s or "").lower()).replace("the ", "").strip()


def head_match(pred, gold):
    w = norm(pred).split()
    return bool(w) and (w[-1] == (norm(gold).split() or [""])[-1] or w[-1] in norm(gold))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/test.jsonl")
    ap.add_argument("--rat", required=True)
    ap.add_argument("--ans", required=True)
    args = ap.parse_args()

    items = {json.loads(l)["qid"]: json.loads(l) for l in open(args.split) if l.strip()}
    ans = {r["qid"]: r for r in map(json.loads, open(args.ans)) if r.get("response")}
    rows = []
    for r in map(json.loads, open(args.rat)):
        if not r.get("response") or r["qid"] not in ans:
            continue
        s = items[r.get("src_qid", r["qid"])]["sup"]
        resp = r["response"]
        pred = parse_strict(resp)
        m = re.match(r"(?:In view|View) ([A-D])", field(resp, "Reason") or "")
        rows.append({
            "view": r.get("view", 0),
            "rat_ok": pred == r["gold"],
            "ans_ok": parse_strict(ans[r["qid"]]["response"]) == r["gold"],
            "type_ok": (field(resp, "Edit type") or "").lower() == s["edit_type"].lower(),
            "obj_ok": head_match(field(resp, "Edited object"), s["object"]),
            "obj_exact": norm(field(resp, "Edited object")) == norm(s["object"]),
            "consistent": m is not None and m.group(1) == pred,
        })

    mean = lambda xs, k: 100.0 * sum(x[k] for x in xs) / max(len(xs), 1)
    for name, sub in (("original order", [x for x in rows if x["view"] == 0]),
                      ("all 4 cyclic views", rows)):
        print("\n== %s  (n=%d)" % (name, len(sub)))
        print("  rationale field accuracy (%%): edit type %.1f   object (head) %.1f   "
              "object (exact) %.1f   reason view == answer %.1f"
              % (mean(sub, "type_ok"), mean(sub, "obj_ok"), mean(sub, "obj_exact"),
                 mean(sub, "consistent")))
        print("  %-34s %5s %10s %10s %8s" % ("stratum (rationale model's lines)",
                                             "n", "rat acc", "ans acc", "gap"))
        strata = [("type right", lambda x: x["type_ok"]),
                  ("type wrong", lambda x: not x["type_ok"]),
                  ("type & object right", lambda x: x["type_ok"] and x["obj_ok"]),
                  ("type & object wrong", lambda x: not x["type_ok"] and not x["obj_ok"]),
                  ("all", lambda x: True)]
        for label, f in strata:
            g = [x for x in sub if f(x)]
            print("  %-34s %5d %10.1f %10.1f %+8.1f" % (
                label, len(g), mean(g, "rat_ok"), mean(g, "ans_ok"),
                mean(g, "rat_ok") - mean(g, "ans_ok")))


if __name__ == "__main__":
    main()

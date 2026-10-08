"""
eval_rationale_intervention.py -- does the rationale-trained model's answer
depend on the content of its rationale?

The first three rationale lines (Edit type / Edited object / What changed)
contain no panel letters, so they can be forced as an assistant-turn prefix
without revealing the answer. The model then continues from "Reason:" -- it
still has to decide WHICH view carries the edit -- and ends with the JSON.

  --condition gold       true type, object and description of this item
  --condition gold_type  only the true "Edit type:" line; the model writes
                         the object / description itself
  --condition corrupt    type, object and description of another test item
                         with a DIFFERENT edit type (fixed donor per item)

If the answer relies on the rationale, gold should raise accuracy above free
generation and corrupt should lower it. Records have the eval_local.py
schema (response = forced prefix + continuation), so report.py and
make_tables.py work on them unchanged.

  python eval_rationale_intervention.py --adapter ckpts/rat-s1/best \
      --condition gold --views cyclic4 --vit-varlen \
      --out results/runs/rat-s1/intervene-gold_cyc4.jsonl

Resume requires a matching configuration sidecar. Local adapter files are
hashed, while remote Hub IDs are recorded without pinning their revisions.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from common import (DEFAULT_BASE, PROMPT_SHA, PROTOCOL, VERSION, append_jsonl,
                    build_messages, check_seq, cyclic_views, load_images,
                    load_model, load_processor, parse_strict,
                    patch_vision_varlen, read_jsonl, resolve_images, run_key)
from run_provenance import (ProvenanceError, ensure_output_config,
                            evaluation_config)


def prefix_lines(sup, condition):
    """Forced start of the assistant turn, in the training target's format."""
    if condition == "gold_type":
        return "Edit type: %s\n" % sup["edit_type"]
    return ("Edit type: %s\nEdited object: %s\nWhat changed: %s.\nReason:"
            % (sup["edit_type"], sup["object"], sup["edit_description"].rstrip(".")))


def donors(items, seed=0):
    """Each item -> a fixed other item whose edit type differs."""
    rng = random.Random(seed)
    out = {}
    for it in items:
        pool = [d for d in items if d["sup"]["edit_type"] != it["sup"]["edit_type"]]
        out[it["qid"]] = rng.choice(pool)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="data/test.jsonl")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--condition", required=True,
                    choices=["gold", "gold_type", "corrupt"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--views", default="canonical", choices=["canonical", "cyclic4"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the forced prefixes and exit (no model)")
    ap.add_argument("--vit-varlen", action="store_true")
    args = ap.parse_args()
    if args.max_new_tokens <= 0:
        ap.error("--max-new-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        ap.error("--limit must be positive")
    tag = args.tag or "%s-%s" % (args.adapter.strip("/").replace("/", "-"), args.condition)

    items = [json.loads(l) for l in open(args.split) if l.strip()]
    if args.condition == "corrupt" and len({it["sup"]["edit_type"] for it in items}) < 2:
        ap.error("--condition corrupt needs at least two edit types in --split")
    donor = donors(items) if args.condition == "corrupt" else {}
    if args.limit:
        items = items[:args.limit]
    if args.views == "cyclic4":
        items = [v for it in items for v in cyclic_views(it)]

    def prefix_of(it):
        src = it.get("src_qid", it["qid"])
        sup = donor[src]["sup"] if args.condition == "corrupt" else it["sup"]
        return prefix_lines(sup, args.condition)

    if args.dry_run:
        for it in items[:6]:
            print("== %s (gold %s, tier %s)\n%s\n" % (it["qid"], it["answer"],
                                                     it["tier"], prefix_of(it)))
        return

    options = {k: v for k, v in vars(args).items()
               if k not in {"split", "out", "tag", "base", "adapter", "dry_run"}}
    options.update(do_sample=False, dtype="bfloat16", donor_seed=0,
                   gen_views="all", no_generate=False, forced_choice=False)
    try:
        config = evaluation_config(
            split=args.split, base=args.base, adapter=args.adapter,
            prompt_sha=PROMPT_SHA, protocol=PROTOCOL, version=VERSION,
            tag=tag, options=options,
            source_files=[__file__, Path(__file__).with_name("common.py")])
        ensure_output_config(args.out, config)
    except (ProvenanceError, OSError, json.JSONDecodeError) as exc:
        ap.error(str(exc))

    done = {r["run_key"] for r in read_jsonl(args.out)}
    todo = [i for i in items if run_key(i["qid"], VERSION, tag, PROTOCOL) not in done]
    print(f"{len(items)} records, {len(todo)} to run  [{tag}]", flush=True)
    if todo:
        if args.vit_varlen:
            patch_vision_varlen()
        proc, budget = load_processor(args.base)
        from peft import PeftModel
        model = PeftModel.from_pretrained(load_model(args.base, args.attn), args.adapter)
        model = model.merge_and_unload().eval()
        for k, it in enumerate(todo, 1):
            paths = resolve_images(it)
            pre = prefix_of(it)
            text = proc.apply_chat_template(build_messages(paths), tokenize=False,
                                            add_generation_prompt=True) + pre
            inp = proc(text=[text], images=load_images(paths),
                       return_tensors="pt").to(model.device)
            check_seq(inp.input_ids, it["qid"])
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, temperature=None,
                                     top_p=None, top_k=None)
            gen = out[0][inp.input_ids.shape[1]:]
            resp = pre + proc.decode(gen, skip_special_tokens=True)
            pred = parse_strict(resp)
            rec = {"run_key": run_key(it["qid"], VERSION, tag, PROTOCOL),
                   "qid": it["qid"], "dataset": it["dataset"], "tier": it["tier"],
                   "version": VERSION, "protocol": PROTOCOL, "model_tag": tag,
                   "adapter": args.adapter, "condition": args.condition,
                   "forced_prefix": pre, "gold": it["answer"],
                   "response": resp, "parsed_answer": pred,
                   "correct": pred == it["answer"],
                   "truncated": int(gen.shape[0]) >= args.max_new_tokens,
                   "usage": {"completion_tokens": int(gen.shape[0])},
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if "view" in it:
                rec.update(src_qid=it["src_qid"], view=it["view"], perm=it["perm"])
            append_jsonl(args.out, rec)
            if k % 50 == 0 or k == len(todo):
                print(f"[{k}/{len(todo)}]", flush=True)

    from report import load, summarize
    recs = {q: r for q, r in load(args.out).items() if r.get("model_tag") == tag}
    summarize("%s  %s" % (tag, args.out), recs, by=["tier"])


if __name__ == "__main__":
    main()

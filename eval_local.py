"""
eval_local.py -- evaluate Qwen2.5-VL (optionally + a LoRA adapter) on the
odd-one-out protocol. One JSONL record per (item, view); re-running the same
command skips records already on disk only when its configuration sidecar
matches. Use a new output path when changing a run configuration.

Two measurements per record:
  generative  greedy decoding up to --max-new-tokens; the answer is read by
              the protocol parser (the last line must be the JSON), unparsed
              = wrong. This is the headline metric.
  forced      teacher-force '{"odd_one_out": "' right after the assistant tag
              and take the argmax over the A/B/C/D logits (--forced-choice).

--views cyclic4 scores every item in its 4 cyclic panel orders (view 0 = the
original order). --gen-views canonical generates on view 0 only (forced choice
still runs on all views); --gen-views all generates on every view.

  # baseline
  python eval_local.py --split data/test.jsonl --tag base-q25 \
      --out outputs/eval/base_cyclic4.jsonl --forced-choice --views cyclic4
  # fine-tuned
  python eval_local.py --split data/test.jsonl --adapter ckpts/rat-s1/best \
      --tag rat-s1-best --out outputs/eval/released_rationale.jsonl \
      --forced-choice --views cyclic4 --gen-views all --vit-varlen
"""
import argparse
import json
import time
from pathlib import Path

import torch

from common import (ANSWER_PREFIX, DEFAULT_BASE, LETTERS, PROMPT_SHA, PROTOCOL,
                    PROTOCOL_MAX_TOKENS, VERSION, append_jsonl, build_messages,
                    check_seq, cyclic_views, letter_token_ids, load_images,
                    load_model, load_processor, parse_fenced, parse_lenient,
                    parse_strict, patch_vision_varlen, read_jsonl,
                    resolve_images, run_key)
from run_provenance import (ProvenanceError, ensure_output_config,
                            evaluation_config)


def encode(proc, image_paths, suffix="", qid=""):
    text = proc.apply_chat_template(build_messages(image_paths), tokenize=False,
                                    add_generation_prompt=True) + suffix
    enc = proc(text=[text], images=load_images(image_paths), return_tensors="pt")
    check_seq(enc.input_ids, qid)
    return enc


@torch.inference_mode()
def generate(model, proc, image_paths, max_new_tokens, qid=""):
    inputs = encode(proc, image_paths, qid=qid).to(model.device)
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         temperature=None, top_p=None, top_k=None)
    gen = out[0][inputs.input_ids.shape[1]:]
    usage = {"prompt_tokens": int(inputs.input_ids.shape[1]),
             "completion_tokens": int(gen.shape[0])}
    return proc.decode(gen, skip_special_tokens=True), usage, time.time() - t0


@torch.inference_mode()
def forced_choice(model, proc, letter_ids, image_paths, qid=""):
    inputs = encode(proc, image_paths, suffix=ANSWER_PREFIX, qid=qid).to(model.device)
    sel = model(**inputs).logits[0, -1].float()[letter_ids]
    probs = torch.softmax(sel, dim=-1)
    return (LETTERS[int(sel.argmax())],
            {L: round(float(p), 4) for L, p in zip(LETTERS, probs)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--max-new-tokens", type=int, default=PROTOCOL_MAX_TOKENS)
    ap.add_argument("--forced-choice", action="store_true")
    ap.add_argument("--no-generate", action="store_true")
    ap.add_argument("--views", default="canonical", choices=["canonical", "cyclic4"])
    ap.add_argument("--gen-views", default="canonical", choices=["canonical", "all"])
    ap.add_argument("--vit-varlen", action="store_true")
    args = ap.parse_args()
    if args.no_generate:
        args.forced_choice = True
    if args.max_new_tokens <= 0:
        ap.error("--max-new-tokens must be positive")

    options = {k: v for k, v in vars(args).items()
               if k not in {"split", "out", "tag", "base", "adapter"}}
    options.update(do_sample=False, dtype="bfloat16", forced_prefix=ANSWER_PREFIX)
    try:
        config = evaluation_config(
            split=args.split, base=args.base, adapter=args.adapter,
            prompt_sha=PROMPT_SHA, protocol=PROTOCOL, version=VERSION,
            tag=args.tag, options=options,
            source_files=[__file__, Path(__file__).with_name("common.py")])
        ensure_output_config(args.out, config)
    except (ProvenanceError, OSError, json.JSONDecodeError) as exc:
        ap.error(str(exc))

    items = [json.loads(l) for l in open(args.split) if l.strip()]
    if args.views == "cyclic4":
        items = [v for it in items for v in cyclic_views(it)]
    done = {r["run_key"] for r in read_jsonl(args.out)}
    todo = [i for i in items
            if run_key(i["qid"], VERSION, args.tag, PROTOCOL) not in done]
    print(f"{len(items)} records, {len(todo)} to run", flush=True)

    if todo:
        if args.vit_varlen:
            patch_vision_varlen()
        proc, budget = load_processor(args.base)
        model = load_model(args.base, args.attn)
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
        model.eval()
        letter_ids = letter_token_ids(proc.tokenizer) if args.forced_choice else None
        request = {"prompt_sha": PROMPT_SHA[:16], "temperature": 0.0,
                   "do_sample": False, "max_new_tokens": args.max_new_tokens,
                   "dtype": "bfloat16", "attn": args.attn, **budget}

        for k, it in enumerate(todo, 1):
            paths = resolve_images(it)
            rec = {"run_key": run_key(it["qid"], VERSION, args.tag, PROTOCOL),
                   "qid": it["qid"], "dataset": it["dataset"], "tier": it.get("tier"),
                   "band": it.get("band"), "magnitude": it.get("magnitude"),
                   "version": VERSION, "protocol": PROTOCOL, "model_tag": args.tag,
                   "base_model": args.base, "adapter": args.adapter,
                   "request": request, "gold": it["answer"],
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            if "view" in it:
                rec.update(src_qid=it["src_qid"], view=it["view"], perm=it["perm"])
            if not args.no_generate and (args.gen_views == "all" or it.get("view", 0) == 0):
                resp, usage, dt = generate(model, proc, paths, args.max_new_tokens, it["qid"])
                strict = parse_strict(resp)
                rec.update(response=resp, parsed_answer=strict,
                           fenced_answer=parse_fenced(resp),
                           lenient_answer=parse_lenient(resp),
                           correct=strict == it["answer"],
                           truncated=usage["completion_tokens"] >= args.max_new_tokens,
                           usage=usage, latency_s=round(dt, 3))
            else:
                rec.update(response=None, parsed_answer=None, correct=None)
            if letter_ids is not None:
                fc, probs = forced_choice(model, proc, letter_ids, paths, it["qid"])
                rec.update(forced_answer=fc, forced_probs=probs,
                           forced_correct=fc == it["answer"])
            append_jsonl(args.out, rec)
            if k % 50 == 0 or k == len(todo):
                print(f"[{k}/{len(todo)}]", flush=True)

    from report import load, summarize
    recs = {q: r for q, r in load(args.out).items() if r.get("model_tag") == args.tag}
    summarize("%s  %s" % (args.tag, args.out), recs, by=[])


if __name__ == "__main__":
    main()

"""
train_lora.py -- LoRA SFT of Qwen2.5-VL-7B-Instruct on the odd-one-out task.

LoRA is attached to every linear projection of the language model
(q/k/v/o/gate/up/down, 28 layers -> 196 modules); the vision encoder and the
merger are frozen. bf16, batch 1 x 16 gradient accumulation, cosine schedule.

Two supervision targets, identical in everything else:
  --target answer      {"odd_one_out": "X"}
  --target rationale   Edit type / Edited object / What changed / Reason,
                       then {"odd_one_out": "X"} (see common.build_target)

Model selection: every --save-steps optimizer steps the adapter is scored on
the validation set (99 items, one letter-balanced rotation each); the best one
is kept in <out>/best. Answer-only runs are scored by forced choice, rationale
runs by greedy generation + the protocol parser.

  python train_lora.py --target answer    --seed 1 --out out/ans-s1 --vit-varlen
  python train_lora.py --target rationale --seed 1 --out out/rat-s1 --vit-varlen

An interrupted run resumes only with --resume and matching configuration.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from common import (ANSWER_PREFIX, DEFAULT_BASE, LETTERS, build_messages,
                    build_target, check_seq, cyclic_views, letter_token_ids,
                    load_images, load_model, load_processor, parse_strict,
                    patch_vision_varlen, resolve_images)

SEQ_KEYS = ("input_ids", "attention_mask", "labels")


def log(msg=""):
    if int(os.environ.get("RANK", 0)) == 0:
        print(msg, flush=True)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
class OddOneOut(Dataset):
    """One sample = prompt + 4 panels + target. The loss is on the assistant
    turn only (target text, <|im_end|> and the trailing newline)."""

    def __init__(self, path, processor, target="answer"):
        with open(path) as f:
            self.items = [json.loads(l) for l in f if l.strip()]
        self.proc, self.target = processor, target

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        paths = resolve_images(it)
        imgs = load_images(paths)
        full = build_messages(paths, target=build_target(it, self.target))
        text_full = self.proc.apply_chat_template(full, tokenize=False,
                                                  add_generation_prompt=False)
        text_prompt = self.proc.apply_chat_template(build_messages(paths),
                                                    tokenize=False,
                                                    add_generation_prompt=True)
        enc = self.proc(text=[text_full], images=imgs, return_tensors="pt")
        n_prefix = self.proc(text=[text_prompt], images=imgs,
                             return_tensors="pt").input_ids.shape[1]
        check_seq(enc.input_ids, it["qid"])
        labels = enc.input_ids.clone()
        labels[:, :n_prefix] = -100
        out = dict(enc)
        out["labels"] = labels
        for k in SEQ_KEYS:
            out[k] = out[k][0]
        return out


def collate(features):
    assert len(features) == 1, "per-device batch size must be 1"
    return {k: (v.unsqueeze(0) if k in SEQ_KEYS else v)
            for k, v in features[0].items()}


# --------------------------------------------------------------------------
# Trainer with an answer-letter diagnostic
# --------------------------------------------------------------------------
class LetterTrainer(Trainer):
    """Also logs the NLL / accuracy of the answer letter on training batches
    (chance: 1.386 / 0.25). The token-averaged loss cannot show this: it is
    dominated by boilerplate tokens. For rationale targets the metric is taken
    at the FIRST mention of the gold letter ("In view C ..."), the position
    where the model commits; for answer-only it is the JSON letter."""

    def __init__(self, *a, letter_ids=None, space_letter_ids=None, **kw):
        super().__init__(*a, **kw)
        self.letter_ids = torch.tensor(letter_ids)
        self.space_ids = torch.tensor(space_letter_ids)
        self._acc = [0.0, 0, 0]                  # nll sum, correct, count

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kw)
        with torch.no_grad():
            labels = inputs["labels"][:, 1:]
            bare = self.letter_ids.to(labels.device)
            space = self.space_ids.to(labels.device)
            for b in range(labels.shape[0]):
                lab = labels[b]
                pos = torch.isin(lab, bare).nonzero().flatten()
                if not len(pos):
                    continue
                k = int((bare == lab[pos[-1]]).nonzero())       # JSON letter = gold
                first = ((lab == bare[k]) | (lab == space[k])).nonzero().flatten()[0]
                ids = bare if lab[first] == bare[k] else space
                logit = outputs.logits[b, first].float()
                self._acc[0] += float(torch.nn.functional.cross_entropy(
                    logit[None], lab[first][None]))
                self._acc[1] += int(int(logit[ids].argmax()) == k)
                self._acc[2] += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *a, **kw):
        if "loss" in logs and self._acc[2]:
            logs["letter_nll"] = self._acc[0] / self._acc[2]
            logs["letter_acc"] = self._acc[1] / self._acc[2]
            self._acc = [0.0, 0, 0]
        return super().log(logs, *a, **kw)


class ProgressLog(TrainerCallback):
    """One line per logging step (parsed by plot_training.py)."""

    def on_log(self, args, state, control, logs=None, **kw):
        if not state.is_world_process_zero or not logs or "loss" not in logs:
            return
        row = ["step %5d/%-5d" % (state.global_step, state.max_steps),
               "ep %.2f" % float(logs.get("epoch", state.epoch or 0.0)),
               "loss %.4f" % float(logs["loss"])]
        if logs.get("letter_nll") is not None:
            row.append("letter nll %.3f acc %.2f" % (logs["letter_nll"],
                                                     logs["letter_acc"]))
        if logs.get("learning_rate") is not None:
            row.append("lr %.2e" % float(logs["learning_rate"]))
        if logs.get("grad_norm") is not None:
            row.append("gnorm %.2f" % float(logs["grad_norm"]))
        log("  ".join(row))


# --------------------------------------------------------------------------
# Validation / model selection
# --------------------------------------------------------------------------
class Validate(TrainerCallback):
    """Scores the adapter at every checkpoint and keeps the best in <out>/best.

    Views: each val item is shown once, rotated cyclically so that the odd
    image sits at slot (i mod 4) -- gold letters are balanced and a model
    that always answers the same letter scores 0.25.

    forced (answer target): teacher-force '{"odd_one_out": "' after the
      assistant tag and take the argmax over the A/B/C/D logits.
    generative (rationale target): greedy decoding (max_new_tokens tokens),
      scored with the protocol parser; also logs parse rate and whether the
      "Edit type:" line names the true edit.
    """

    def __init__(self, model, processor, val_path, log_path, generative,
                 max_new_tokens=192):
        self.model, self.proc = model, processor
        self.generative, self.max_new_tokens = generative, max_new_tokens
        items = [json.loads(l) for l in open(val_path) if l.strip()]
        self.items = [next(v for v in cyclic_views(it) if v["answer"] == LETTERS[i % 4])
                      for i, it in enumerate(items)]
        self.letter_ids = letter_token_ids(processor.tokenizer)
        self.metric = "balanced" + ("/gen" if generative else "")
        self.log_path = Path(log_path)
        self.best, self.done = -1.0, set()
        for r in (json.loads(l) for l in open(self.log_path) if l.strip()) \
                if self.log_path.exists() else []:          # restore on resume
            if r.get("views") == self.metric:
                self.best = max(self.best, r["val_forced_acc"])
                self.done.add(r["step"])

    def _inputs(self, paths, suffix=""):
        text = self.proc.apply_chat_template(build_messages(paths), tokenize=False,
                                             add_generation_prompt=True) + suffix
        return self.proc(text=[text], images=load_images(paths),
                         return_tensors="pt").to(self.model.device)

    @torch.no_grad()
    def score(self):
        was_training = self.model.training
        self.model.eval()
        ok, preds, nll, parsed, edit_ok = [], [], 0.0, 0, 0
        for it in self.items:
            paths = resolve_images(it)
            if self.generative:
                inp = self._inputs(paths)
                out = self.model.generate(**inp, max_new_tokens=self.max_new_tokens,
                                          do_sample=False, use_cache=True,
                                          temperature=None, top_p=None, top_k=None)
                resp = self.proc.decode(out[0][inp.input_ids.shape[1]:],
                                        skip_special_tokens=True)
                pred = parse_strict(resp)
                parsed += pred is not None
                line = next((l for l in resp.splitlines()
                             if l.lower().startswith("edit type:")), "")
                edit_ok += (line.split(":", 1)[-1].strip().lower()
                            == it["sup"]["edit_type"].lower())
            else:
                logits = self.model(**self._inputs(paths, ANSWER_PREFIX)).logits[0, -1]
                sel = logits[self.letter_ids].float()
                pred = LETTERS[int(sel.argmax())]
                nll -= float(torch.log_softmax(sel, -1)[LETTERS.index(it["answer"])])
            ok.append(pred == it["answer"])
            preds.append(pred)
        if was_training:
            self.model.train()
        n = len(ok)
        return {"acc": sum(ok) / n, "n": n,
                "nll": None if self.generative else nll / n,
                "parse": parsed / n, "edit_acc": edit_ok / n,
                "pred": {L: preds.count(L) for L in LETTERS}}

    def on_save(self, args, state, control, **kw):
        if not state.is_world_process_zero or state.global_step in self.done:
            return
        self.done.add(state.global_step)
        sc = self.score()
        rec = {"epoch": round(state.epoch or 0, 4), "step": state.global_step,
               "val_forced_acc": sc["acc"], "n_val": sc["n"], "views": self.metric,
               "val_letter_nll": sc["nll"], "val_pred": sc["pred"]}
        if self.generative:
            rec.update(val_parse=sc["parse"], val_edit_acc=sc["edit_acc"])
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec, allow_nan=False) + "\n")
        new_best = sc["acc"] > self.best
        log("  VAL  step %5d  ep %.2f  acc %.3f  pred %s%s"
            % (state.global_step, rec["epoch"], sc["acc"],
               "/".join(str(sc["pred"][L]) for L in LETTERS),
               "  <- best" if new_best else ""))
        if new_best:
            self.best = sc["acc"]
            self.model.save_pretrained(Path(args.output_dir) / "best")


# --------------------------------------------------------------------------
def llm_lora_targets(model):
    """Full module paths of the LLM's linear projections. Suffix matching
    ("q_proj") would also hit the ViT's own projections, so full paths are
    used and anything under `visual` is excluded."""
    suffixes = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
    names = sorted(n for n, m in model.named_modules()
                   if isinstance(m, torch.nn.Linear)
                   and "visual" not in n.split(".")
                   and n.split(".")[-1] in suffixes)
    assert names
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train_aug.jsonl")
    ap.add_argument("--val", default="data/val.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--target", default="answer", choices=["answer", "rationale"])
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--log-steps", type=int, default=10)
    ap.add_argument("--val-max-new-tokens", type=int, default=192)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--resume", action="store_true",
                    help="resume an identical interrupted run in --out")
    ap.add_argument("--vit-varlen", action="store_true",
                    help="faster ViT attention (common.patch_vision_varlen)")
    args = ap.parse_args()

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise SystemExit("This release supports single-GPU training. Launch with python, not multi-process torchrun.")
    if not torch.cuda.is_available():
        raise SystemExit("Training requires a CUDA GPU. Result reproduction works on CPU with scripts/reproduce_results.py.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("The recorded training configuration requires BF16 support.")
    from common import PROMPT_SHA
    from training_safety import file_sha256, prepare_training_output
    provenance = {k: v for k, v in vars(args).items() if k not in {"resume", "out"}}
    try:
        provenance.update(train_sha256=file_sha256(args.train),
                          val_sha256=file_sha256(args.val), prompt_sha256=PROMPT_SHA,
                          torch_version=torch.__version__,
                          transformers_version=__import__("transformers").__version__,
                          peft_version=__import__("peft").__version__)
        prepare_training_output(args.out, provenance, args.resume)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc

    set_seed(args.seed)
    if args.vit_varlen:
        patch_vision_varlen()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    proc, budget = load_processor(args.base)
    model = load_model(args.base, args.attn, {"": local_rank})
    model.config.use_cache = False

    from peft import LoraConfig, get_peft_model
    targets = llm_lora_targets(model)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=targets))
    for n, p in model.named_parameters():
        if "visual" in n.split("."):
            p.requires_grad_(False)
    model.enable_input_require_grads()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("base %s  target=%s  LoRA r=%d alpha=%d on %d modules  trainable %.1fM"
        % (args.base, args.target, args.rank, args.alpha, len(targets), n_train / 1e6))

    train_ds = OddOneOut(args.train, proc, args.target)
    log("example target (%s):\n%s" % (train_ds.items[0]["qid"],
                                      build_target(train_ds.items[0], args.target)))

    steps_per_epoch = math.ceil(len(train_ds) / args.accum)
    total_steps = int(steps_per_epoch * args.epochs)
    warmup_steps = max(1, round(args.warmup_ratio * total_steps))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(
        {**vars(args), "image_budget": budget, "n_lora_targets": len(targets),
         "trainable_params": n_train, "warmup_steps": warmup_steps,
         "planned_total_steps": total_steps,
         "transformers": __import__("transformers").__version__}, indent=2))

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        optim="adamw_torch",
        logging_steps=args.log_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[],
        disable_tqdm=True,
        dataloader_num_workers=args.workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )
    trainer = LetterTrainer(model=model, args=targs, train_dataset=train_ds,
                            data_collator=collate,
                            letter_ids=letter_token_ids(proc.tokenizer),
                            space_letter_ids=letter_token_ids(proc.tokenizer, " "))
    from transformers.trainer_callback import PrinterCallback
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(ProgressLog())
    val_cb = Validate(model, proc, args.val, out / "val_log.jsonl",
                      generative=args.target == "rationale",
                      max_new_tokens=args.val_max_new_tokens)
    trainer.add_callback(val_cb)

    from transformers.trainer_utils import get_last_checkpoint
    last = get_last_checkpoint(args.out) if args.resume else None
    t0 = time.time()
    trainer.train(resume_from_checkpoint=last)
    if trainer.state.is_world_process_zero:
        model.save_pretrained(out / "final")
        proc.save_pretrained(out / "final")
        log("done in %.1f h; best val acc %.3f -> %s"
            % ((time.time() - t0) / 3600, val_cb.best, out / "best"))


if __name__ == "__main__":
    main()

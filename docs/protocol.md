# Fine-tuning protocol

## Data and supervision

The official split contains 700 training questions, 99 validation questions, and 201 test questions. Ordered question lists are in `data/splits/`. Shared scenes, scene families, and identical panel bytes remain inside one split. Question images and the unified manifest come from the [SSoN dataset on Hugging Face](https://huggingface.co/datasets/HuiziCao/SSoN). See [data_format.md](data_format.md) for preparation.

Each training question supplies four permutations, placing its intruder at each answer letter once. The permutation generator uses seed 0. There are 2,800 examples, with 700 answers at each letter. Validation uses one cyclic permutation per question, placing the answer at position `i mod 4` in the official validation order. Changing that order changes validation inputs.

Answer supervision trains on the final answer JSON. Rationale supervision adds generator-provided edit type, edited object, change description, and supporting evidence before the same answer JSON. View labels in these fields are remapped with the image permutation. Supervision comes from generation records and is used in training targets only. The test prompt never includes these fields or an answer-key image.

The full assistant target is supervised. User and image prompt tokens are masked from the loss. The logged answer-letter loss is a diagnostic and is not the training objective.

## Recorded settings

| Setting | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Precision | BF16 |
| Training devices | One CUDA GPU |
| LoRA rank, alpha, dropout | 16, 32, 0.05 |
| LoRA targets | Decoder q, k, v, o, gate, up, and down projections |
| Target modules | 196 across 28 decoder layers |
| Trainable parameters | 40,370,176 per adapter |
| Frozen components | Vision encoder and merger |
| Epochs | 3 |
| Per-device batch size | 1 |
| Gradient accumulation | 16 |
| Optimizer | AdamW through `adamw_torch` |
| Learning rate | 0.0001 |
| Schedule | Cosine |
| Warmup | 16 steps from a 0.03 ratio |
| Weight decay | 0 |
| Gradient clipping | 1.0 |
| Gradient checkpointing | Enabled with non-reentrant mode |
| Seed | 1 |
| Optimizer steps | 525 |
| Checkpoint interval | 25 steps |
| Logging interval | 10 steps |
| Image minimum pixels | 3,136 |
| Image maximum pixels | 1,200,000 |
| Maximum tokenized training sequence | 16,384 |
| Rationale validation generation budget | 192 new tokens |
| Test generation budget | 16,384 new tokens |
| Decoding | Greedy |

The training configurations enable the optional variable-length vision-attention optimization in `common.py`. It depends on PyTorch and Transformers internals. Add `--standard-sdpa` to the runner to use standard SDPA instead, and record the selected attention implementation with the run.

## Validation and checkpoint selection

The answer condition selects its best checkpoint using forced next-letter accuracy. The rationale condition uses generated responses and the strict parser. The earliest checkpoint is kept when validation accuracy ties. The included logs select step 250 for the answer run with 37 correct of 99, and step 425 for the rationale run with 43 correct of 99.

`best/` is the selected adapter. `final/` is the last training adapter and is not the selection used for reported test results. Trainer `checkpoint-*` directories retain the state required for resuming an interrupted new run. Released adapter weights alone do not retain optimizer, scheduler, or random state.

## Test metrics

The headline result is strict generative answer accuracy on all 201 test questions in their original panel order. An invalid final answer counts as wrong. The last nonempty line must parse according to `common.parse_strict`.

Forced-choice scores, format-tolerant parsing, and four cyclic panel orders are separate diagnostics. The saved baseline has generated responses only for the canonical order and forced-choice records for all four orders. Both fine-tuned runs contain generated responses for all four orders.

Canonical accuracy intervals use Wilson's method over binary question outcomes. Four-order mean accuracy uses an item bootstrap with 2,000 resamples and seed 0. Each question contributes its mean over four orders. Paired canonical comparisons use the two-sided exact McNemar test.

The stored `ts` field is a timestamp. `analyze_rationale.py` provides heuristic rationale diagnostics; the paper's True solve metric uses human-adjudicated explanations and is separate from these diagnostics.

## Reproduction scope

The saved responses reproduce the reported fine-tuning counts directly. Data preparation deterministically regenerates the fixed split and balanced augmentation. The original full augmented manifest and its input row order were not archived, so regenerated batch order can differ from the original training run.

The current target builder remaps panel references in all three rationale fields: object, change description, and supporting evidence. The initial implementation remapped only the evidence field. The effect on the original training examples cannot be determined without their full manifest.

For reproducible new runs, retain an exact base-model revision or local snapshot, environment versions, prepared-data hashes, and run configuration sidecars. The runner accepts a local base-model path through its `base` field. Software versions, attention implementation, and batch order can affect the optimization trajectory.

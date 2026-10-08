# Checkpoints

SSoN provides two PEFT LoRA adapters for **Qwen2.5-VL-7B-Instruct**, hosted together in the [SSoN checkpoint repository](https://huggingface.co/HuiziCao/SSoN-Qwen2.5-VL-7B-LoRA).

| Adapter | Supervision | Hugging Face | Test accuracy |
| --- | --- | --- | --- |
| Answer | Final answer JSON | [Checkpoints](https://huggingface.co/HuiziCao/SSoN-Qwen2.5-VL-7B-LoRA) | 33.33% |
| Rationale and answer | Generator rationale followed by final answer JSON | [Checkpoints](https://huggingface.co/HuiziCao/SSoN-Qwen2.5-VL-7B-LoRA) | 41.79% |

Both use `Qwen/Qwen2.5-VL-7B-Instruct` as the base model. Each adapter contains 40,370,176 trainable parameters across 392 tensors; the weight file is 161,539,072 bytes. Scores use the same 201 held-out questions in canonical panel order.

## Local layout

Download each adapter from the shared repository into its own local directory containing:

- `adapter_config.json`
- `adapter_model.safetensors`

The evaluator loads the processor and base weights from the configured base model, then applies the selected adapter.

## Evaluation

Prepare the dataset using the [README instructions](../README.md#prepare-the-data), then run:

```bash
python scripts/run_experiment.py --config configs/answer.json --stage eval --adapter /path/to/answer-adapter --eval-out outputs/eval/answer-adapter.jsonl
python scripts/run_experiment.py --config configs/rationale.json --stage eval --adapter /path/to/rationale-adapter --eval-out outputs/eval/rationale-adapter.jsonl
```

Set `ODDOUT_BANK` or pass `--bank /path/to/BANK_dataset`. Use `--standard-sdpa` if the optional variable-length attention optimization is unavailable.

## Checkpoint selection

The answer adapter is selected at step 250, with 37 correct answers out of 99 validation questions. The rationale-and-answer adapter is selected at step 425, with 43 correct answers out of 99. See [the protocol](protocol.md#validation-and-checkpoint-selection) for the selection metrics.

Training creates `best/` for the validation-selected adapter and `final/` for the final training adapter. Resume training from a Trainer `checkpoint-*` directory, which includes optimizer and scheduler state.

## Import a local adapter archive

For archives using the `ckpts/ans-s1/best/` and `ckpts/rat-s1/best/` layout:

```bash
python scripts/import_adapters.py --archive /path/to/qwen_lora_release.zip
```

This extracts the selected weights and configurations into `ckpts/`. They can then be evaluated using `--adapter ckpts/ans-s1/best` or `--adapter ckpts/rat-s1/best`.

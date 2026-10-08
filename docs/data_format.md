# Official data and preparation

The official split contains 700 training, 99 validation, and 201 test questions. The ordered question lists in `data/splits/` define membership and order. `prepare_official_data.py` checks these lists against their stored fingerprints. Use `--splits-dir` to read identical copies from another location.

The [SSoN dataset on Hugging Face](https://huggingface.co/datasets/HuiziCao/SSoN) provides the question images and unified manifest, including generator supervision. Download it before preparing training data. The repository includes the official split lists and model responses for result reproduction.

## Preparing the bank

If the dataset includes a unified manifest, use it directly with `prepare_official_data.py`. For the per-source CSV layout, first run `build_manifest.py`. This importer expects one directory per source collection, one `manifest_*.csv` in each source directory, and one folder per question. Each question folder contains its canonical images `A.png`, `B.png`, `C.png`, and `D.png`. Generation metadata is read from `repro.json`, `row.json`, `v2_info.json`, or the supported `q*.json` fallback. The CSV supplies the answer and edit facts.

```bash
python build_manifest.py \
  --bank /path/to/BANK_dataset \
  --out data/manifest.jsonl

python prepare_official_data.py \
  --manifest data/manifest.jsonl \
  --bank /path/to/BANK_dataset \
  --out data
```

The second command verifies all 4,000 panel references against their stored SHA-256 values. It also checks the fixed ID lists, declared scene and family groups, supervision completeness, and canonical panel order. It creates the following files.

| File | Content |
| --- | --- |
| `train.jsonl` | 700 canonical training records in the official training ID order |
| `val.jsonl` | 99 canonical validation records in the official validation ID order |
| `test.jsonl` | 201 canonical test records in the official test ID order |
| `train_aug.jsonl` | 2,800 training records with four panel orders per training question |
| `official_preparation.json` | Input and output fingerprints, counts, and validation status |

To replace existing prepared files, add `--overwrite`. The script validates inputs before writing and records hashes of the generated files in `official_preparation.json`. The experiment runner checks this report before loading prepared data.

For metadata-only inspection, add `--skip-image-check`. Its report sets `status` to `metadata_only_unverified` and `training_ready` to `false`. This mode checks declared image hashes for split leakage without reading images; run preparation without the flag before training.

## Unified manifest

The input is a UTF-8 JSONL file with exactly one canonical record for each of the 1,000 official IDs. Blank lines are ignored. Duplicate JSON keys, nonstandard numeric values, duplicate question IDs, missing official IDs, and extra IDs are rejected.

| Field | Type | Meaning |
| --- | --- | --- |
| `qid` | String | Exact question ID from the official split files |
| `dataset` | String | One of `3dfront`, `infinigen`, `ithor_procthor`, `robothor`, `sage`, or `tdw` |
| `tier` | String | One of `swap`, `move`, `rotate`, `mirror`, `delete`, or `detail` |
| `answer` | String | Trusted intruder letter in canonical A to D order |
| `gt` | Optional string | When present, must match the canonical `answer` |
| `scene` | String | Declared scene identifier, normally namespaced by source |
| `family` | String or null | Declared scene-family identifier, or null when no family is declared |
| `images` | Four strings | Canonical A, B, C, D question-image paths |
| `rel_images` | Optional four strings | The same paths relative to the bank root |
| `image_sha256` | Four strings | Lowercase SHA-256 digests in the same canonical panel order |
| `sup` | Object | Generator-supplied edit facts, retained as target metadata |

Other fields, including `dataset_dir`, `band`, `magnitude`, and additional source metadata, are retained. `images` is normalized to resolved absolute paths for the model loader. `rel_images` is preserved for relocating the bank, and the preparation report records the input manifest fingerprint.

The reporting tag `ithor_procthor` combines iTHOR and ProcTHOR. RoboTHOR has its own tag. This grouping yields six reporting tags for the seven underlying collections.

Example schema:

```json
{
  "qid": "EXAMPLE_ID",
  "dataset": "ithor_procthor",
  "tier": "move",
  "answer": "B",
  "gt": "B",
  "scene": "ithor_procthor|EXAMPLE_SCENE",
  "family": null,
  "images": ["EXAMPLE_ID/A.png", "EXAMPLE_ID/B.png", "EXAMPLE_ID/C.png", "EXAMPLE_ID/D.png"],
  "rel_images": ["EXAMPLE_ID/A.png", "EXAMPLE_ID/B.png", "EXAMPLE_ID/C.png", "EXAMPLE_ID/D.png"],
  "image_sha256": ["<SHA256_A>", "<SHA256_B>", "<SHA256_C>", "<SHA256_D>"],
  "sup": {
    "edit_type": "move",
    "object": "the white cup",
    "edit_description": "The white cup changes position in view B",
    "gt_reason": "View B places the cup farther from the toaster than views A and C.",
    "distractor": null
  }
}
```

## Image path resolution

Pass `--bank` when the manifest was generated on another machine. With `rel_images`, the supplied bank root takes precedence over stale absolute `images` paths. The `ODDOUT_BANK` environment variable or a local `BANK_dataset` found by `common.find_bank` can supply the same root when `--bank` is omitted.

Without a bank root, existing absolute `images` paths are used directly. Relative `images` paths are resolved against the manifest directory, then the current working directory. With a bank root but no `rel_images`, relative `images` paths are tried against that root first.

Canonical filenames must identify A, B, C, and D in that order. PNG, JPG, JPEG, and WebP extensions are accepted. Supply local question images within the bank root, including symlink targets; URL references and answer-key images are not accepted. Every image file must match its stored digest.

## Supervision and panel remapping

Every manifest record must contain nonempty generator fields `sup.edit_type`, `sup.object`, `sup.edit_description`, and `sup.gt_reason`. The edit type must equal the canonical `tier`. Empty fields and placeholders such as `TODO` are rejected. Use the dataset's generation records as supervision.

View references in `sup.object`, `sup.edit_description`, and `sup.gt_reason` must use explicit uppercase panel letters inside recognized spans such as `view B`, `panels A and C`, or `images A, C and D`. Ambiguous standalone letters and lowercase panel references are rejected. The target builder remaps supported view references in all three fields while preserving the canonical generator metadata. See [protocol.md](protocol.md) for the implementation change from the initial training code.

The optional `sup.distractor` is a canonical zero-based panel index from 0 to 3. It must identify an anchor, not the intruder. It may be null. Other supervision metadata is retained without being added to model input.

`sup` always refers to the original canonical panel order, including inside augmented rows. The `perm` field records the relationship to the displayed order. For example, `perm = [2, 0, 3, 1]` means that displayed panels A, B, C, and D contain canonical panels C, A, D, and B. The row's `answer` is remapped to the displayed order. Its optional `gt` remains the original canonical metadata. `build_target` uses `answer` for the final JSON and remaps recognized view references when constructing a rationale target.

The model input contains only the common prompt and four question images. Generator facts and answers enter assistant training targets only. Validation and test predictions are scored against the answer key after generation.

## Fixed split and deterministic augmentation

The grouping check uses `prepare_data.build_groups`. Any shared image digest, declared scene, or declared family joins questions into one connected group. Every connected group must remain entirely within one official split. Cross-split groups are reported as preparation errors.

Training augmentation uses `prepare_data.augment` with seed 0. It reads canonical training records in `train_qids.txt` order, makes four permutations with the intruder at each answer letter once, and shuffles the resulting 2,800 rows. This produces exactly 700 training targets per answer letter. Prepared validation and test files retain canonical panel order; validation-time permutations and evaluation diagnostics are applied by the model scripts as described in [protocol.md](protocol.md).

The preparation report records augmentation settings and file fingerprints for each run.

## Lightweight verification

```bash
python -m unittest discover -s tests -p test_official_data.py -v
```

These CPU tests use temporary synthetic images and metadata to check ordering, counts, image digests, split separation, supervision, target remapping, and recovery from write errors. No GPU or model download is needed.

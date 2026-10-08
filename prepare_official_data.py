"""Materialize the archived SSoN split without searching for another split.

The official question lists define membership and canonical row order. Images
and generator supervision must come from the released question bank, never
from model predictions. This module uses only the Python standard library.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile

from common import (
    LETTERS,
    build_target,
    find_bank,
    panel_letter_leftovers,
    parse_strict,
    sha256_file,
)
from prepare_data import augment, build_groups


ROOT = Path(__file__).resolve().parent
DEFAULT_SPLITS = ROOT / "data" / "splits"
SPLIT_COUNTS = {"train": 700, "val": 99, "test": 201}
# These digests identify the archived ordered lists. Normalized line endings
# permit an unchanged checkout on Windows without permitting new membership.
OFFICIAL_LIST_SHA256 = {
    "train": "ca4eaca175307fa07090f008e8513e4b3d8944b59734dae14a62cea9c15e8ff6",
    "val": "9b11869a1b7b758b33ddecd167e27c9967dd021bffd5cd383455915aaa34999e",
    "test": "8d1288b1a917111e61fc05ac4f1799992f9021d547b58a767bcf49807fe9ca0e",
}
TIERS = {"swap", "move", "rotate", "mirror", "delete", "detail"}
DATASETS = {"3dfront", "infinigen", "ithor_procthor", "robothor", "sage", "tdw"}
SUPERVISION_FIELDS = ("edit_type", "object", "edit_description", "gt_reason")
OUTPUT_NAMES = (
    "train.jsonl", "val.jsonl", "test.jsonl", "train_aug.jsonl",
    "official_preparation.json",
)
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
PLACEHOLDERS = {"", "na", "n/a", "none", "null", "unknown", "todo", "tbd"}
VIEW_REFERENCE = re.compile(
    r"\b(?:views?|panels?|images?)\s+"
    r"([A-D](?:(?:\s*,\s*(?:and\s+)?|\s+and\s+|\s*&\s*|\s*/\s*)[A-D])*)\b",
    re.IGNORECASE,
)


class PreparationError(ValueError):
    """An input is missing, invalid, or inconsistent with the official split."""


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _sample(values):
    values = sorted(values)
    return ", ".join(values[:4]) + (" ..." if len(values) > 4 else "")


def load_official_splits(splits_dir):
    """Load and authenticate the archived ordered question lists."""
    splits_dir = Path(splits_dir).resolve()
    lists, provenance, all_ids = {}, {}, set()
    for name, count in SPLIT_COUNTS.items():
        path = splits_dir / f"{name}_qids.txt"
        if not path.is_file():
            raise PreparationError(f"Missing {path}. Restore the archived data/splits files.")
        raw = path.read_bytes()
        try:
            qids = [line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()]
        except UnicodeDecodeError as exc:
            raise PreparationError(f"{path} must be UTF-8 text.") from exc
        if len(qids) != count:
            raise PreparationError(f"{path} has {len(qids)} IDs. The official {name} split needs {count}.")
        duplicates = [qid for qid, n in Counter(qids).items() if n > 1]
        if duplicates:
            raise PreparationError(f"Duplicate IDs in {path}. {_sample(duplicates)}")
        overlap = set(qids) & all_ids
        if overlap:
            raise PreparationError(f"IDs appear in multiple splits. {_sample(overlap)}")
        normalized = ("\n".join(qids) + "\n").encode("utf-8")
        normalized_sha = _digest(normalized)
        if normalized_sha != OFFICIAL_LIST_SHA256[name]:
            raise PreparationError(
                f"{path} differs from the archived ordered {name} list. "
                "Restore the official list rather than generating a new split."
            )
        lists[name] = qids
        all_ids.update(qids)
        provenance[name] = {
            "path": str(path), "sha256": _digest(raw),
            "ordered_ids_sha256": normalized_sha, "count": count,
        }
    return lists, provenance


def _reject_constant(value):
    raise PreparationError(f"Nonstandard JSON constant {value} is not allowed in the manifest.")


def _object_without_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise PreparationError(f"Duplicate JSON key {key!r} in the manifest.")
        out[key] = value
    return out


def load_manifest(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise PreparationError(
            f"Manifest not found at {path}. Run build_manifest.py with the released "
            "BANK_dataset, or supply the released unified manifest with --manifest."
        )
    items, seen = [], set()
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(
                    line, parse_constant=_reject_constant,
                    object_pairs_hook=_object_without_duplicate_keys,
                )
            except (ValueError, PreparationError) as exc:
                raise PreparationError(f"{path} line {lineno} is invalid JSON. {exc}") from exc
            if not isinstance(item, dict):
                raise PreparationError(f"{path} line {lineno} must be a JSON object.")
            qid = item.get("qid")
            if not isinstance(qid, str) or not qid.strip() or qid != qid.strip():
                raise PreparationError(f"{path} line {lineno} needs a nonempty qid without surrounding whitespace.")
            if qid in seen:
                raise PreparationError(f"Duplicate manifest qid {qid}.")
            seen.add(qid)
            items.append(item)
    if not items:
        raise PreparationError(f"{path} is empty. Supply all 1,000 canonical question records.")
    return items


def _is_text(value):
    return isinstance(value, str) and value.strip() and value.strip().lower() not in PLACEHOLDERS


def _validate_paths(item, key, required):
    paths = item.get(key)
    if paths is None and not required:
        return
    if not isinstance(paths, list) or len(paths) != 4:
        raise PreparationError(f"{item['qid']} needs four {key} paths in canonical A, B, C, D order.")
    for letter, value in zip(LETTERS, paths):
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise PreparationError(f"{item['qid']} has an invalid {key} path for panel {letter}.")
        path = Path(value)
        if "://" in value or "\\" in value or "\x00" in value:
            raise PreparationError(f"{item['qid']} {key} must use local paths with forward slashes.")
        if path.stem != letter or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise PreparationError(
                f"{item['qid']} {key} panel {letter} must reference the question image named {letter}.png "
                "or the corresponding JPG or WebP. Answer-key images are not model inputs."
            )
        if key == "rel_images" and (path.is_absolute() or ".." in path.parts):
            raise PreparationError(f"{item['qid']} rel_images must remain inside the question-bank directory.")
    if len(set(paths)) != 4:
        raise PreparationError(f"{item['qid']} reuses a path within {key}.")


def validate_item(item):
    """Check canonical metadata before any augmentation or filesystem writes."""
    qid = item["qid"]
    if not isinstance(item.get("dataset"), str) or item["dataset"] not in DATASETS:
        raise PreparationError(f"{qid} has an invalid dataset tag. Use one of {', '.join(sorted(DATASETS))}.")
    if not isinstance(item.get("tier"), str) or item["tier"] not in TIERS:
        raise PreparationError(f"{qid} has an invalid tier. Use one of {', '.join(sorted(TIERS))}.")
    if item.get("answer") not in LETTERS:
        raise PreparationError(f"{qid} answer must be one of A, B, C, D.")
    if "gt" in item and item["gt"] != item["answer"]:
        raise PreparationError(f"{qid} gt and answer disagree in the canonical manifest.")
    if item.get("perm") not in (None, [0, 1, 2, 3]) or item.get("src_qid") not in (None, qid):
        raise PreparationError(f"{qid} is already permuted. Supply canonical question records only.")
    if "view" in item and item["view"] not in (None, 0):
        raise PreparationError(f"{qid} is a cyclic view rather than a canonical record.")
    if not _is_text(item.get("scene")):
        raise PreparationError(f"{qid} needs a declared scene identifier for leakage checks.")
    if "family" not in item or (item["family"] is not None and not _is_text(item["family"])):
        raise PreparationError(f"{qid} needs family as a declared identifier or null when none exists.")
    _validate_paths(item, "images", required=True)
    _validate_paths(item, "rel_images", required=False)
    hashes = item.get("image_sha256")
    if (not isinstance(hashes, list) or len(hashes) != 4
            or any(not isinstance(h, str) or not HEX64.fullmatch(h) for h in hashes)):
        raise PreparationError(f"{qid} needs four lowercase SHA-256 image digests in A, B, C, D order.")
    if len(set(hashes)) != 4:
        raise PreparationError(f"{qid} contains duplicate panel-image hashes.")
    sup = item.get("sup")
    if not isinstance(sup, dict):
        raise PreparationError(f"{qid} is missing generator supervision in sup. Do not replace it with model predictions.")
    for field in SUPERVISION_FIELDS:
        if not _is_text(sup.get(field)):
            raise PreparationError(
                f"{qid} needs nonempty generator sup.{field}. "
                "Recover it from the generation metadata. Generic fallback rationales are not accepted."
            )
        if any(re.search(r"\b[a-d]\b", span.group(1)) for span in VIEW_REFERENCE.finditer(sup[field])):
            raise PreparationError(f"{qid} sup.{field} uses a lowercase panel reference. Use uppercase A to D in view spans.")
        leftovers = panel_letter_leftovers(sup[field])
        if leftovers:
            raise PreparationError(
                f"{qid} sup.{field} has ambiguous panel letters {leftovers}. "
                "Express panel references as view B, panels A and C, or images A, C and D "
                "so augmentation can remap them. Clarify literal object identifiers separately."
            )
    if sup["edit_type"] != item["tier"]:
        raise PreparationError(f"{qid} sup.edit_type must match its canonical tier.")
    distractor = sup.get("distractor")
    if distractor is not None:
        if type(distractor) is not int or distractor not in range(4):
            raise PreparationError(f"{qid} sup.distractor must be a canonical index from 0 to 3 or null.")
        if LETTERS[distractor] == item["answer"]:
            raise PreparationError(f"{qid} sup.distractor identifies the intruder rather than an anchor.")


def _contained_path(root, relative, qid):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise PreparationError(f"{qid} has a relative image path or symlink outside the bank directory.")
    return path


def resolve_item_images(item, manifest_path, bank, skip_image_check=False):
    """Materialize usable absolute paths and verify supplied image digests."""
    root = Path(bank).expanduser().resolve() if bank else None
    resolved = []
    for index, value in enumerate(item["images"]):
        if root is not None and item.get("rel_images"):
            candidates = [_contained_path(root, item["rel_images"][index], item["qid"])]
        elif Path(value).is_absolute():
            candidates = [Path(value)]
        else:
            candidates = []
            if root is not None:
                candidates.append(_contained_path(root, value, item["qid"]))
            candidates.append((manifest_path.parent / value).resolve())
            candidates.append(Path(value).resolve())
        path = next((p for p in candidates if p.is_file()), candidates[0])
        if not skip_image_check:
            if not path.is_file():
                raise PreparationError(
                    f"{item['qid']} panel {LETTERS[index]} was not found at {path}. "
                    "Download the question bank and pass --bank or set ODDOUT_BANK. "
                    "Use --skip-image-check only for an explicitly unverified metadata audit."
                )
            actual = sha256_file(path)
            if actual != item["image_sha256"][index]:
                raise PreparationError(
                    f"{item['qid']} panel {LETTERS[index]} SHA-256 mismatch at {path}. "
                    "Restore the matching original image and manifest rather than changing the digest blindly."
                )
        resolved.append(str(path.resolve()))
    if len(set(resolved)) != 4:
        raise PreparationError(f"{item['qid']} resolves multiple panels to the same image file.")
    output = dict(item)
    output["images"] = resolved
    return output


def check_split_leakage(items, membership):
    """Use the historical grouping logic and expose the responsible group."""
    gids = build_groups(items)
    groups = defaultdict(list)
    for item, gid in zip(items, gids):
        groups[gid].append(item["qid"])
    for qids in groups.values():
        partitions = {membership[qid] for qid in qids}
        if len(partitions) > 1:
            raise PreparationError(
                f"Cross-split leakage through a shared image hash, scene, or family in "
                f"{', '.join(sorted(partitions))}. Affected IDs include {_sample(qids)}. "
                "Check the source metadata. Do not silently move official IDs."
            )
    return len(groups)


def _jsonl_bytes(items):
    return ("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in items)).encode("utf-8")


def _write_output_transaction(out, payloads, overwrite):
    """Stage all bytes, atomically replace individual files, and roll back errors.

    A report is the final commit marker. Its hashes identify the matching data
    files, including after an interrupted external process or concurrent read.
    Unrelated content such as the source manifest and archived split lists stays.
    """
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".prepare_official.lock"
    try:
        lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise PreparationError(f"Preparation is already writing to {out}. If a prior process stopped, inspect and remove its lock.") from exc
    os.close(lock_fd)
    staged = None
    installed, backups = [], {}
    try:
        staged = Path(tempfile.mkdtemp(prefix=".prepare-official-", dir=out))
        existing = [name for name in payloads if (out / name).exists()]
        if existing and not overwrite:
            raise PreparationError(f"Output files already exist in {out}. Use another --out or pass --overwrite deliberately.")
        for name, data in payloads.items():
            dest = out / name
            if dest.exists() and not dest.is_file():
                raise PreparationError(f"Cannot replace {dest} because it is not a regular file.")
            with (staged / name).open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        for name in payloads:
            dest = out / name
            if dest.exists():
                backup = staged / (name + ".previous")
                shutil.copy2(dest, backup)
                backups[name] = backup
            os.replace(staged / name, dest)
            installed.append(name)
    except Exception:
        for name in reversed(installed):
            if name in backups:
                os.replace(backups[name], out / name)
            else:
                (out / name).unlink(missing_ok=True)
        raise
    finally:
        if staged is not None:
            shutil.rmtree(staged)
        lock.unlink(missing_ok=True)


def prepare(manifest, out="data", splits_dir=DEFAULT_SPLITS, bank=None,
            skip_image_check=False, overwrite=False):
    """Validate and materialize the fixed release data without running a model."""
    manifest = Path(manifest).expanduser().resolve()
    if manifest in {(Path(out).resolve() / name) for name in OUTPUT_NAMES}:
        raise PreparationError("The source manifest cannot also be one of the prepared output files. Use manifest.jsonl as a separate input.")
    ordered_ids, split_provenance = load_official_splits(splits_dir)
    raw_items = load_manifest(manifest)
    manifest_sha = sha256_file(manifest)
    membership = {qid: name for name, qids in ordered_ids.items() for qid in qids}
    by_id = {item["qid"]: item for item in raw_items}
    missing, extra = set(membership) - set(by_id), set(by_id) - set(membership)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"Missing {len(missing)} official IDs including {_sample(missing)}")
        if extra:
            parts.append(f"Unexpected {len(extra)} IDs including {_sample(extra)}")
        raise PreparationError(". ".join(parts) + ". Supply exactly the 1,000 canonical official items.")
    resolved_bank = bank or find_bank()
    if resolved_bank and not skip_image_check and not Path(resolved_bank).expanduser().is_dir():
        raise PreparationError(f"Question-bank directory {resolved_bank} does not exist. Correct --bank or ODDOUT_BANK.")
    items = []
    for name in SPLIT_COUNTS:
        for qid in ordered_ids[name]:
            item = by_id[qid]
            validate_item(item)
            items.append(resolve_item_images(item, manifest, resolved_bank, skip_image_check))
    n_groups = check_split_leakage(items, membership)
    by_id = {item["qid"]: item for item in items}
    splits = {name: [by_id[qid] for qid in qids] for name, qids in ordered_ids.items()}
    rng = random.Random(0)
    splits["train_aug"] = [row for item in splits["train"] for row in augment(item, rng)]
    rng.shuffle(splits["train_aug"])
    if len(splits["train_aug"]) != 2800 or Counter(row["answer"] for row in splits["train_aug"]) != Counter(dict.fromkeys(LETTERS, 700)):
        raise PreparationError("Training augmentation failed to produce 2,800 rows with 700 answers at each letter.")
    for row in splits["train_aug"]:
        if parse_strict(build_target(row, "rationale")) != row["answer"]:
            raise PreparationError(f"Rationale target disagrees with the remapped answer for {row['qid']}.")
    payloads = {f"{name}.jsonl": _jsonl_bytes(rows) for name, rows in splits.items()}
    report = {
        "schema_version": 1,
        "status": "metadata_only_unverified" if skip_image_check else "images_and_metadata_verified",
        "training_ready": not skip_image_check,
        "manifest": {"path": str(manifest), "sha256": manifest_sha, "count": len(items)},
        "official_split_files": split_provenance,
        "counts": {name: len(rows) for name, rows in splits.items()},
        "per_dataset": {name: dict(sorted(Counter(row["dataset"] for row in rows).items())) for name, rows in splits.items()},
        "per_tier": {name: dict(sorted(Counter(row["tier"] for row in rows).items())) for name, rows in splits.items()},
        "per_letter": {name: dict(sorted(Counter(row["answer"] for row in rows).items())) for name, rows in splits.items()},
        "n_groups": n_groups,
        "split_leakage_checks": ["image_sha256", "scene", "family"],
        "image_verification": {
            "verified_panel_references": 0 if skip_image_check else len(items) * 4,
            "bank": str(Path(resolved_bank).expanduser().resolve()) if resolved_bank else None,
            "skipped": bool(skip_image_check),
            "scope": "File existence and SHA-256 equality. Image decoding and factual content are not re-audited.",
        },
        "augmentation": {
            "seed": 0, "source_order": "archived train_qids.txt order",
            "orders_per_training_item": 4, "global_shuffle": True,
            "historical_ordering_note": (
                "The original augmented JSONL and original manifest row order were not included in the supplied archive. "
                "This release deterministically regenerates four balanced orders from the archived question list. "
                "An identical historical batch order or tensor trajectory is not guaranteed."
            ),
        },
        "supervision": {
            "origin": "Generator facts supplied in the input manifest",
            "canonical_fields_preserved": True,
            "used_as_model_input": False,
            "note": "build_target remaps canonical view references only when constructing the assistant training target.",
        },
        "path_note": "images are resolved absolute paths. rel_images and all other source metadata are preserved.",
        "outputs": {name: {"sha256": _digest(data), "bytes": len(data)} for name, data in payloads.items()},
    }
    payloads["official_preparation.json"] = (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    _write_output_transaction(out, payloads, overwrite)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Canonical 1,000-item JSONL from the released bank")
    parser.add_argument("--out", default="data", help="Directory for the four JSONL files and preparation report")
    parser.add_argument("--splits-dir", default=str(DEFAULT_SPLITS), help="Directory containing unchanged archived train, val, and test qid lists")
    parser.add_argument("--bank", help="Question-bank root used with rel_images, otherwise ODDOUT_BANK or local BANK_dataset")
    parser.add_argument("--skip-image-check", action="store_true", help="Metadata-only audit. Outputs are explicitly unverified and not training-ready")
    parser.add_argument("--overwrite", action="store_true", help="Deliberately replace previous prepared files after successful validation")
    args = parser.parse_args()
    try:
        report = prepare(**vars(args))
    except (PreparationError, OSError, UnicodeError) as exc:
        parser.exit(2, f"Preparation failed. {exc}\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"]}, indent=2))
    if args.skip_image_check:
        print("Image bytes were not checked. Run again without --skip-image-check before training.")
    else:
        print(f"Prepared data written to {Path(args.out).resolve()}.")


if __name__ == "__main__":
    main()

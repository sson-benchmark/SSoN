"""Temporary synthetic fixtures exercise release preparation without a GPU."""

import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import build_messages, build_target, parse_strict, remap_panel_letters
from prepare_official_data import (
    DEFAULT_SPLITS,
    LETTERS,
    PreparationError,
    load_official_splits,
    prepare,
)


def png_bytes(index):
    """One distinct valid RGB pixel per panel, stored only in a temporary dir."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    pixel = bytes(((index >> 16) & 255, (index >> 8) & 255, index & 255))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"\0" + pixel)) + chunk(b"IEND", b""))


class OfficialPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.base = Path(cls.fixture.name)
        cls.bank = cls.base / "BANK_dataset"
        cls.lists, _ = load_official_splits(DEFAULT_SPLITS)
        cls.rows = []
        tiers = ["swap", "move", "rotate", "mirror", "delete", "detail"]
        qids = [qid for split in cls.lists.values() for qid in split]
        for index, qid in enumerate(qids):
            relative = [f"fixture/{qid}/{letter}.png" for letter in LETTERS]
            hashes = []
            for panel, rel in enumerate(relative):
                path = cls.bank / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                payload = png_bytes(index * 4 + panel)
                path.write_bytes(payload)
                hashes.append(hashlib.sha256(payload).hexdigest())
            answer = LETTERS[index % 4]
            others = [letter for letter in LETTERS if letter != answer]
            tier = tiers[index % len(tiers)]
            cls.rows.append({
                "qid": qid, "dataset": "ithor_procthor", "tier": tier,
                "answer": answer, "gt": answer, "scene": f"fixture|{qid}", "family": None,
                "images": [f"/unavailable/original/{p}" for p in relative],
                "rel_images": relative, "image_sha256": hashes,
                "sup": {
                    "edit_type": tier, "object": "the toy cup in view B",
                    "edit_description": "A synthetic fixture change in view C",
                    "gt_reason": f"View {answer} has the toy change. Views {' and '.join(others[:2])} support it.",
                    "distractor": None, "accepted_solutions": "fixture metadata only",
                },
                "extra_source_metadata": {"retain": True},
            })

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)
        self.manifest = self.work / "manifest.jsonl"
        self.out = self.work / "prepared"

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, rows=None):
        rows = self.rows if rows is None else rows
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def run_prepare(self, rows=None, **kwargs):
        self.write_manifest(rows)
        options = {"manifest": self.manifest, "out": self.out, "bank": self.bank}
        options.update(kwargs)
        return prepare(**options)

    def load(self, name):
        return [json.loads(line) for line in (self.out / f"{name}.jsonl").read_text().splitlines()]

    def test_full_preparation_verifies_bytes_and_official_order(self):
        shuffled = copy.deepcopy(self.rows)
        random.Random(22).shuffle(shuffled)
        report = self.run_prepare(shuffled)
        self.assertTrue(report["training_ready"])
        self.assertEqual(report["image_verification"]["verified_panel_references"], 4000)
        self.assertEqual(report["counts"], {"train": 700, "val": 99, "test": 201, "train_aug": 2800})
        for name, qids in self.lists.items():
            actual = self.load(name)
            self.assertEqual([row["qid"] for row in actual], qids)
            self.assertTrue(all(Path(p).is_file() for row in actual for p in row["images"]))
            self.assertTrue(all(row["extra_source_metadata"] == {"retain": True} for row in actual))
        for filename, meta in report["outputs"].items():
            self.assertEqual(hashlib.sha256((self.out / filename).read_bytes()).hexdigest(), meta["sha256"])

    def test_augmentation_balances_and_remaps_all_supervision_fields(self):
        self.run_prepare(skip_image_check=True)
        augmented = self.load("train_aug")
        self.assertEqual(Counter(row["answer"] for row in augmented), Counter(dict.fromkeys(LETTERS, 700)))
        by_source = {row["qid"]: row for row in self.rows}
        occurrences = Counter()
        for row in augmented:
            src = by_source[row["src_qid"]]
            perm = row["perm"]
            letter_of = lambda idx: LETTERS[perm.index(idx)]
            self.assertEqual(row["answer"], letter_of(LETTERS.index(src["answer"])))
            self.assertEqual(row["image_sha256"], [src["image_sha256"][i] for i in perm])
            self.assertEqual(row["rel_images"], [src["rel_images"][i] for i in perm])
            self.assertEqual(row["sup"], src["sup"])
            target = build_target(row, "rationale")
            for field in ("object", "edit_description", "gt_reason"):
                self.assertIn(remap_panel_letters(src["sup"][field], letter_of), target)
            self.assertEqual(parse_strict(target), row["answer"])
            self.assertEqual(parse_strict(build_target(row, "answer")), row["answer"])
            occurrences[row["src_qid"]] += 1
        self.assertEqual(set(occurrences.values()), {4})
        self.assertEqual(len(occurrences), 700)

    def test_metadata_only_is_explicitly_unverified(self):
        report = self.run_prepare(skip_image_check=True, bank=self.work / "absent_bank")
        self.assertEqual(report["status"], "metadata_only_unverified")
        self.assertFalse(report["training_ready"])
        self.assertEqual(report["image_verification"]["verified_panel_references"], 0)
        self.assertIn("not guaranteed", report["augmentation"]["historical_ordering_note"])

    def test_model_input_contains_only_prompt_and_images(self):
        item = copy.deepcopy(self.rows[0])
        item["sup"]["object"] = "SECRET_GENERATOR_OBJECT"
        paths = [str(self.bank / rel) for rel in item["rel_images"]]
        messages = build_messages(paths)
        self.assertNotIn("SECRET_GENERATOR_OBJECT", json.dumps(messages))
        self.assertEqual(len(messages), 1)
        self.assertEqual(len(messages[0]["content"]), 5)

    def test_missing_extra_and_duplicate_ids_rejected(self):
        for kind in ("missing", "extra", "duplicate"):
            with self.subTest(kind=kind):
                rows = copy.deepcopy(self.rows)
                if kind == "missing":
                    rows.pop()
                elif kind == "extra":
                    extra = copy.deepcopy(rows[-1])
                    extra["qid"] = "invented_extra_question"
                    rows.append(extra)
                else:
                    rows.append(copy.deepcopy(rows[-1]))
                with self.assertRaises(PreparationError):
                    self.run_prepare(rows, skip_image_check=True)
        self.assertFalse((self.out / "train.jsonl").exists())

    def test_changed_split_membership_rejected_even_with_same_counts(self):
        splits = self.work / "splits"
        shutil.copytree(DEFAULT_SPLITS, splits)
        train, val = self.lists["train"][:], self.lists["val"][:]
        train[0], val[0] = val[0], train[0]
        (splits / "train_qids.txt").write_text("\n".join(train) + "\n")
        (splits / "val_qids.txt").write_text("\n".join(val) + "\n")
        with self.assertRaisesRegex(PreparationError, "differs from the archived"):
            self.run_prepare(splits_dir=splits, skip_image_check=True)

    def test_wrong_split_counts_rejected(self):
        splits = self.work / "splits"
        shutil.copytree(DEFAULT_SPLITS, splits)
        (splits / "val_qids.txt").write_text("\n".join(self.lists["val"][:-1]) + "\n")
        with self.assertRaisesRegex(PreparationError, "official val split needs 99"):
            load_official_splits(splits)

    def test_shared_image_scene_and_family_rejected(self):
        test_index = 799
        for field in ("image_sha256", "scene", "family"):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows)
                if field == "image_sha256":
                    rows[test_index][field][0] = rows[0][field][0]
                elif field == "scene":
                    rows[test_index][field] = rows[0][field]
                else:
                    rows[test_index][field] = rows[0][field] = "fixture|shared-family"
                with self.assertRaisesRegex(PreparationError, "Cross-split leakage"):
                    self.run_prepare(rows, skip_image_check=True)

    def test_incomplete_supervision_and_ambiguous_letters_rejected(self):
        cases = [("object", ""), ("gt_reason", "TODO"), ("edit_description", "B changed"),
                 ("edit_type", "rotate"), ("gt_reason", "View b changed"),
                 ("gt_reason", "Views A and b show the change")]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                rows = copy.deepcopy(self.rows)
                rows[0]["sup"][field] = value
                with self.assertRaises(PreparationError):
                    self.run_prepare(rows, skip_image_check=True)
        rows = copy.deepcopy(self.rows)
        del rows[0]["sup"]
        with self.assertRaisesRegex(PreparationError, "missing generator supervision"):
            self.run_prepare(rows, skip_image_check=True)

    def test_wrong_answer_and_panel_mapping_rejected(self):
        for kind in ("answer", "panel_order", "permuted"):
            with self.subTest(kind=kind):
                rows = copy.deepcopy(self.rows)
                if kind == "answer":
                    rows[0]["answer"] = "B"
                elif kind == "panel_order":
                    rows[0]["images"][0], rows[0]["images"][1] = rows[0]["images"][1], rows[0]["images"][0]
                else:
                    rows[0]["perm"] = [1, 0, 2, 3]
                with self.assertRaises(PreparationError):
                    self.run_prepare(rows, skip_image_check=True)

    def test_invalid_field_types_rejected_with_preparation_error(self):
        for field, value in (("dataset", []), ("tier", {}), ("scene", []), ("family", []), ("sup", [])):
            with self.subTest(field=field):
                rows = copy.deepcopy(self.rows)
                rows[0][field] = value
                with self.assertRaises(PreparationError):
                    self.run_prepare(rows, skip_image_check=True)

    def test_wrong_image_hash_fails_before_any_output(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["image_sha256"][0] = "0" * 64
        with self.assertRaisesRegex(PreparationError, "SHA-256 mismatch"):
            self.run_prepare(rows)
        self.assertFalse(self.out.exists())

    def test_missing_images_fail_with_actionable_message(self):
        with self.assertRaisesRegex(PreparationError, "Correct --bank or ODDOUT_BANK"):
            self.run_prepare(bank=self.work / "missing")

    def test_relative_images_resolve_from_manifest_directory(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row["images"] = [str(self.bank / p) for p in row["rel_images"]]
            del row["rel_images"]
        # A relative path independent of the shell working directory.
        (self.work / "image-bank").symlink_to(self.bank, target_is_directory=True)
        for row in rows:
            row["images"] = ["image-bank/" + str(Path(p).relative_to(self.bank)) for p in row["images"]]
        self.run_prepare(rows, bank=None)
        self.assertEqual(self.load("train")[0]["images"], [str((self.bank / p).resolve()) for p in self.rows[0]["rel_images"]])

    def test_path_escape_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["rel_images"][0] = "../outside/A.png"
        with self.assertRaisesRegex(PreparationError, "inside the question-bank"):
            self.run_prepare(rows, skip_image_check=True)

    def test_no_overwrite_without_explicit_flag_and_deterministic_bytes(self):
        self.run_prepare(skip_image_check=True)
        before = {p.name: p.read_bytes() for p in self.out.glob("*.jsonl")}
        with self.assertRaisesRegex(PreparationError, "already exist"):
            self.run_prepare(skip_image_check=True)
        self.run_prepare(skip_image_check=True, overwrite=True)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.out.glob("*.jsonl")})

    def test_staging_failure_restores_previous_output(self):
        self.run_prepare(skip_image_check=True)
        before = {p.name: p.read_bytes() for p in self.out.iterdir() if p.is_file()}
        import os
        original_replace = os.replace
        failed = False

        def fail_once(source, destination):
            nonlocal failed
            if Path(destination).name == "test.jsonl" and not failed:
                failed = True
                raise OSError("fixture injected write failure")
            return original_replace(source, destination)

        with patch("prepare_official_data.os.replace", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "injected write failure"):
                self.run_prepare(skip_image_check=True, overwrite=True)
        after = {p.name: p.read_bytes() for p in self.out.iterdir() if p.is_file()}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

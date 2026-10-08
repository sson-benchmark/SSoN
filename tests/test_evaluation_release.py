"""CPU-only checks for release evaluation guards and item-level intervals."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import make_tables
import report
from run_provenance import (ProvenanceError, adapter_identity,
                            ensure_output_config, evaluation_config, fingerprint)


class RunGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "new" / "predictions.jsonl"
        self.split = self.root / "test.jsonl"
        self.split.write_text('{"qid":"one"}\n')
        self.adapter = self.root / "adapter"
        self.adapter.mkdir()
        (self.adapter / "adapter_config.json").write_text('{"r":16}')
        (self.adapter / "adapter_model.safetensors").write_bytes(b"weight-one")
        self.options = {"views": "cyclic4", "gen_views": "all",
                        "no_generate": False, "forced_choice": True,
                        "attn": "sdpa", "vit_varlen": False,
                        "max_new_tokens": 16384, "condition": "gold"}

    def config(self, **overrides):
        kwargs = dict(split=self.split, base="example/base",
                      adapter=str(self.adapter), prompt_sha="prompt-one",
                      protocol="separate_v1_local", version="v3", tag="run-one",
                      options=copy.deepcopy(self.options))
        kwargs.update(overrides)
        return evaluation_config(**kwargs)

    def guard(self, config=None):
        with contextlib.redirect_stderr(io.StringIO()):
            return ensure_output_config(self.output, config or self.config())

    def test_same_config_resumes_without_touching_results(self):
        config = self.config()
        self.guard(config)
        sidecar = Path(str(self.output) + ".config.json")
        before = sidecar.read_bytes()
        self.output.write_text('{"run_key":"existing"}\n')
        self.guard(config)
        self.assertEqual(sidecar.read_bytes(), before)
        self.assertEqual(self.output.read_text(), '{"run_key":"existing"}\n')
        self.assertEqual(json.loads(before)["fingerprint"], fingerprint(config))

    def test_every_behavior_flag_prevents_cache_reuse(self):
        original = self.config()
        self.guard(original)
        self.output.write_text("saved result\n")
        changed = {"views": "canonical", "gen_views": "canonical",
                   "no_generate": True, "forced_choice": False,
                   "attn": "eager", "vit_varlen": True,
                   "max_new_tokens": 512, "condition": "corrupt"}
        for key, value in changed.items():
            with self.subTest(key=key):
                new = copy.deepcopy(original)
                new["options"][key] = value
                with self.assertRaisesRegex(ProvenanceError, "new --out path"):
                    self.guard(new)
                self.assertEqual(self.output.read_text(), "saved result\n")

    def test_base_prompt_tag_and_split_changes_rejected(self):
        self.guard()
        for kwargs in ({"base": "other/base"}, {"prompt_sha": "prompt-two"},
                       {"tag": "run-two"}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ProvenanceError):
                    self.guard(self.config(**kwargs))
        self.split.write_text('{"qid":"two"}\n')
        with self.assertRaises(ProvenanceError):
            self.guard()

    def test_local_adapter_weight_change_rejected(self):
        self.guard()
        (self.adapter / "adapter_model.safetensors").write_bytes(b"weight-two")
        with self.assertRaisesRegex(ProvenanceError, "adapter"):
            self.guard()

    def test_local_adapter_config_change_rejected(self):
        self.guard()
        (self.adapter / "adapter_config.json").write_text('{"r":32}')
        with self.assertRaisesRegex(ProvenanceError, "adapter"):
            self.guard()

    def test_sharded_local_weights_are_all_fingerprinted(self):
        (self.adapter / "adapter_model-00001-of-00002.safetensors").write_bytes(b"one")
        (self.adapter / "adapter_model-00002-of-00002.safetensors").write_bytes(b"two")
        identity = adapter_identity(str(self.adapter))
        self.assertEqual(len(identity["files"]), 4)
        self.assertEqual(identity["config"], {"r": 16})

    def test_hub_adapter_is_explicitly_unpinned(self):
        identity = adapter_identity("organization/adapter")
        self.assertEqual(identity["id"], "organization/adapter")
        self.assertIsNone(identity["revision"])
        self.assertIn("not pinned", identity["verification"])

    def test_incomplete_local_adapter_fails_early(self):
        (self.adapter / "adapter_model.safetensors").unlink()
        with self.assertRaisesRegex(ProvenanceError, "adapter weights"):
            self.config()

    def test_nonempty_legacy_output_is_never_adopted(self):
        self.output.parent.mkdir()
        self.output.write_text("legacy results\n")
        with self.assertRaisesRegex(ProvenanceError, "no configuration sidecar"):
            self.guard()
        self.assertFalse(Path(str(self.output) + ".config.json").exists())
        self.assertEqual(self.output.read_text(), "legacy results\n")

    def test_empty_output_can_be_initialized(self):
        self.output.parent.mkdir()
        self.output.touch()
        self.guard()
        self.assertEqual(self.output.stat().st_size, 0)

    def test_orphan_sidecar_same_config_can_resume(self):
        self.guard()
        self.assertFalse(self.output.exists())
        self.guard()
        with self.assertRaises(ProvenanceError):
            self.guard(self.config(tag="new-run"))

    def test_invalid_or_tampered_sidecars_are_rejected(self):
        self.output.parent.mkdir()
        sidecar = Path(str(self.output) + ".config.json")
        for contents in ("not json", "[]", "{}", '{"config":{"x":NaN}}', json.dumps({
                "schema_version": 1, "config": self.config(),
                "fingerprint": "tampered"})):
            with self.subTest(contents=contents[:30]):
                sidecar.write_text(contents)
                with self.assertRaises(ProvenanceError):
                    self.guard()
                self.assertEqual(sidecar.read_text(), contents)


class IntervalTests(unittest.TestCase):
    def test_canonical_wilson_is_unchanged(self):
        items = {str(i): {"tier": "move"} for i in range(201)}
        scores = {q: {"gen0": [int(q) < 44]} for q in items}
        summary = make_tables.metric_summary(scores, items, "gen0", all_views=True)
        self.assertEqual(summary["n"], 201)
        self.assertAlmostEqual(summary["overall"], 100 * 44 / 201)
        for actual, expected in zip(summary["ci"], report.wilson(44, 201)):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(summary["ci_method"], "Wilson")

    def test_four_views_resample_complete_source_items(self):
        items = {"a": {"tier": "move"}, "b": {"tier": "rotate"},
                 "c": {"tier": "move"}}
        scores = {"a": {"gen4": [True, False, False, False]},
                  "b": {"gen4": [True, True, True, False]},
                  "c": {"gen4": [True, True]}}
        with patch.object(make_tables, "cluster_boot", wraps=report.cluster_boot) as boot:
            result = make_tables.metric_summary(scores, items, "gen4", all_views=True)
        boot.assert_called_once_with([0.25, 0.75])
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["overall"], 50)
        self.assertEqual(result["ci"], report.cluster_boot([0.25, 0.75]))
        self.assertEqual(result["ci_method"], "item bootstrap")

    def test_incomplete_forced_and_distractor_groups_are_not_mixed(self):
        items = {"a": {"tier": "move"}, "b": {"tier": "move"}}
        for metric in ("fc4", "dis4"):
            with self.subTest(metric=metric):
                scores = {"a": {metric: [True] * 4}, "b": {metric: [False]}}
                result = make_tables.metric_summary(scores, items, metric, all_views=True)
                self.assertEqual(result["n"], 1)
                self.assertEqual(result["overall"], 100)

    def test_missing_groups_return_none_and_nonbinary_values_fail(self):
        items = {"a": {"tier": "move"}}
        self.assertIsNone(make_tables.metric_summary({}, items, "gen4", True))
        with self.assertRaisesRegex(ValueError, "Nonbinary"):
            make_tables.metric_summary({"a": {"gen0": [0.5]}}, items, "gen0")


class GuardOrderTests(unittest.TestCase):
    def load_evaluator(self, filename):
        # Import the CLI with a minimal torch stub. The guard must run before
        # any inference helper is called, so torch need not be installed here.
        torch = types.ModuleType("torch")
        torch.inference_mode = lambda: (lambda fn: fn)
        spec = importlib.util.spec_from_file_location("release_evaluator", ROOT / filename)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch": torch}):
            spec.loader.exec_module(module)
        return module

    def test_both_clis_guard_before_reading_cache_or_loading_model(self):
        for filename in ("eval_local.py", "eval_rationale_intervention.py"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                split = root / "test.jsonl"
                split.write_text(json.dumps({"qid": "one", "tier": "move",
                                             "answer": "A", "sup": {"edit_type": "move"}}) + "\n")
                output = root / "legacy.jsonl"
                output.write_text("legacy\n")
                module = self.load_evaluator(filename)
                argv = [filename, "--split", str(split), "--out", str(output),
                        "--tag", "test"]
                if "intervention" in filename:
                    argv += ["--adapter", "organization/adapter", "--condition", "gold"]
                with patch.object(sys, "argv", argv), \
                        patch.object(module, "read_jsonl") as read_cache, \
                        patch.object(module, "load_model") as load_model, \
                        contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit):
                    module.main()
                read_cache.assert_not_called()
                load_model.assert_not_called()
                self.assertEqual(output.read_text(), "legacy\n")


if __name__ == "__main__":
    unittest.main()

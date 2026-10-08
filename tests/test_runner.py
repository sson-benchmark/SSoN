import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_experiment import command, verify_prepared_data


class RunnerTests(unittest.TestCase):
    def test_boolean_flags_and_path_arguments(self):
        argv = command("train_lora.py", {"vit_varlen": True, "resume": False, "out": "path with spaces"})
        self.assertIn("--vit-varlen", argv)
        self.assertNotIn("--resume", argv)
        self.assertEqual(argv[-1], "path with spaces")

    def test_prepared_data_cannot_change_after_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            outputs = {}
            for name in ("train.jsonl", "val.jsonl", "test.jsonl", "train_aug.jsonl"):
                content = (name + "\n").encode()
                (data / name).write_bytes(content)
                outputs[name] = {"sha256": hashlib.sha256(content).hexdigest()}
            path = data / "official_preparation.json"
            report = {"training_ready": True, "outputs": outputs}
            path.write_text(json.dumps(report))
            stages = [("eval", {"split": "data/test.jsonl"})]
            verify_prepared_data(root, stages)
            with self.assertRaisesRegex(ValueError, "Official runner requires"):
                verify_prepared_data(root, [("eval", {"split": "another_test.jsonl"})])
            (data / "test.jsonl").write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed or is missing"):
                verify_prepared_data(root, stages)
            report["training_ready"] = False
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "without image verification"):
                verify_prepared_data(root, stages)


if __name__ == "__main__":
    unittest.main()

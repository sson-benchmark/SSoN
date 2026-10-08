import tempfile
import unittest
from pathlib import Path
from training_safety import prepare_training_output


class ResumeGuardTests(unittest.TestCase):
    def test_explicit_matching_resume_only(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            config = {"target": "answer", "data_hash": "first"}
            prepare_training_output(out, config)
            with self.assertRaisesRegex(ValueError, "not empty"):
                prepare_training_output(out, config)
            with self.assertRaisesRegex(ValueError, "No Trainer checkpoint"):
                prepare_training_output(out, config, resume=True)
            (out / "checkpoint-25").mkdir()
            prepare_training_output(out, config, resume=True)
            with self.assertRaisesRegex(ValueError, "changed"):
                prepare_training_output(out, {**config, "target": "rationale"}, resume=True)


if __name__ == "__main__":
    unittest.main()

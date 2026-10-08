"""Standard-library safeguards for explicit, configuration-matched resume."""
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_training_output(out, config, resume=False):
    """Reject occupied outputs or mismatched resumes before loading a model."""
    out = Path(out)
    record = out / "training_provenance.json"
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    payload = {"schema_version": 1, "fingerprint": signature, "configuration": config}
    if resume:
        if not record.is_file():
            raise ValueError("Resume requires training_provenance.json from this release. Use a new output directory for a new run.")
        old = json.loads(record.read_text())
        if old.get("fingerprint") != signature:
            raise ValueError("Training configuration or input data changed. Use a new output directory.")
        checkpoints = [p for p in out.glob("checkpoint-*") if p.is_dir()]
        if not checkpoints:
            raise ValueError("No Trainer checkpoint is available to resume. Adapter-only best folders are for inference.")
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is not empty. Use --resume for the identical interrupted run or choose a new --out.")
    out.mkdir(parents=True, exist_ok=True)
    temporary = record.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(record)

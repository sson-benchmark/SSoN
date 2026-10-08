"""Prevent an evaluation output from mixing incompatible run configurations.

The immutable ``<output>.config.json`` sidecar records a canonical SHA-256
fingerprint. Local adapter configuration and weight files are hashed in full.
Remote Hub IDs are recorded as IDs, not resolved revisions. The guard cannot
detect changes behind an unpinned remote ID, base weights, image bytes, or
software dependencies. For exact reruns, use immutable downloaded model
snapshots and the same dataset and environment. Never run simultaneous writers
against one output. Archived legacy results remain readable by report.py.
"""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path


class ProvenanceError(ValueError):
    """An output cannot safely be resumed with the requested configuration."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(config):
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def adapter_identity(adapter):
    if not adapter:
        return None
    path = Path(adapter).expanduser()
    if path.is_dir():
        config = path / "adapter_config.json"
        weights = sorted(set(path.glob("adapter_model*.safetensors")) |
                         set(path.glob("adapter_model*.bin")))
        if not config.is_file() or not weights:
            raise ProvenanceError(
                f"Local adapter {path} needs adapter_config.json and adapter weights.")
        # Parse the config as well as hashing it, so malformed files fail early.
        with config.open() as stream:
            settings = json.load(stream)
        if not isinstance(settings, dict):
            raise ProvenanceError(f"Local adapter configuration must be a JSON object at {config}.")
        files = [config] + weights + sorted(path.glob("adapter_model*.index.json"))
        return {"kind": "local", "path": str(path.resolve()), "config": settings,
                "files": {p.name: sha256_file(p) for p in files}}
    if path.exists() or str(adapter).startswith(("/", "./", "../", "~")) \
            or (path.parent != Path(".") and path.parent.exists()):
        raise ProvenanceError(f"Local adapter directory not found or invalid at {adapter}.")
    return {"kind": "hub", "id": str(adapter), "revision": None,
            "verification": "ID only, remote revision is not pinned"}


def evaluation_config(*, split, base, adapter, prompt_sha, protocol, version,
                      tag, options, source_files=()):
    """Build the guard input without contacting a model host or loading a GPU."""
    base_path = Path(base).expanduser()
    base_id = str(base_path.resolve()) if base_path.is_dir() else str(base)
    return {
        "base": {"id": base_id, "kind": "local" if base_path.is_dir() else "hub"},
        "adapter": adapter_identity(adapter),
        "split_sha256": sha256_file(split),
        "prompt_sha256": prompt_sha,
        "protocol": protocol,
        "version": version,
        "tag": tag,
        "options": options,
        "source_sha256": {Path(p).name: sha256_file(p) for p in source_files},
    }


def ensure_output_config(output, config):
    """Create the sidecar once, or verify it before any records are skipped.

    A same-config sidecar without an output is valid after a pre-inference
    interruption. Empty outputs may be initialized. Nonempty legacy outputs,
    changed settings, corrupt sidecars, and changed adapter files are rejected.
    The output itself is never modified by this function.
    """
    output = Path(output)
    sidecar = Path(str(output) + ".config.json")
    if output.exists() and not output.is_file():
        raise ProvenanceError(f"Evaluation output is not a file at {output}.")
    wanted = fingerprint(config)

    def fail(reason):
        raise ProvenanceError(
            f"Cannot resume {output}. {reason} Choose a new --out path. "
            "Existing results were not changed. Legacy results can still be "
            "analyzed with report.py.")

    def validate_existing():
        try:
            with sidecar.open() as stream:
                saved = json.load(stream)
            saved_config = saved["config"]
            if not isinstance(saved_config, dict):
                raise TypeError("config must be a JSON object")
            saved_fingerprint = fingerprint(saved_config)
        except (OSError, KeyError, TypeError, AttributeError, ValueError) as exc:
            fail(f"Its configuration sidecar cannot be read ({exc}).")
        if saved.get("schema_version") != 1 or saved.get("fingerprint") != saved_fingerprint:
            fail("Its configuration sidecar is invalid.")
        if saved["fingerprint"] != wanted:
            changed = sorted(k for k in set(saved_config) | set(config)
                             if saved_config.get(k) != config.get(k))
            fail("Run configuration changed in " + ", ".join(changed) + ".")

    if sidecar.exists():
        validate_existing()
    else:
        if output.exists() and output.stat().st_size:
            fail("This nonempty output has no configuration sidecar.")
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        document = {"schema_version": 1, "fingerprint": wanted, "config": config}
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                             dir=sidecar.parent,
                                             prefix=".run-config-", delete=False) as stream:
                tmp_path = Path(stream.name)
                json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            # Link a completed file into place without replacing another writer's
            # sidecar. This is atomic and both paths are on the same filesystem.
            try:
                os.link(tmp_path, sidecar)
            except FileExistsError:
                validate_existing()
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    adapter = config.get("adapter") or {}
    if adapter.get("kind") == "hub" or config.get("base", {}).get("kind") == "hub":
        print("Run guard records remote model IDs without pinning Hub revisions. "
              "Use immutable local snapshots for stronger reproducibility.", file=sys.stderr)
    return wanted

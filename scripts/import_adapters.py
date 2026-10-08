#!/usr/bin/env python3
"""Import selected PEFT adapters from a local checkpoint archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ("ans-s1", "rat-s1")
BASE = "Qwen/Qwen2.5-VL-7B-Instruct"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_weights(path):
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated Safetensors header")
        size = struct.unpack("<Q", prefix)[0]
        if size > 8 * 1024 * 1024:
            raise ValueError("Unexpectedly large Safetensors header")
        header = json.loads(handle.read(size))
    tensors = [v for k, v in header.items() if k != "__metadata__"]
    total, spans = 0, []
    for value in tensors:
        if value["dtype"] != "F32":
            raise ValueError("The released adapter must use F32 tensors")
        count = 1
        for dim in value["shape"]:
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError("Invalid tensor shape")
            count *= dim
        start, end = value["data_offsets"]
        if end - start != count * 4:
            raise ValueError("Tensor byte count differs from its shape")
        total += count
        spans.append((start, end))
    cursor = 0
    for start, end in sorted(spans):
        if start != cursor:
            raise ValueError("Non-contiguous or overlapping tensor payload")
        cursor = end
    if path.stat().st_size != 8 + size + cursor:
        raise ValueError("Safetensors payload length mismatch")
    if len(tensors) != 392 or total != 40_370_176:
        raise ValueError("Adapter dimensions differ from the supplied release")
    return {"tensors": len(tensors), "parameters": total}


def import_archive(archive, out):
    archive, out = Path(archive).resolve(), Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"source_zip_sha256": digest(archive), "adapters": {}}
    with zipfile.ZipFile(archive) as source, tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        staged = []
        names = source.namelist()
        for name in ADAPTERS:
            files = {}
            for filename in ("adapter_config.json", "adapter_model.safetensors"):
                suffix = f"ckpts/{name}/best/{filename}"
                matches = [p for p in names if p == suffix or p.endswith("/" + suffix)]
                if len(matches) != 1:
                    raise ValueError("Expected one ZIP member ending in " + suffix)
                path = Path(tmp) / name / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.open(matches[0]) as src, path.open("wb") as dest:
                    shutil.copyfileobj(src, dest)
                files[filename] = path
            config = json.loads(files["adapter_config.json"].read_text())
            if config.get("base_model_name_or_path") != BASE or config.get("r") != 16:
                raise ValueError("Unexpected base model or LoRA rank for " + name)
            info = check_weights(files["adapter_model.safetensors"])
            info["files"] = {}
            for filename, path in files.items():
                sha = digest(path)
                destination = out / name / "best" / filename
                if destination.exists() and digest(destination) != sha:
                    raise ValueError("Refusing to overwrite different adapter at " + str(destination))
                info["files"][filename] = {"sha256": sha, "bytes": path.stat().st_size}
                staged.append((path, destination))
            report["adapters"][name] = info
        for src, dest in staged:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.move(str(src), str(dest))
        (out / "import_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "ckpts")
    args = parser.parse_args()
    try:
        report = import_archive(args.archive, args.out)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        parser.error(str(exc))
    for name, info in report["adapters"].items():
        print(name, info["parameters"], "parameters imported")
    print("The source ZIP is unchanged. No model inference was performed.")


if __name__ == "__main__":
    main()

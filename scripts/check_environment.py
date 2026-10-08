#!/usr/bin/env python3
"""Report the environment required by the recorded single-GPU pipeline."""
import importlib.metadata
import json
import platform
import sys


def main():
    report = {"python": platform.python_version(), "packages": {}, "errors": []}
    for name in ("torch", "transformers", "peft", "accelerate", "pillow", "numpy"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["errors"].append("Missing package " + name)
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from peft import LoraConfig

        report["cuda_available"] = torch.cuda.is_available()
        report["cuda_runtime"] = torch.version.cuda
        report["bf16_supported"] = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            report["gpu"] = {"name": prop.name, "memory_gib": round(prop.total_memory / 2**30, 2)}
        if not report["bf16_supported"]:
            report["errors"].append("Training needs a CUDA GPU with BF16 support")
        try:
            from torch.nn.attention.varlen import varlen_attn
            from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as model_impl

            required = ("Qwen2_5_VLVisionAttention", "apply_rotary_pos_emb_vision", "is_flash_attention_requested")
            report["optional_varlen_imports"] = all(hasattr(model_impl, x) for x in required)
        except (ImportError, AttributeError):
            report["optional_varlen_imports"] = False
        if not report["optional_varlen_imports"]:
            report["attention_note"] = "Use --standard-sdpa or install the recorded compatible environment"
    except Exception as exc:
        report["errors"].append(type(exc).__name__ + " " + str(exc))
    report["check_scope"] = "Dependency imports and GPU capability only. No model was loaded."
    print(json.dumps(report, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())

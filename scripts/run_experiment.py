#!/usr/bin/env python3
"""Run the recorded single-GPU configuration or print its commands."""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_prepared_data(root, settings_by_stage):
    data_dir = root / "data"
    report_path = data_dir / "official_preparation.json"
    if not report_path.is_file():
        raise ValueError("First run prepare_official_data.py with the complete bank. See README.md.")
    report = json.loads(report_path.read_text())
    if report.get("training_ready") is not True:
        raise ValueError("Data was prepared without image verification. Rerun preparation with image checks before model execution.")
    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "train_aug.jsonl"):
        path = data_dir / name
        expected = report.get("outputs", {}).get(name, {}).get("sha256")
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Prepared file changed or is missing. Rerun preparation for " + name)
    for stage, settings in settings_by_stage:
        expected = {"train": "train_aug.jsonl", "val": "val.jsonl"} if stage == "train" else {"split": "test.jsonl"}
        for field, name in expected.items():
            path = Path(settings.get(field, ""))
            path = path if path.is_absolute() else root / path
            if path.resolve() != (data_dir / name).resolve():
                raise ValueError("Official runner requires " + str(data_dir / name) + " for " + field)


def command(script, settings):
    argv = [sys.executable, str(ROOT / script)]
    for key, value in settings.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        elif value is not None and value is not False:
            argv.extend([flag, str(value)])
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--run-dir", help="use a new training directory and its best adapter")
    parser.add_argument("--adapter", help="use a supplied adapter for eval instead of the configured training output")
    parser.add_argument("--eval-out", help="write reevaluation into a new result path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--standard-sdpa", action="store_true", help="disable the optional recorded varlen optimization")
    parser.add_argument("--dry-run", action="store_true", help="print commands without data/model loading")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text())
    stages = ["train", "eval"] if args.stage == "all" else [args.stage]
    jobs, settings_by_stage = [], []
    for stage in stages:
        if stage not in config:
            raise SystemExit("This configuration has no " + stage + " stage. Choose --stage eval for the baseline.")
        settings = dict(config[stage])
        if args.run_dir:
            if stage == "train":
                settings["out"] = args.run_dir
            elif "adapter" in settings:
                settings["adapter"] = str(Path(args.run_dir) / "best")
        if args.standard_sdpa:
            settings["vit_varlen"] = False
        if stage == "train" and args.resume:
            settings["resume"] = True
        if stage == "eval":
            if args.adapter:
                settings["adapter"] = args.adapter
            if args.eval_out:
                settings["out"] = args.eval_out
        jobs.append(command("train_lora.py" if stage == "train" else "eval_local.py", settings))
        settings_by_stage.append((stage, settings))
    if not args.dry_run:
        try:
            verify_prepared_data(ROOT, settings_by_stage)
        except (ValueError, OSError) as exc:
            raise SystemExit(str(exc)) from exc
    env = os.environ.copy()
    if args.bank:
        env["ODDOUT_BANK"] = str(args.bank.resolve())
    for argv in jobs:
        print(shlex.join(argv), flush=True)
        if not args.dry_run:
            subprocess.run(argv, cwd=ROOT, env=env, check=True)
    if args.dry_run:
        print("Dry run only. No images or models were loaded.")


if __name__ == "__main__":
    main()

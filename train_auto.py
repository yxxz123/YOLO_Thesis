from __future__ import annotations

import argparse
import copy
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent


@dataclass
class JobResult:
    model: str
    command: list[str]
    returncode: int
    elapsed_seconds: float


@dataclass
class AutoTrainConfig:
    models: list[str] = field(default_factory=list)
    data: str | None = None
    epochs: int | None = None
    batch: int | float | None = None
    imgsz: int | str | None = None
    device: str | int | None = None
    project: str | None = "train_auto"
    name_template: str = "{stem}"
    swanlab: bool = False
    swanlab_project: str = "Ultralytics-YOLO"
    swanlab_run_name: str | None = None
    swanlab_mode: str = "cloud"
    swanlab_log_interval: int = 10
    stop_on_fail: bool = False
    dry_run: bool = False
    extra: dict[str, object] = field(default_factory=dict)
    baseline_monitor: bool = False
    baseline_path: str | None = None
    baseline_metric: str = "metrics/mAP50-95(B)"
    baseline_start_ratio: float = 0.5
    baseline_min_gap: float = 0.03
    baseline_patience: int = 1


AUTO_TRAIN_CONFIG = AutoTrainConfig(
    models=[
        # "ultralytics/cfg/models/26/yolo26n.yaml",
    ],
    data="/root/dataset/dataset_visdrone/data.yaml",
    epochs=300,
    batch=16,
    imgsz=640,
    device="0",
    project="train_auto",
    name_template="{stem}",
    swanlab=False,
    swanlab_project="Ultralytics-YOLO",
    swanlab_run_name=None,
    swanlab_mode="cloud",
    swanlab_log_interval=10,
    extra={
        "cache": False,
        "workers": 4,
        "patience": 50,
        "amp": True,
    },
)


def _format_yolo_arg(key: str, value: object) -> str:
    return f"{key}={value}"


def _name_from_template(template: str, model: str, index: int = 1) -> str:
    path = Path(model)
    return template.format(index=index, stem=path.stem, name=path.name, model=model)


def build_train_command(config: AutoTrainConfig, model: str, index: int = 1) -> list[str]:
    command: list[str] = [sys.executable, "-m", "ultralytics.cfg", "train"]
    args: dict[str, object] = {
        "model": model,
        "data": config.data,
        "epochs": config.epochs,
        "batch": config.batch,
        "imgsz": config.imgsz,
        "device": config.device,
        "project": config.project,
        "name": _name_from_template(config.name_template, model, index),
        "swanlab": config.swanlab,
        "swanlab_project": config.swanlab_project,
        "swanlab_run_name": config.swanlab_run_name,
        "swanlab_mode": config.swanlab_mode,
        "swanlab_log_interval": config.swanlab_log_interval,
    }
    if config.baseline_monitor:
        args.update(
            {
                "baseline_monitor": config.baseline_monitor,
                "baseline_path": config.baseline_path,
                "baseline_metric": config.baseline_metric,
                "baseline_start_ratio": config.baseline_start_ratio,
                "baseline_min_gap": config.baseline_min_gap,
                "baseline_patience": config.baseline_patience,
            }
        )
    args.update(config.extra)

    command.extend(_format_yolo_arg(key, value) for key, value in args.items() if value is not None)
    return command


def build_train_commands(config: AutoTrainConfig) -> list[list[str]]:
    return [build_train_command(config, model, index) for index, model in enumerate(config.models, start=1)]


def _parse_extra(items: Sequence[str] | None) -> dict[str, str]:
    extra: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--extra values must use key=value syntax: {item}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError(f"--extra key cannot be empty: {item}")
        extra[key] = value
    return extra


def _parse_int_or_str(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _parse_batch(value: str) -> int | float | str:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multiple Ultralytics YOLO training jobs sequentially.")
    parser.add_argument("--models", "-m", nargs="+", help="YAML or .pt model paths to train in order")
    parser.add_argument("--data", help="dataset YAML path")
    parser.add_argument("--epochs", type=int, help="training epochs")
    parser.add_argument("--batch", type=_parse_batch, help="batch size")
    parser.add_argument("--imgsz", type=_parse_int_or_str, help="image size")
    parser.add_argument("--device", help="training device")
    parser.add_argument("--project", help="Ultralytics project directory")
    parser.add_argument("--name-template", help="run name template, supports {index}, {stem}, {name}, and {model}")
    parser.add_argument("--swanlab", action="store_true", default=None, help="enable SwanLab training visualization")
    parser.add_argument("--swanlab-project", help="SwanLab project name")
    parser.add_argument("--swanlab-run-name", help="SwanLab experiment name")
    parser.add_argument("--swanlab-mode", help="SwanLab mode: cloud/online, local, offline, or disabled")
    parser.add_argument("--swanlab-log-interval", type=int, help="steps between SwanLab loss/lr logs; <=0 disables")
    parser.add_argument("--dry-run", action="store_true", default=None, help="print commands without launching training")
    parser.add_argument("--stop-on-fail", action="store_true", default=None, help="stop after first failed training job")
    parser.add_argument("--extra", nargs="+", help="additional YOLO key=value arguments")
    parser.add_argument("--baseline-monitor", action="store_true", default=None, help="enable baseline results.csv monitor")
    parser.add_argument("--baseline-path", help="baseline results.csv path")
    parser.add_argument("--baseline-metric", help="metric column to compare against the baseline")
    parser.add_argument("--baseline-start-ratio", type=float, help="fraction of epochs to wait before enforcing baseline")
    parser.add_argument("--baseline-min-gap", type=float, help="minimum allowed current-baseline gap before a bad epoch")
    parser.add_argument("--baseline-patience", type=int, help="consecutive bad comparison epochs before stopping")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace, default: AutoTrainConfig = AUTO_TRAIN_CONFIG) -> AutoTrainConfig:
    config = copy.deepcopy(default)
    for field_name in (
        "models",
        "data",
        "epochs",
        "batch",
        "imgsz",
        "device",
        "project",
        "swanlab_project",
        "swanlab_run_name",
        "swanlab_mode",
        "swanlab_log_interval",
        "baseline_path",
        "baseline_metric",
        "baseline_start_ratio",
        "baseline_min_gap",
        "baseline_patience",
    ):
        value = getattr(args, field_name)
        if value is not None:
            setattr(config, field_name, value)
    if args.name_template is not None:
        config.name_template = args.name_template
    if args.swanlab is not None:
        config.swanlab = args.swanlab
    if args.dry_run is not None:
        config.dry_run = args.dry_run
    if args.stop_on_fail is not None:
        config.stop_on_fail = args.stop_on_fail
    if args.baseline_monitor is not None:
        config.baseline_monitor = args.baseline_monitor
    if args.extra:
        config.extra.update(_parse_extra(args.extra))
    return config


def validate_config(config: AutoTrainConfig) -> None:
    if not config.models:
        raise ValueError("At least one model is required. Set AUTO_TRAIN_CONFIG.models or pass --models.")
    if not config.data and "data" not in config.extra:
        raise ValueError("Dataset path is required. Set data, pass --data, or include data=... in extra.")
    if config.baseline_monitor and not config.baseline_path:
        raise ValueError("baseline_path is required when baseline_monitor=True.")


def print_plan(config: AutoTrainConfig) -> None:
    print("[train_auto] plan:")
    print(f"[train_auto]   models: {len(config.models)}")
    print(f"[train_auto]   data: {config.data}")
    print(f"[train_auto]   epochs: {config.epochs}")
    print(f"[train_auto]   batch: {config.batch}")
    print(f"[train_auto]   imgsz: {config.imgsz}")
    print(f"[train_auto]   device: {config.device}")
    print(f"[train_auto]   project: {config.project}")
    print(f"[train_auto]   swanlab: {config.swanlab}")
    if config.swanlab:
        print(f"[train_auto]   swanlab_project: {config.swanlab_project}")
        print(f"[train_auto]   swanlab_run_name: {config.swanlab_run_name}")
        print(f"[train_auto]   swanlab_mode: {config.swanlab_mode}")
        print(f"[train_auto]   swanlab_log_interval: {config.swanlab_log_interval}")
    print(f"[train_auto]   baseline_monitor: {config.baseline_monitor}")
    if config.baseline_monitor:
        print(f"[train_auto]   baseline_path: {config.baseline_path}")
        print(f"[train_auto]   baseline_metric: {config.baseline_metric}")
        print(f"[train_auto]   baseline_start_ratio: {config.baseline_start_ratio}")
        print(f"[train_auto]   baseline_min_gap: {config.baseline_min_gap}")
        print(f"[train_auto]   baseline_patience: {config.baseline_patience}")
    for index, model in enumerate(config.models, start=1):
        print(f"[train_auto]   name[{index}]: {_name_from_template(config.name_template, model, index)}")


def _print_dry_run(commands: Sequence[Sequence[str]]) -> None:
    print("[train_auto] DRY-RUN: commands that would be executed:")
    for index, command in enumerate(commands, start=1):
        print(f"[train_auto]   {index}: {shlex.join(command)}")


def run_jobs(config: AutoTrainConfig) -> list[JobResult]:
    results: list[JobResult] = []
    commands = build_train_commands(config)
    total = len(commands)

    for index, (model, command) in enumerate(zip(config.models, commands), start=1):
        print(f"[train_auto] start {index}/{total}: {model}", flush=True)
        print(f"[train_auto] command: {shlex.join(command)}", flush=True)
        started_at = time.monotonic()
        completed = subprocess.run(command, cwd=str(ROOT), check=False)
        elapsed_seconds = time.monotonic() - started_at

        result = JobResult(model=model, command=command, returncode=completed.returncode, elapsed_seconds=elapsed_seconds)
        results.append(result)
        print(
            f"[train_auto] finish {index}/{total}: {model} "
            f"returncode={result.returncode} elapsed={result.elapsed_seconds:.1f}s",
            flush=True,
        )

        if result.returncode != 0 and config.stop_on_fail:
            print("[train_auto] stop-on-fail enabled; remaining jobs skipped.", flush=True)
            break

    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args, AUTO_TRAIN_CONFIG)
    validate_config(config)
    commands = build_train_commands(config)
    print_plan(config)

    if config.dry_run:
        _print_dry_run(commands)
        return 0

    try:
        results = run_jobs(config)
    except KeyboardInterrupt:
        print("[train_auto] interrupted by user.", flush=True)
        return 130

    failed = [result for result in results if result.returncode != 0]
    print(
        f"[train_auto] summary: total={len(results)} failed={len(failed)} "
        f"skipped={len(config.models) - len(results)}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

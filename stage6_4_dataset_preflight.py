#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 6.4 dataset preflight.

Purpose:
- Find train / val / test image and YOLO-label directories.
- Audit label integrity and infer source / sequence IDs.
- Detect sequence overlap between train, val and test.
- Create a test manifest for the later frozen-selector independent evaluation.

This script DOES NOT run YOLO inference, tune a selector, or evaluate AP.
Do not use the current Stage 6.3 validation set as a substitute for test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def list_images(folder: Path) -> list[Path]:
    if not folder or not folder.is_dir():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def choose_existing_dir(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.is_dir() and list_images(path):
            return path.resolve()
    return None


def discover_split(
    root: Path,
    split: str,
    image_override: str | None,
    label_override: str | None,
) -> tuple[Path | None, Path | None]:
    if image_override:
        image_dir = Path(image_override).expanduser().resolve()
    else:
        image_dir = choose_existing_dir([
            root / "images" / split,
            root / split / "images",
            root / split,
        ])

    if label_override:
        label_dir = Path(label_override).expanduser().resolve()
    else:
        label_dir = None
        if image_dir:
            candidates = [
                root / "labels" / split,
                root / split / "labels",
            ]
            if image_dir.name == "images":
                candidates.insert(0, image_dir.parent / "labels")
            label_dir = next((p.resolve() for p in candidates if p.is_dir()), None)

    return image_dir, label_dir


def infer_source(path: Path, source_map: dict[str, str]) -> str:
    keys = [
        path.name.lower(),
        path.stem.lower(),
        path.as_posix().lower(),
    ]
    for key in keys:
        if key in source_map:
            return source_map[key]

    text = " ".join([path.name.lower(), *[p.lower() for p in path.parts]])
    has_uavdt = "uavdt" in text
    has_visdrone = "visdrone" in text
    if has_uavdt and not has_visdrone:
        return "uavdt"
    if has_visdrone and not has_uavdt:
        return "visdrone"
    return "unknown"


def sequence_id_from_name(path: Path) -> str:
    stem = path.stem
    # Typical source file: uavdt_M0209_000001 -> sequence uavdt_M0209.
    # Remove only a final numeric frame/index token.
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and re.fullmatch(r"\d+", parts[1]):
        return parts[0]
    return stem


def resolve_label(image_path: Path, image_dir: Path, label_dir: Path) -> Path:
    relative = image_path.relative_to(image_dir)
    candidate = (label_dir / relative).with_suffix(".txt")
    if candidate.is_file():
        return candidate
    flat = label_dir / f"{image_path.stem}.txt"
    if flat.is_file():
        return flat
    return candidate


def validate_yolo_label(path: Path) -> tuple[int, list[str]]:
    if not path.is_file():
        return 0, ["missing_label_file"]

    errors: list[str] = []
    object_count = 0
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except Exception as exc:
        return 0, [f"label_read_error:{type(exc).__name__}"]

    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line_{line_no}_expected_5_values_got_{len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
        except Exception:
            errors.append(f"line_{line_no}_non_numeric")
            continue
        if cls < 0:
            errors.append(f"line_{line_no}_negative_class")
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
            errors.append(f"line_{line_no}_coordinate_out_of_0_1")
        if w <= 0 or h <= 0:
            errors.append(f"line_{line_no}_non_positive_wh")
        object_count += 1

    return object_count, errors


def load_source_map(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    mapping: dict[str, str] = {}
    with Path(path).expanduser().open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"file_name", "source"}.issubset(reader.fieldnames):
            raise RuntimeError("source-map CSV必须包含file_name,source两列")
        for row in reader:
            source = row["source"].strip().lower()
            if source not in {"uavdt", "visdrone"}:
                raise RuntimeError(f"source-map中存在未知source：{source}")
            name = Path(row["file_name"]).name.lower()
            mapping[name] = source
            mapping[Path(name).stem] = source
    return mapping


def audit_one_split(
    split: str,
    image_dir: Path | None,
    label_dir: Path | None,
    source_map: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    if image_dir is None or not image_dir.is_dir():
        return {
            "split": split,
            "found": False,
            "image_dir": str(image_dir) if image_dir else None,
            "label_dir": str(label_dir) if label_dir else None,
            "images": 0,
            "status": "split_not_found",
        }, [], set()

    images = list_images(image_dir)
    if not images:
        return {
            "split": split,
            "found": False,
            "image_dir": str(image_dir),
            "label_dir": str(label_dir) if label_dir else None,
            "images": 0,
            "status": "no_images",
        }, [], set()

    rows: list[dict[str, Any]] = []
    source_counts: Counter = Counter()
    group_ids: set[str] = set()
    missing_labels = 0
    bad_labels = 0
    empty_labels = 0
    unreadable_images = 0
    unknown_sources = 0
    total_objects = 0

    for image_id, image_path in enumerate(images, start=1):
        source = infer_source(image_path, source_map)
        source_counts[source] += 1
        if source == "unknown":
            unknown_sources += 1

        sequence_id = sequence_id_from_name(image_path)
        group_id = f"{source}:{sequence_id}"
        group_ids.add(group_id)

        label_path = resolve_label(image_path, image_dir, label_dir) if label_dir else None
        if label_path is None:
            object_count, errors = 0, ["label_directory_not_found"]
        else:
            object_count, errors = validate_yolo_label(label_path)

        if "missing_label_file" in errors:
            missing_labels += 1
        if any(not e.startswith("missing_label_file") for e in errors):
            bad_labels += 1
        if label_path and label_path.is_file() and object_count == 0 and not errors:
            empty_labels += 1
        total_objects += object_count

        try:
            with Image.open(image_path) as im:
                width, height = im.size
                im.verify()
        except Exception:
            width, height = 0, 0
            unreadable_images += 1

        rows.append({
            "split": split,
            "image_id": image_id,
            "file_name": image_path.name,
            "relative_path": str(image_path.relative_to(image_dir)).replace("\\", "/"),
            "absolute_path": str(image_path.resolve()),
            "source": source,
            "sequence_id": sequence_id,
            "group_id": group_id,
            "width": width,
            "height": height,
            "label_path": str(label_path.resolve()) if label_path else "",
            "label_exists": int(bool(label_path and label_path.is_file())),
            "object_count": object_count,
            "label_errors": ";".join(errors),
        })

    status = "passed"
    if not label_dir or not label_dir.is_dir():
        status = "missing_label_directory"
    elif missing_labels or bad_labels:
        status = "label_audit_failed"
    elif unreadable_images:
        status = "unreadable_images"
    elif unknown_sources:
        status = "unknown_source"

    result = {
        "split": split,
        "found": True,
        "image_dir": str(image_dir.resolve()),
        "label_dir": str(label_dir.resolve()) if label_dir else None,
        "images": len(images),
        "source_counts": dict(source_counts),
        "sequence_groups": len(group_ids),
        "missing_label_files": missing_labels,
        "invalid_label_files": bad_labels,
        "empty_label_files": empty_labels,
        "unreadable_images": unreadable_images,
        "unknown_source_images": unknown_sources,
        "total_objects": total_objects,
        "status": status,
    }
    return result, rows, group_ids


def overlap_report(group_sets: dict[str, set[str]]) -> dict[str, Any]:
    pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]
    out = {}
    for a, b in pairs:
        ga, gb = group_sets.get(a, set()), group_sets.get(b, set())
        overlap = sorted(ga & gb)
        out[f"{a}_vs_{b}"] = {
            "groups_a": len(ga),
            "groups_b": len(gb),
            "overlap_groups": len(overlap),
            "overlap_preview": overlap[:30],
            "passed": len(overlap) == 0,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit HighwayVehicle train/val/test split and sequence leakage."
    )
    parser.add_argument("--dataset-root", default="datasets/HighwayVehicle-v1.0")
    parser.add_argument("--output-dir", default="runs/detect/stage6_scene_selector/stage6_4_preflight")
    parser.add_argument("--train-image-dir")
    parser.add_argument("--train-label-dir")
    parser.add_argument("--val-image-dir")
    parser.add_argument("--val-label-dir")
    parser.add_argument("--test-image-dir")
    parser.add_argument("--test-label-dir")
    parser.add_argument("--source-map", default=None,
                        help="Optional CSV with file_name,source columns.")
    args = parser.parse_args()

    root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在：{root}")

    source_map = load_source_map(args.source_map)

    overrides = {
        "train": (args.train_image_dir, args.train_label_dir),
        "val": (args.val_image_dir, args.val_label_dir),
        "test": (args.test_image_dir, args.test_label_dir),
    }

    split_results: dict[str, Any] = {}
    split_rows: dict[str, list[dict[str, Any]]] = {}
    group_sets: dict[str, set[str]] = {}

    for split in ("train", "val", "test"):
        image_dir, label_dir = discover_split(root, split, *overrides[split])
        result, rows, groups = audit_one_split(
            split, image_dir, label_dir, source_map
        )
        split_results[split] = result
        split_rows[split] = rows
        group_sets[split] = groups

    overlaps = overlap_report(group_sets)
    test = split_results["test"]

    if not test.get("found"):
        overall_status = "NO_TEST_SPLIT"
    elif test["status"] != "passed":
        overall_status = "TEST_AUDIT_FAILED"
    elif not split_results["train"].get("found") or not split_results["val"].get("found"):
        overall_status = "TRAIN_OR_VAL_SPLIT_NOT_FOUND"
    elif not overlaps["train_vs_test"]["passed"] or not overlaps["val_vs_test"]["passed"]:
        overall_status = "TEST_NOT_SEQUENCE_INDEPENDENT"
    else:
        overall_status = "READY_FOR_STAGE6_4"

    test_manifest = output_dir / "test_manifest.csv"
    if split_rows["test"]:
        fields = list(split_rows["test"][0].keys())
        with test_manifest.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(split_rows["test"])

    summary = {
        "experiment": "Stage6.4_Dataset_Preflight",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(root),
        "status": overall_status,
        "split_audit": split_results,
        "sequence_overlap": overlaps,
        "test_manifest": str(test_manifest) if split_rows["test"] else None,
        "requirements_before_stage6_4": [
            "A test split must exist and must not be the Stage 6.3 validation split.",
            "YOLO test labels must be present and valid.",
            "Test source must be identifiable as uavdt or visdrone.",
            "Test sequence IDs must not overlap either train or val sequence IDs.",
            "Do not refit the frozen M3 selector or change its probability threshold on test.",
        ],
    }

    summary_path = output_dir / "dataset_audit.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 78)
    print("Stage 6.4 dataset preflight")
    print("Dataset:", root)
    for split in ("train", "val", "test"):
        x = split_results[split]
        print(
            f"{split:5s}: found={x.get('found')} "
            f"images={x.get('images', 0)} "
            f"groups={x.get('sequence_groups', 0)} "
            f"sources={x.get('source_counts', {})} "
            f"missing_labels={x.get('missing_label_files', 'NA')} "
            f"invalid_labels={x.get('invalid_label_files', 'NA')} "
            f"status={x.get('status')}"
        )
    for name, x in overlaps.items():
        print(f"{name}: overlap_groups={x['overlap_groups']} passed={x['passed']}")
    print("OVERALL:", overall_status)
    print("Audit JSON:", summary_path)
    if split_rows["test"]:
        print("Test manifest:", test_manifest)
    print("=" * 78)

    return 0 if overall_status == "READY_FOR_STAGE6_4" else 2


if __name__ == "__main__":
    raise SystemExit(main())

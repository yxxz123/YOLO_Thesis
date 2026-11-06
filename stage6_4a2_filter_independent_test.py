#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 6.4A2: filter the test split to sequence-independent images.

The original test split has sequence overlap with train and val. This script
removes the UNION of all test sequence groups that overlap either split.
It does not modify images or labels and does not run inference.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "runs/detect/stage6_scene_selector/stage6_4_preflight"
AUDIT_JSON = BASE / "dataset_audit.json"
TEST_MANIFEST = BASE / "test_manifest.csv"
OUT_DIR = BASE

EXPECTED_TOTAL = 1536
EXPECTED_CLEAN_TOTAL = 1372
EXPECTED_SOURCE_COUNTS = {"uavdt": 1288, "visdrone": 84}
EXPECTED_CLEAN_GROUPS = 89


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"拒绝写入空CSV：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


def main() -> int:
    if not AUDIT_JSON.is_file() or not TEST_MANIFEST.is_file():
        raise FileNotFoundError(
            "缺少Stage 6.4A输出：dataset_audit.json或test_manifest.csv"
        )

    audit = read_json(AUDIT_JSON)
    rows = read_csv(TEST_MANIFEST)

    if audit.get("status") != "TEST_NOT_SEQUENCE_INDEPENDENT":
        raise RuntimeError(
            f"预期原始审计状态为TEST_NOT_SEQUENCE_INDEPENDENT，"
            f"实际为{audit.get('status')}"
        )
    if len(rows) != EXPECTED_TOTAL:
        raise RuntimeError(f"原始test manifest应有{EXPECTED_TOTAL}行，实际{len(rows)}")

    # The preflight script stores every overlapping group in overlap_preview
    # when the overlap count is <= 30. Refuse to proceed if the list is truncated.
    train_overlap = audit["sequence_overlap"]["train_vs_test"]
    val_overlap = audit["sequence_overlap"]["val_vs_test"]

    for name, item in (
        ("train_vs_test", train_overlap),
        ("val_vs_test", val_overlap),
    ):
        count = int(item["overlap_groups"])
        preview = item.get("overlap_preview", [])
        if count != len(preview):
            raise RuntimeError(
                f"{name} overlap_preview不完整：count={count}, preview={len(preview)}"
            )

    train_groups = set(train_overlap["overlap_preview"])
    val_groups = set(val_overlap["overlap_preview"])
    excluded_groups = train_groups | val_groups

    if len(excluded_groups) != 18:
        raise RuntimeError(
            f"交叉序列并集应为18组，实际为{len(excluded_groups)}组；"
            "请重新检查序列ID解析，不要继续测试"
        )

    test_ids = set()
    all_groups = set()
    clean_rows = []
    excluded_rows = []

    for row in rows:
        image_id = int(row["image_id"])
        if image_id in test_ids:
            raise RuntimeError(f"test manifest存在重复image_id={image_id}")
        test_ids.add(image_id)

        if row["split"] != "test":
            raise RuntimeError(f"manifest包含非test记录：image_id={image_id}")

        source = row["source"].strip().lower()
        if source not in {"uavdt", "visdrone"}:
            raise RuntimeError(f"未知数据来源：image_id={image_id}, source={source}")

        sequence_id = row["sequence_id"].strip()
        expected_group = f"{source}:{sequence_id}"
        if row["group_id"] != expected_group:
            raise RuntimeError(
                f"group_id与source/sequence_id不一致：image_id={image_id}"
            )

        if row["label_exists"] != "1" or row["label_errors"].strip():
            raise RuntimeError(f"标签检查未通过：image_id={image_id}")

        if int(row["width"]) <= 0 or int(row["height"]) <= 0:
            raise RuntimeError(f"图片尺寸非法：image_id={image_id}")

        all_groups.add(row["group_id"])
        if row["group_id"] in excluded_groups:
            excluded_rows.append(row)
        else:
            clean_rows.append(row)

    clean_source_counts = Counter(row["source"] for row in clean_rows)
    excluded_source_counts = Counter(row["source"] for row in excluded_rows)
    clean_group_ids = {row["group_id"] for row in clean_rows}

    # Final leakage guards.
    if clean_group_ids & train_groups:
        raise RuntimeError("过滤后仍有test/train序列交叉")
    if clean_group_ids & val_groups:
        raise RuntimeError("过滤后仍有test/val序列交叉")

    if len(clean_rows) != EXPECTED_CLEAN_TOTAL:
        raise RuntimeError(
            f"独立test图片数应为{EXPECTED_CLEAN_TOTAL}，实际为{len(clean_rows)}"
        )
    if dict(clean_source_counts) != EXPECTED_SOURCE_COUNTS:
        raise RuntimeError(
            f"独立test来源数量异常：{dict(clean_source_counts)}"
        )
    if len(clean_group_ids) != EXPECTED_CLEAN_GROUPS:
        raise RuntimeError(
            f"独立test序列组数应为{EXPECTED_CLEAN_GROUPS}，"
            f"实际为{len(clean_group_ids)}"
        )
    if len(excluded_rows) != 164 or dict(excluded_source_counts) != {"visdrone": 164}:
        raise RuntimeError(
            f"排除图片数量异常：n={len(excluded_rows)}, "
            f"source_counts={dict(excluded_source_counts)}"
        )

    independent_path = OUT_DIR / "independent_test_manifest.csv"
    excluded_path = OUT_DIR / "excluded_sequence_overlap.csv"
    summary_path = OUT_DIR / "sequence_filter_summary.json"

    write_csv(independent_path, clean_rows)
    write_csv(excluded_path, excluded_rows)

    summary = {
        "experiment": "Stage6.4A2_Sequence_Independent_Test_Filter",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "READY_FOR_STAGE6_4B",
        "original_test_images": len(rows),
        "excluded_overlap_images": len(excluded_rows),
        "excluded_overlap_groups": len(excluded_groups),
        "excluded_source_counts": dict(excluded_source_counts),
        "independent_test_images": len(clean_rows),
        "independent_test_source_counts": dict(clean_source_counts),
        "independent_test_groups": len(clean_group_ids),
        "independent_test_group_source_counts": dict(
            Counter(g.split(":", 1)[0] for g in clean_group_ids)
        ),
        "independent_test_visdrone_groups": len(
            [g for g in clean_group_ids if g.startswith("visdrone:")]
        ),
        "independent_test_uavdt_groups": len(
            [g for g in clean_group_ids if g.startswith("uavdt:")]
        ),
        "test_train_overlap_after_filter": 0,
        "test_val_overlap_after_filter": 0,
        "train_val_overlap_groups_remain_in_original_splits": int(
            audit["sequence_overlap"]["train_vs_val"]["overlap_groups"]
        ),
        "input_sha256": {
            "dataset_audit.json": sha256_file(AUDIT_JSON),
            "test_manifest.csv": sha256_file(TEST_MANIFEST),
        },
        "output_sha256": {
            "independent_test_manifest.csv": sha256_file(independent_path),
            "excluded_sequence_overlap.csv": sha256_file(excluded_path),
        },
        "interpretation_limit": (
            "The filtered independent test contains 1,288 UAVDT images from "
            "5 sequences and only 84 VisDrone images from 84 sequences. "
            "The M3 selector is activated only for VisDrone, so its independent "
            "selector evaluation sample is 84 images/sequences. Report this "
            "sample size explicitly; do not imply that all 1,372 images test "
            "the selector."
        ),
        "files": {
            "independent_manifest": str(independent_path.relative_to(ROOT)),
            "excluded_manifest": str(excluded_path.relative_to(ROOT)),
        },
    }
    write_json(summary_path, summary)

    print("=" * 78)
    print("Stage 6.4A2 completed")
    print("Original test:", len(rows))
    print("Excluded overlap:", len(excluded_rows), "images /", len(excluded_groups), "groups")
    print("Independent test:", len(clean_rows), dict(clean_source_counts))
    print("Independent sequence groups:", len(clean_group_ids))
    print("VisDrone selector test:", summary["independent_test_visdrone_groups"], "images/sequences")
    print("STATUS:", summary["status"])
    print("Manifest:", independent_path)
    print("Summary:", summary_path)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

r"""独立检查已生成的清单、原图哈希、子集覆盖和已知重复关系。

运行：.\.venv\Scripts\python.exe scripts\verify_data.py
可用 --processed-dir 指向重跑结果，用 --report-dir 保存验收日志 JSON。
不训练模型，也不根据测试集效果调整参数。
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

import numpy as np
import pandas as pd

from data_utils import CLASSES, SPLITS, read_attributions
from download_data import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_manifest(path):
    """读回 CSV 时保留字符串哈希/照片 ID，避免数字型 ID 或空文本被自动转换。"""
    return pd.read_csv(path, keep_default_na=False, dtype={
        "relative_path": str, "source_id": str, "phash": str, "sha256": str, "pixel_sha256": str})


def require(condition, message):
    """将不满足的数据约束转为清楚的异常；不依赖可被 -O 关闭的 assert。"""
    if not condition:
        raise ValueError(message)


def verify_invariants(frame, distance):
    """只根据清单检查互斥与覆盖，并重新两两比较跨子集 pHash，不调用分组算法。"""
    require(frame.relative_path.is_unique, "清单中存在重复文件路径。")
    active = frame[frame.included]
    require(len(active) > 0 and set(active.split) == set(SPLITS), "三个子集必须均有样本。")
    require((active.status == "valid").all(), "无效图片进入了训练/验证/测试子集。")
    require((frame.loc[~frame.included, "split"] == "excluded").all(), "排除样本仍被分配到子集。")
    require((frame.loc[frame.included, "exclusion_reason"] == "").all(), "保留样本含排除原因。")
    expected_labels = frame.class_name.map(dict(zip(CLASSES, range(len(CLASSES)))))
    require((frame.label == expected_labels).all(), "类别与标签映射不一致。")
    for split in SPLITS:
        require(set(active[active.split == split].class_name) == set(CLASSES), f"{split} 缺少类别。")
    for column in ["duplicate_group", "sha256", "pixel_sha256", "source_id"]:
        require((active.groupby(column).split.nunique() == 1).all(), f"{column} 存在跨子集泄漏。")
    require(active.pixel_sha256.is_unique, "清洗后仍有完全相同像素副本。")
    hashes = [(row.relative_path, int(row.phash, 16), row.split) for row in active.itertuples()]
    for position, (left, left_hash, left_split) in enumerate(hashes):
        for right, right_hash, right_split in hashes[position + 1:]:
            if left_split != right_split and (left_hash ^ right_hash).bit_count() <= distance:
                raise ValueError(f"发现跨子集的 pHash 候选：{left} / {right}")
    return active


def verify(args):
    """读回处理产物，并与实际磁盘原图和人工复核配置交叉核对，返回验证报告。"""
    directory, raw = args.processed_dir.resolve(), args.raw_dir.resolve()
    frame = read_manifest(directory / "manifest.csv")
    config = json.loads((directory / "split_config.json").read_text(encoding="utf-8"))
    mapping = json.loads((directory / "class_to_idx.json").read_text(encoding="utf-8"))
    require(mapping == config["class_to_idx"] == dict(zip(CLASSES, range(len(CLASSES)))), "类别映射文件不一致。")
    active = verify_invariants(frame, config["phash_distance"])
    credits = read_attributions(raw / "LICENSE.txt")
    require(set(frame.relative_path) == set(credits), "清单与许可证图片列表不一致。")
    for row in frame.itertuples():
        relative = PurePosixPath(row.relative_path)
        require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in row.relative_path,
                "清单包含不安全路径。")
        path = (raw / row.relative_path).resolve()
        require(path.is_relative_to(raw), "图片路径超出原始数据目录。")
        require(path.is_file() and sha256_file(path) == row.sha256, f"图片缺失或内容变化：{row.relative_path}")
    fingerprint = hashlib.sha256("\n".join(frame.relative_path + ":" + frame.sha256).encode()).hexdigest()
    require(fingerprint == config["dataset_fingerprint"], "数据指纹与划分配置不一致。")
    for split in SPLITS:
        saved = read_manifest(directory / f"{split}.csv").reset_index(drop=True)
        expected = frame[frame.split == split].reset_index(drop=True)
        pd.testing.assert_frame_equal(saved, expected)
    reviewed = pd.read_csv(args.review_pairs, keep_default_na=False)
    require(sha256_file(args.review_pairs) == config["reviewed_pairs_sha256"], "人工复核配置已改变，请重新准备数据。")
    lookup = frame.set_index("relative_path")
    for pair in reviewed.itertuples(index=False):
        left, right = lookup.loc[pair.left], lookup.loc[pair.right]
        require(left.duplicate_group == right.duplicate_group, "人工复核图片未放入同一组。")
    report = {"status": "PASS", "verified_at": datetime.now().astimezone().isoformat(),
              "manifest_rows": len(frame), "included_images": len(active),
              "raw_files_hash_verified": len(frame), "attribution_coverage": len(credits),
              "split_counts": {split: int((active.split == split).sum()) for split in SPLITS},
              "known_duplicate_cross_split_pairs": 0, "manifest_sha256": sha256_file(directory / "manifest.csv"),
              "checks": ["类别映射与路径唯一", "子集互斥、覆盖与每类齐全", "无效/剔除样本隔离",
                         "原图逐文件 SHA-256", "字节/像素/照片 ID/重复组无跨集",
                         "独立两两检查跨子集 pHash", "人工复核关系", "子集 CSV 与总清单完全一致", "完整署名覆盖"]}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                      encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("DATA VERIFICATION PASSED")


def main():
    """命令行入口；任何约束失败都以非零退出码阻止误用数据。"""
    parser = argparse.ArgumentParser(description="核对数据清单和已知重复关系，独立于准备入口。")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw/flower_photos")
    parser.add_argument("--processed-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports/local/data")
    parser.add_argument("--review-pairs", type=Path, default=PROJECT_ROOT / "configs/duplicate_review.csv")
    verify(parser.parse_args())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except (OSError, ValueError, KeyError, AssertionError) as error:
        print(f"DATA VERIFICATION FAILED：{error}")
        sys.exit(1)

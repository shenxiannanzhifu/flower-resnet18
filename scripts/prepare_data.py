r"""v0.2 扫描花卉图片、去除确定重复、整组划分并输出真实统计和图表。

运行：.\.venv\Scripts\python.exe scripts\prepare_data.py
验收快照：添加 --report-dir reports/v0.2；日常运行默认写 reports/local/data。
只生成清单和报告，不复制/修改原图，不实现数据加载器或模型训练。
"""

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from data_utils import CLASSES, SPLITS, RATIOS, LICENSE_URL
from data_utils import inspect_image, read_attributions, group_duplicates, split_groups
from download_data import EXPECTED_SHA256, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 在导入 Matplotlib 之前配置可写缓存，避免 Windows 沙箱写用户主目录失败。
CACHE_DIR = PROJECT_ROOT / "reports/local/matplotlib-cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_json(path, value):
    """输入路径和可序列化值；统一 UTF-8/LF，便于版本比较与跨平台读取。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_csv(path, frame):
    """输入路径和 Pandas 表，固定行结束符，保证相同数据得到相同清单哈希。"""
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def plot_statistics(frame, counts, report_dir):
    """输入审计表及分组计数表，绘制类别分布/子集分布与尺寸分析图。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    positions = np.arange(len(CLASSES))
    for offset, column, color in [(-0.2, "raw", "#adbdc7"), (0.2, "included", "#238b8d")]:
        bars = axes[0].bar(positions + offset, counts[column], width=0.38, color=color, label=column)
        axes[0].bar_label(bars, fontsize=8, padding=3)
    axes[0].set(xticks=positions, xticklabels=CLASSES, title="Class counts: before / after cleaning", ylabel="Images")
    axes[0].set_ylim(0, counts.raw.max() * 1.16)
    axes[0].legend(frameon=False)
    bottom = np.zeros(len(CLASSES))
    for split, color in zip(SPLITS, ["#238b8d", "#e9ad4d", "#7068ad"]):
        values = counts[split].to_numpy()
        axes[1].bar(CLASSES, values, bottom=bottom, color=color, label=split)
        for x, y, height in zip(positions, bottom, values):
            axes[1].text(x, y + height / 2, str(height), ha="center", va="center", color="white", fontsize=9)
        bottom += values
    axes[1].set(title="Group-aware stratified split", ylabel="Images")
    axes[1].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    axes[1].set_ylim(0, bottom.max() * 1.18)
    fig.savefig(report_dir / "class_distribution.png", dpi=150)
    plt.close(fig)

    valid = frame[frame.status == "valid"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].scatter(valid.width, valid.height, alpha=0.18, s=12, color="#238b8d")
    axes[0].set(xlabel="Width (pixels)", ylabel="Height (pixels)", title="Original image dimensions")
    axes[1].hist(valid.width.to_numpy() / valid.height.to_numpy(), bins=30, color="#7068ad", edgecolor="white")
    axes[1].set(xlabel="Width / height", ylabel="Images", title="Aspect-ratio distribution")
    fig.savefig(report_dir / "image_dimensions.png", dpi=150)
    plt.close(fig)


def plot_gallery(items, raw_root, output, columns, attributions):
    """输入 [(图片编号, 相对路径, 标题)]，生成图片网格并返回逐图署名记录。"""
    columns = min(columns, len(items))
    rows = math.ceil(len(items) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 2.7), squeeze=False)
    credits = []
    for axis in axes.flat:
        axis.axis("off")
    for axis, (identifier, relative_path, title) in zip(axes.flat, items):
        with Image.open(raw_root / relative_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            axis.imshow(image)
        axis.set_title(f"{identifier} | {title}", fontsize=9)
        credits.append({"figure": output.name, "image_id": identifier, "relative_path": relative_path,
                        **attributions[relative_path], "modification": "缩放为展示缩略图；未改写原始文件"})
    fig.text(0.5, 0.01, "Images: CC BY 2.0 | Authors and sources: image_attributions.csv", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    # 照片网格采用 JPEG，避免把数 MB 的无损照片拼图误当作必要源码体积。
    fig.savefig(output, dpi=120, pil_kwargs={"quality": 90, "optimize": True})
    plt.close(fig)
    return credits


def build_report(frame, edges, moments, targets, config, raw_root, report_dir, attributions):
    """输入真实审计结果；用 Pandas 分组统计、NumPy 数值分析，并输出证据。"""
    active = frame[frame.included]
    counts = pd.DataFrame({"class_name": CLASSES}).set_index("class_name")
    counts["raw"] = frame.groupby("class_name").size()
    counts["included"] = active.groupby("class_name").size()
    for split in SPLITS:
        counts[split] = active[active.split == split].groupby("class_name").size()
    counts["excluded"] = counts.raw - counts.included
    counts = counts.fillna(0).astype(int).reset_index()
    write_csv(report_dir / "class_counts.csv", counts)
    write_csv(report_dir / "manifest.csv", frame)
    write_csv(report_dir / "duplicate_edges.csv", edges)
    write_csv(report_dir / "excluded_images.csv", frame[~frame.included])
    plot_statistics(frame, counts, report_dir)

    # RGB 均值/标准差只使用训练子集，不从验证/测试集估计用于训练的统计量。
    training_moments = np.stack([moments[path] for path in active[active.split == "train"].relative_path]).sum(axis=0)
    mean = training_moments[1:4] / training_moments[0]
    std = np.sqrt(np.maximum(training_moments[4:7] / training_moments[0] - mean ** 2, 0))
    valid = frame[frame.status == "valid"]
    dimensions = valid[["width", "height"]].to_numpy()
    aspect = dimensions[:, 0] / dimensions[:, 1]
    groups = valid.groupby("duplicate_group")
    summary = {
        "version": "v0.2", "generated_at": datetime.now().astimezone().isoformat(),
        "parameters": config, "raw_images": len(frame), "valid_images": len(valid),
        "invalid_images": int((frame.status != "valid").sum()), "included_images": len(active),
        "excluded_reasons": frame[~frame.included].exclusion_reason.value_counts().to_dict(),
        "split_counts": {split: int((active.split == split).sum()) for split in SPLITS},
        "class_counts": counts.to_dict(orient="records"),
        "target_counts_by_split_and_class": targets.tolist(),
        "duplicate_edge_counts": edges.kind.value_counts().to_dict(),
        "multi_image_groups": int((groups.size() > 1).sum()),
        "largest_group": int(groups.size().max()), "cross_label_groups": int((groups.class_name.nunique() > 1).sum()),
        "group_split_leakage": int((active.groupby("duplicate_group").split.nunique() > 1).sum()),
        "size_quantiles": {"quantiles": [0, 0.25, 0.5, 0.75, 1],
                           "width": np.quantile(dimensions[:, 0], [0, .25, .5, .75, 1]).tolist(),
                           "height": np.quantile(dimensions[:, 1], [0, .25, .5, .75, 1]).tolist(),
                           "aspect_ratio": np.quantile(aspect, [0, .25, .5, .75, 1]).tolist()},
        "train_rgb_statistics": {"mean": mean.tolist(), "std": std.tolist(), "scale": "0..1",
                                 "pixel_count": int(training_moments[0]), "method": "训练子集原始 RGB 像素加权；未缩放/裁剪/增强"},
        "clean_class_imbalance_ratio": float(counts.included.max() / counts.included.min()),
        "manifest_sha256": sha256_file(report_dir / "manifest.csv"),
        "limitations": ["近似分组为保守的数据质量规则，不代表所有成员是完全相同图片。",
                        "pHash 对大幅裁剪、旋转或不同视角可能漏检，不能保证所有潜在近重复已发现。",
                        "未人工逐张确认全部类别标签；训练/验证/测试只用于本数据集评估，不能代表真实场景泛化。"]}
    if summary["group_split_leakage"]:
        raise ValueError("检测到重复组跨子集泄漏，停止生成成功报告。")

    rng = np.random.default_rng(config["seed"])
    samples = []
    for class_name in CLASSES:
        candidates = active[(active.class_name == class_name) & (active.split == "train")].relative_path.to_numpy()
        for path in rng.choice(candidates, size=3, replace=False):
            samples.append((f"S{len(samples)+1:02d}", str(path), class_name + " / train"))
    credits = plot_gallery(samples, raw_root, report_dir / "samples.jpg", 3, attributions)
    pairs = edges[edges.kind.isin(["phash_candidate", "reviewed_scene"])].reset_index(drop=True)
    for page in range(math.ceil(len(pairs) / 10)):
        items = []
        for index, pair in pairs.iloc[page*10:(page+1)*10].iterrows():
            for side in ["left", "right"]:
                name = pair[side]
                items.append((f"P{index+1:02d}-{side[0].upper()}", name,
                              name.split("/")[0] + f" / d={int(pair.distance)}"))
        credits.extend(plot_gallery(items, raw_root, report_dir / f"duplicate_pairs_{page+1}.jpg", 4, attributions))
    write_csv(report_dir / "image_attributions.csv", pd.DataFrame(credits))
    write_json(report_dir / "summary.json", summary)
    return summary


def prepare(args):
    """执行本阶段流程；有旧划分时仅允许相同输入与参数重跑，避免静默改变测试集。"""
    raw_root, output_dir, report_dir = args.raw_dir.resolve(), args.output_dir.resolve(), args.report_dir.resolve()
    attributions = read_attributions(raw_root / "LICENSE.txt")
    image_paths = sorted(path for path in raw_root.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        raise ValueError("没有图片，请先运行 download_data.py。")
    paths = {path.relative_to(raw_root).as_posix() for path in image_paths}
    if paths != set(attributions):
        raise ValueError(f"图片与署名不能一一对应：缺少署名 {len(paths-set(attributions))}，缺少图片 {len(set(attributions)-paths)}。")
    reviewed = pd.read_csv(args.review_pairs, keep_default_na=False)
    records, moments = [], {}
    for index, path in enumerate(image_paths, 1):
        row, values = inspect_image(path, raw_root)
        records.append(row)
        moments[row["relative_path"]] = values
        if index % 500 == 0:
            print(f"已检查 {index}/{len(image_paths)} 张图片", flush=True)
    frame, edges = group_duplicates(pd.DataFrame(records), args.phash_distance, reviewed)
    frame, targets = split_groups(frame, args.seed)
    fingerprint = hashlib.sha256("\n".join(frame.relative_path + ":" + frame.sha256).encode()).hexdigest()
    config = {"seed": args.seed, "ratios": RATIOS.tolist(), "phash_bits": 63,
              "phash_distance": args.phash_distance, "reviewed_pairs_sha256": sha256_file(args.review_pairs),
              "dataset_fingerprint": fingerprint, "official_archive_sha256": EXPECTED_SHA256,
              "raw_path_base": "data/raw/flower_photos", "class_to_idx": dict(zip(CLASSES, range(len(CLASSES))))}
    old_config = output_dir / "split_config.json"
    if old_config.exists() and json.loads(old_config.read_text(encoding="utf-8")) != config:
        raise ValueError("已有划分对应不同输入/参数；请使用新的 --output-dir 和 --report-dir，不能覆盖现有测试集。")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "manifest.csv", frame)
    write_csv(output_dir / "duplicate_edges.csv", edges)
    for split in SPLITS:
        write_csv(output_dir / f"{split}.csv", frame[frame.split == split])
    write_json(output_dir / "class_to_idx.json", config["class_to_idx"])
    write_json(output_dir / "split_config.json", config)
    write_json(report_dir / "split_config.json", config)
    summary = build_report(frame, edges, moments, targets, config, raw_root, report_dir, attributions)
    print(f"原始 {len(frame)}；保留 {summary['included_images']}；划分 {summary['split_counts']}")
    print(f"清单：{output_dir}\n报告：{report_dir}\nDATA PREPARATION PASSED")


def main():
    """解析输入目录、可复现种子和输出目录；返回清楚的成功或失败标记。"""
    parser = argparse.ArgumentParser(description="数据审计、重复分组、划分与可视化；不训练模型。")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw/flower_photos")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/processed")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports/local/data")
    parser.add_argument("--review-pairs", type=Path, default=PROJECT_ROOT / "configs/duplicate_review.csv")
    parser.add_argument("--seed", type=int, default=42, help="只控制分组分配和训练样本展示；验收后不要随意修改。")
    parser.add_argument("--phash-distance", type=int, default=8, choices=range(0, 17), metavar="0..16",
                        help="63 位 pHash 的候选阈值，默认 8；近似候选不据此删除。")
    prepare(parser.parse_args())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"DATA PREPARATION FAILED：{error}")
        sys.exit(1)

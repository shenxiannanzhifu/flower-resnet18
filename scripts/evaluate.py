"""对已固定的正式模型执行完整测试评估，保存逐图结果、分类指标和图表。

运行：.venv/Scripts/python.exe scripts/evaluate.py --selection reports/v0.5/model_selection.json
默认写入 reports/local/test-evaluation，拒绝覆盖非空目录；重复运行是复核同一固定模型，不能再用结果调参。
评估前须先用 freeze_model.py 固定模型；不训练、不按测试指标挑选模型。
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "reports/local/matplotlib-cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.checkpoint import load_checkpoint
from src.data import FlowerDataset, make_loader, read_json
from src.evaluation import classification_metrics, training_summary_matches, verified_training_summary
from data_utils import read_attributions


def plot_confusion(matrix: np.ndarray, names: list[str], path: Path) -> None:
    """绘制原始数量混淆矩阵；数字与 CSV 使用同一份计算结果，行真值列预测。"""
    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    artist = axis.imshow(matrix, cmap="Blues", vmin=0)
    fig.colorbar(artist, ax=axis, label="Images")
    axis.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=names, yticklabels=names,
             xlabel="Predicted class", ylabel="True class", title="v0.5 | Held-out test confusion matrix")
    for i in range(len(names)):
        for j in range(len(names)):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center",
                      color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_errors(frame: pd.DataFrame, dataset: FlowerDataset, directory: Path) -> None:
    """选置信度最高的六个错误，展示原图和实际固定裁剪视图，并保存逐图 CC BY 署名。

    选择规则预先固定，不挑更好看的错误；错误图只用于局限描述，不回流训练或改标签。
    """
    errors = frame.loc[~frame.correct].sort_values(["confidence", "relative_path"], ascending=[False, True]).head(6)
    errors.to_csv(directory / "error_examples.csv", index=False)
    attributes = read_attributions(dataset.raw_dir / "LICENSE.txt")
    credit_rows = []
    if errors.empty:
        (directory / "no_errors.txt").write_text("本次测试未出现错误预测。\n", encoding="utf-8")
        return
    rows = (len(errors) + 1) // 2
    fig, axes = plt.subplots(rows, 4, figsize=(13, rows * 3.2), squeeze=False)
    # transform 的前两项恰为 Resize/CenterCrop；不显示归一化后的非自然 RGB 值。
    spatial_transform = transforms.Compose(dataset.transform.transforms[:2])
    for axis in axes.flat:
        axis.axis("off")
    for index, row in enumerate(errors.itertuples(index=False)):
        identifier = f"E{index + 1:02d}"
        with Image.open(dataset.raw_dir / row.relative_path) as image:
            original = ImageOps.exif_transpose(image).convert("RGB")
            crop = spatial_transform(original)
            left = axes[index // 2, (index % 2) * 2]
            right = axes[index // 2, (index % 2) * 2 + 1]
            left.imshow(original)
            right.imshow(crop)
        left.set_title(f"{identifier} original | True: {row.true_class}", fontsize=9)
        right.set_title(f"Input crop | Pred: {row.predicted_class}\np={row.confidence:.3f}", fontsize=9)
        credit_rows.append({"figure_id": identifier, "relative_path": row.relative_path,
                            **attributes[row.relative_path],
                            "modifications": f"EXIF orientation, RGB conversion, display resizing; right panel bilinear resize short edge {dataset.settings['eval_resize']} and center crop {dataset.settings['image_size']}"})
    fig.suptitle(f"{len(errors)} most confident errors | Original versus model input | Test set", fontsize=13)
    fig.text(0.5, 0.012, "Images: CC BY 2.0 | Authors and sources: image_attributions.csv", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(directory / "error_examples.png", dpi=140)
    plt.close(fig)
    pd.DataFrame(credit_rows).to_csv(directory / "image_attributions.csv", index=False)


def main() -> None:
    """核对固定选择与训练记录，再对完整 test 清单仅作前向预测。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("reports/v0.5/model_selection.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/local/test-evaluation"))
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size 必须为正数。")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；可指定 --device cpu。")
    directory = (ROOT / args.output_dir).resolve()
    if not directory.is_relative_to(ROOT) or (directory.exists() and any(directory.iterdir())):
        raise ValueError("评估输出须位于项目内且为空，拒绝覆盖既有结果。")
    selection_path = ROOT / args.selection
    selection = read_json(selection_path)
    if selection["status"] != "FROZEN_BEFORE_TEST":
        raise ValueError("必须提供测试评估前固定的模型选择记录。")
    checkpoint, training_report = ROOT / selection["checkpoint"], ROOT / selection["training_report"]
    summary = verified_training_summary(checkpoint, training_report)
    if (summary["checkpoint_sha256"] != selection["checkpoint_sha256"]
            or summary["best_epoch"] != selection["best_epoch"]
            or not training_summary_matches((training_report / "summary.json").read_bytes(), selection)):
        raise ValueError("模型或训练报告在固定选择后发生变化。")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    model, metadata = load_checkpoint(checkpoint, device)
    dataset = FlowerDataset(ROOT, "test", Path(summary["config"]["data_config"]))
    if (metadata["purpose"] != "formal" or metadata["epoch"] != summary["best_epoch"]
            or metadata["manifest_sha256"] != dataset.manifest_sha256
            or metadata["class_to_idx"] != dataset.class_to_idx or metadata["preprocessing"] != dataset.settings
            or metadata["normalization"] != dataset.summary["train_rgb_statistics"]
            or len(dataset) != dataset.summary["split_counts"]["test"]
            or summary["train_samples"] != dataset.summary["split_counts"]["train"]
            or summary["val_samples"] != dataset.summary["split_counts"]["val"]):
        raise ValueError("模型的映射/预处理/数据版本或完整划分数量不一致。")
    names = [name for name, index in sorted(dataset.class_to_idx.items(), key=lambda item: item[1])]
    records, total_loss = [], 0.0
    before = {name: value.clone() for name, value in model.named_buffers()}
    with torch.inference_mode():
        for images, labels, paths in make_loader(dataset, batch_size=args.batch_size):
            logits = model(images.to(device))
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("测试输出存在 NaN/Inf。")
            total_loss += float(torch.nn.functional.cross_entropy(logits, labels.to(device), reduction="sum"))
            probabilities = logits.softmax(dim=1).cpu().numpy()
            for label, path, probs in zip(labels.tolist(), paths, probabilities):
                predicted = int(probs.argmax())
                records.append({"relative_path": path, "label": label, "prediction": predicted,
                                "true_class": names[label], "predicted_class": names[predicted],
                                "confidence": float(probs[predicted]), "correct": label == predicted,
                                **{f"prob_{name}": float(probs[i]) for i, name in enumerate(names)}})
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    frame = pd.DataFrame(records)
    if len(frame) != len(dataset) or not frame.relative_path.is_unique:
        raise ValueError("测试预测未完整且唯一覆盖清单。")
    metrics = classification_metrics(frame.label.to_numpy(), frame.prediction.to_numpy(), names)
    report = {"status": "PASS", "purpose": "formal", "split": "test",
              "evaluated_at": datetime.now().astimezone().isoformat(), "device": str(device),
              "checkpoint_sha256": selection["checkpoint_sha256"], "selected_epoch": selection["best_epoch"],
              "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
              "manifest_sha256": dataset.manifest_sha256, "loss": total_loss / len(dataset),
              "class_to_idx": dataset.class_to_idx, "batch_size": args.batch_size, "num_workers": 0,
              "batchnorm_unchanged": True, **metrics,
              "limitations": "单次种子、单数据集测试结果；选模和训练配置固定后评估，不代表外部新图泛化",
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("scripts/evaluate.py", "src/evaluation.py", "src/model.py", "src/data.py")}}
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "predictions.csv", index=False)
    pd.DataFrame(metrics["per_class"]).to_csv(directory / "classification_report.csv", index=False)
    matrix = np.asarray(metrics["confusion_matrix"])
    pd.DataFrame(matrix, index=names, columns=names).to_csv(directory / "confusion_matrix.csv", index_label="true_class")
    plot_confusion(matrix, names, directory / "confusion_matrix.png")
    plot_errors(frame, dataset, directory)
    (directory / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Test: {metrics['correct']}/{metrics['samples']}, accuracy={metrics['accuracy']:.6f}, macro-F1={metrics['macro_f1']:.6f}")
    print("FORMAL EVALUATION PASSED — 固定模型的完整测试评估已保存。")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError, AssertionError) as error:
        print(f"EVALUATION FAILED: {error}", file=sys.stderr)
        sys.exit(1)

"""仅在验证集比较中心裁剪与整图输入，并诊断未知类别分数；不读取测试集。

运行：.venv/Scripts/python.exe scripts/compare_input_views.py --device cuda
固定方案见 reports/v0.6.1/protocol.md；默认输出到新的 reports/local/input-view-comparison。
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports/local/matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data import FlowerDataset
from src.inference import load_predictor, predict_image, prepare_input, read_image


def synthetic_probes() -> dict:
    """返回预先固定的八张非花卉图；只检查反例，不用它们估计未知类别准确率。"""
    colors = {"black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128),
              "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
    images = {name: Image.new("RGB", (384, 256), color) for name, color in colors.items()}
    noise = np.random.default_rng(42).integers(0, 256, size=(256, 384, 3), dtype=np.uint8)
    images["noise_seed42"] = Image.fromarray(noise)
    grid = ((np.indices((256, 384)) // 16).sum(axis=0) % 2 * 255).astype(np.uint8)
    images["checkerboard"] = Image.fromarray(np.repeat(grid[:, :, None], 3, axis=2))
    return images


def main() -> None:
    """固定模型逐张比较两个输入方式，保留完整 val 结果，不做参数搜索。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/local/input-view-comparison"))
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("输出目录必须为空，不能覆盖既有比较。")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用；可使用 --device cpu。")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = ROOT / "artifacts/v0.5-formal/best.pt"
    model, metadata = load_predictor(checkpoint, torch.device(args.device))
    dataset = FlowerDataset(ROOT, "val")
    if (dataset.manifest_sha256 != metadata["manifest_sha256"]
            or dataset.class_to_idx != metadata["class_to_idx"]
            or dataset.settings != metadata["preprocessing"]
            or dataset.summary["train_rgb_statistics"] != metadata["normalization"]):
        raise ValueError("验证数据与固定模型不配套。")
    rows = []
    for index, row in enumerate(dataset.frame.itertuples(index=False), 1):
        rgb = read_image(dataset.raw_dir / row.relative_path)
        label = int(row.label)
        for view in ("center", "full"):
            result, _ = predict_image(model, metadata, rgb, view=view)
            # 从同一次前向的 logits 计算损失；float64 仅用于离线统计，不改变模型精度。
            loss = -float(torch.tensor(result["logits"], dtype=torch.float64).log_softmax(0)[label])
            predicted = metadata["class_to_idx"][result["predicted_class"]]
            rows.append({"relative_path": row.relative_path, "label": label, "view": view,
                         "prediction": predicted, "correct": predicted == label,
                         "score": result["score"], "loss": loss,
                         "input_width": result["model_input_size"][0], "input_height": result["model_input_size"][1]})
        if index % 100 == 0 or index == len(dataset):
            print(f"Validation comparison: {index}/{len(dataset)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "validation_predictions.csv", index=False)
    metrics = {}
    for view in ("center", "full"):
        values = frame.loc[frame.view == view]
        if len(values) != len(dataset) or not values.relative_path.is_unique:
            raise ValueError("验证集未完整覆盖。")
        threshold = float(np.quantile(values.score, 0.05))
        metrics[view] = {"samples": len(values), "correct": int(values.correct.sum()),
                         "accuracy": float(values.correct.mean()), "loss": float(values.loss.mean()),
                         "diagnostic_score_threshold": threshold,
                         "known_coverage_at_threshold": float((values.score >= threshold).mean())}
    accuracy_change = metrics["full"]["accuracy"] - metrics["center"]["accuracy"]
    loss_change = metrics["full"]["loss"] - metrics["center"]["loss"]
    adopt = accuracy_change >= -0.01 and loss_change <= 0.05

    probes, probe_rows = synthetic_probes(), []
    for name, rgb in probes.items():
        for view in ("center", "full"):
            result, _ = predict_image(model, metadata, rgb, view=view)
            probe_rows.append({"name": name, "view": view, "prediction": result["predicted_class"],
                               "score": result["score"],
                               "passes_diagnostic_threshold": result["score"] >= metrics[view]["diagnostic_score_threshold"]})
    pd.DataFrame(probe_rows).to_csv(output / "unknown_probes.csv", index=False)
    fig, axes = plt.subplots(2, 4, figsize=(12, 7))
    for axis, (name, rgb) in zip(axes.flat, probes.items()):
        values = [row for row in probe_rows if row["name"] == name]
        axis.imshow(rgb)
        axis.set_title(name + "\n" + "\n".join(f"{row['view']}: {row['prediction']} {row['score']:.1%}" for row in values), fontsize=9)
        axis.axis("off")
    fig.suptitle("Fixed synthetic non-flower probes | Diagnostic counterexamples, not an OOD benchmark")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output / "unknown_probes.png", dpi=140)
    plt.close(fig)

    pixels = np.zeros((300, 900, 3), dtype=np.uint8)
    pixels[:, :30] = (255, 0, 0)
    pixels[:, -30:] = (0, 255, 0)
    pixels[:30, :] = (0, 0, 255)
    pixels[-30:, :] = (255, 255, 0)
    rgb = Image.fromarray(pixels)
    images = [rgb] + [prepare_input(rgb, metadata, view)[1] for view in ("center", "full")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, image, title in zip(axes, images, ("Original: four colored edges", "Center crop: edges lost", "Full image: edges retained")):
        axis.imshow(image)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    fig.suptitle("Synthetic geometry check (not a flower accuracy test)")
    fig.tight_layout()
    fig.savefig(output / "geometry.png", dpi=140)
    plt.close(fig)
    report = {"status": "PASS", "purpose": "validation_development_comparison",
              "checked_at": datetime.now().astimezone().isoformat(), "device": args.device,
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "manifest_sha256": dataset.manifest_sha256, "test_images_read": 0,
              "metrics": metrics, "accuracy_change": accuracy_change, "loss_change": loss_change,
              "adoption_rule": "accuracy_change >= -0.01 and loss_change <= 0.05; fixed before comparison",
              "adopt_full_default": adopt,
              "synthetic_unknown_passed": {view: sum(row["passes_diagnostic_threshold"] for row in probe_rows if row["view"] == view)
                                          for view in ("center", "full")},
              "unknown_detection_solved": False, "threshold_deployed": False,
              "limitation": "验证集已用于选模；比较不是独立测试。八张合成图只能给出反例，不证明通用未知拒识。",
              "git_commit_at_run": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("scripts/compare_input_views.py", "src/inference.py", "src/model.py", "src/data.py")}}
    (output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("INPUT VIEW COMPARISON PASSED", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"比较失败：{error}", file=sys.stderr)
        sys.exit(1)

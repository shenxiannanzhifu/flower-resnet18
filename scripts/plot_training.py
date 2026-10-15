"""从已完成的正式实验 CSV 绘制损失/准确率曲线，不生成示例数值。

运行：.venv/Scripts/python.exe scripts/plot_training.py --output reports/local/training_curves.png
输入默认 reports/v0.5/formal；已有图片拒绝覆盖，换路径可重新生成。
"""

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "reports/local/matplotlib-cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    """并排绘制真实 epoch 指标；训练指标来自更新过程中，验证指标来自该轮结束模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, default=Path("reports/v0.5/formal"))
    parser.add_argument("--output", type=Path, default=Path("reports/local/training_curves.png"))
    args = parser.parse_args()
    directory, output = ROOT / args.training_report, ROOT / args.output
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有图表：{output}")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    history = pd.read_csv(directory / "metrics.csv")
    if summary["status"] != "PASS" or summary["purpose"] != "formal" or len(history) != summary["config"]["epochs"]:
        raise ValueError("仅为已完成的正式实验绘图。")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, metric, ylabel in zip(axes, ("loss", "accuracy"), ("Cross entropy", "Accuracy")):
        axis.plot(history.epoch, history[f"train_{metric}"], label="Train (during updates)", color="#2878b5")
        axis.plot(history.epoch, history[f"val_{metric}"], label="Validation", color="#e07b39")
        axis.axvline(summary["best_epoch"], color="#555555", linestyle="--", label=f"Selected epoch {summary['best_epoch']}")
        axis.set(xlabel="Epoch", ylabel=ylabel)
        if metric == "accuracy":
            axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle(f"v0.5 | Full train/validation | Random initialization | Seed {summary['config']['seed']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"TRAINING CURVES SAVED: {output}")


if __name__ == "__main__":
    main()

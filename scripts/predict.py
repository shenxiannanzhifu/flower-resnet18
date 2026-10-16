"""用正式模型预测单张图片，输出中文结果、JSON 和原图/实际裁剪/分数展示图。

运行：.venv/Scripts/python.exe scripts/predict.py --image examples/sunflower.jpg
默认使用本机 v0.5 正式模型、CPU、Top-3；输出到 reports/local/predictions 下的新目录。
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports/local/matplotlib-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.inference import load_predictor, predict_image, read_image


def check_output_directory(directory: Path) -> None:
    """拒绝非空目录或已有文件，避免一次新预测覆盖旧结果或输入图片。"""
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError(f"输出目录已存在且非空：{directory}；换一个目录，或省略 --output-dir 自动生成新目录。")


def image_credit(image_sha256: str) -> dict | None:
    """只有哈希匹配随仓库提供的公开示例时，自动附上该图片作者和许可；不猜测用户图片来源。"""
    path = ROOT / "examples/attribution.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["example_sha256"] == image_sha256:
            return {key: record[key] for key in ("author", "source_page", "license", "license_url")}
    return None


def save_figure(rgb, crop, report: dict, output: Path) -> None:
    """并列显示原图、真实裁剪和 Top-K 分数；英文图中文字避免不同机器缺中文字体。"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    axes[0].imshow(rgb)
    axes[0].set_title("Original image")
    axes[1].imshow(crop)
    axes[1].set_title("Model input (center crop)")
    for axis in axes[:2]:
        axis.axis("off")
    items = report["top_k"]
    scores = [item["score"] for item in items]
    axes[2].barh(range(len(items)), scores, color=["#247a59"] + ["#97b7a8"] * (len(items) - 1))
    axes[2].set(yticks=range(len(items)), yticklabels=[item["class_name"] for item in items],
                xlim=(0, 1.22), xlabel="Softmax score", title=f"Top-{len(items)} predictions",
                xticks=[0, 0.25, 0.5, 0.75, 1])
    axes[2].invert_yaxis()
    for index, score in enumerate(scores):
        axes[2].text(score + 0.02, index, f"{score:.1%}", va="center", fontsize=10)
    fig.suptitle(f"Prediction: {report['predicted_class']} | Score: {report['score']:.1%}", fontsize=16)
    footer = "Five trained classes only; scores are not calibrated accuracy."
    if report["image_credit"]:
        credit = report["image_credit"]
        footer += f"\nImage: {credit['author']} | {credit['license']} | Source and modifications: prediction.json"
    fig.text(0.5, 0.035, footer, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 0.92), w_pad=2.5)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    """校验参数，实际加载与预测，成功后保存报告；不依赖原数据目录或互联网。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="图片路径；相对路径按当前命令所在目录解析")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/v0.5-formal/best.pt",
                        help="正式模型路径；默认使用项目内 v0.5 正式模型")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="CPU 最易运行；已有 CUDA 环境可选 cuda")
    parser.add_argument("--top-k", type=int, default=3, help="显示前几个类别，五类模型可设 1–5")
    parser.add_argument("--output-dir", type=Path, default=None, help="可选空目录；省略则每次自动生成新目录")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用；请改为 --device cpu。")
    image_path, checkpoint = args.image.resolve(), args.checkpoint.resolve()
    directory = (args.output_dir or ROOT / "reports/local/predictions" /
                 datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
    check_output_directory(directory)
    rgb = read_image(image_path)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, metadata = load_predictor(checkpoint, torch.device(args.device))
    result, crop = predict_image(model, metadata, rgb, args.top_k)
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    report = {"status": "PASS", "purpose": "single_image_prediction",
              "predicted_at": datetime.now().astimezone().isoformat(), "device": args.device,
              "image_name": image_path.name, "image_sha256": image_hash,
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "model_purpose": metadata["purpose"], "selected_epoch": metadata["epoch"], **result,
              "image_credit": image_credit(image_hash),
              "display_modifications": "EXIF orientation, RGB conversion; center panel uses saved resize/center crop; figure resizes display",
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("scripts/predict.py", "src/inference.py", "src/model.py", "src/data.py", "src/checkpoint.py")}}
    directory.mkdir(parents=True, exist_ok=True)
    save_figure(rgb, crop, report, directory / "prediction.png")
    # x 模式防止并发情况下静默覆盖同名结果；已有验收目录在任何推理前就被拒绝。
    with (directory / "prediction.json").open("x", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"预测：{result['display_name']}（{result['predicted_class']}），模型分数 {result['score']:.2%}")
    for rank, item in enumerate(result["top_k"], 1):
        print(f"  {rank}. {item['display_name']} ({item['class_name']}): {item['score']:.2%}")
    print("提示：分数不是实际正确率；模型只能在五类花卉中选择。")
    print(f"结果文件：{directory / 'prediction.json'}")
    print(f"展示图片：{directory / 'prediction.png'}")
    print("PREDICTION PASSED")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(f"预测失败：{error}", file=sys.stderr)
        sys.exit(1)

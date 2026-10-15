"""在独立进程加载试运行模型，预测保存时的验证样本并核对 logits。

运行：.venv/Scripts/python.exe scripts/check_checkpoint.py --device cpu
默认读取 artifacts/v0.4-smoke/best.pt。这是 v0.4 内部流程检查；新图命令与展示在 v0.6 完善。
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoint import check_reference_prediction, load_checkpoint


def main() -> None:
    """读取本项目 checkpoint，严格恢复模型并验证独立进程的预测结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/v0.4-smoke/best.pt"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--report-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；可使用 --device cpu 加载。")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    path = (ROOT / args.checkpoint).resolve()
    report_dir = (ROOT / (args.report_dir or Path(f"reports/local/checkpoint-{args.device}"))).resolve()
    report_path = report_dir / "checkpoint-check.json"
    if report_path.exists() and not report_dir.is_relative_to((ROOT / "reports/local").resolve()):
        raise ValueError("拒绝覆盖既有验收记录；日常复核请使用默认目录。")
    model, metadata = load_checkpoint(path, torch.device(args.device))
    prediction = check_reference_prediction(model, metadata, ROOT)
    report = {"status": "PASS", "checked_at": datetime.now().astimezone().isoformat(),
              "device": args.device, "best_epoch": metadata["epoch"],
              "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "class_to_idx": metadata["class_to_idx"], **prediction}
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"SMOKE reference: {prediction['relative_path']}; prediction: {prediction['predicted_class']}")
    print(f"Reload max difference: {prediction['maximum_absolute_difference']:.8g}")
    print("CHECKPOINT CHECK PASSED — 仅验证试运行模型的加载与预测。")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError, AssertionError) as error:
        print(f"CHECKPOINT CHECK FAILED: {error}", file=sys.stderr)
        sys.exit(1)

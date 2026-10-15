"""在测试评估前固定模型文件及选择依据；不读取测试图片或生成测试指标。

运行：.venv/Scripts/python.exe scripts/freeze_model.py --output reports/local/model-selection.json
默认针对 v0.5 正式模型。记录已存在时拒绝覆盖；评估入口核对该记录的文件哈希。
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation import verified_training_summary


def main():
    """读取训练报告并核对选模规则，输出不可静默覆盖的选择记录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/v0.5-formal/best.pt"))
    parser.add_argument("--training-report", type=Path, default=Path("reports/v0.5/formal"))
    parser.add_argument("--output", type=Path, default=Path("reports/local/model-selection.json"))
    args = parser.parse_args()
    checkpoint, directory, output = ROOT / args.checkpoint, ROOT / args.training_report, ROOT / args.output
    summary = verified_training_summary(checkpoint, directory)
    record = {"status": "FROZEN_BEFORE_TEST", "selected_at": datetime.now().astimezone().isoformat(),
              "checkpoint": str(checkpoint.resolve().relative_to(ROOT)),
              "checkpoint_sha256": summary["checkpoint_sha256"],
              "training_report": str(directory.resolve().relative_to(ROOT)),
              "training_summary_sha256": hashlib.sha256((directory / "summary.json").read_bytes()).hexdigest(),
              "best_epoch": summary["best_epoch"], "validation_loss": summary["best_validation_loss"],
              "validation_accuracy": summary["best_validation_accuracy"],
              "selection_rule": "minimum validation cross entropy; test metrics unavailable at selection",
              "training_config": summary["config"],
              "decision": "按预先固定的方案选择最小验证损失模型；不因测试结果改模型或训练配置"}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"MODEL SELECTION FROZEN: epoch={record['best_epoch']}, sha256={record['checkpoint_sha256']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"MODEL SELECTION FAILED: {error}", file=sys.stderr)
        sys.exit(1)

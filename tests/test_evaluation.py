"""使用可手算的例子检查指标与选模规则；运行 python -m unittest discover -s tests -v。"""

import hashlib
import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
import pandas as pd

from src.evaluation import classification_metrics, verified_training_summary


ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    """检查矩阵轴、宏平均、零分母，以及模型/最优轮次的独立核验。"""

    def test_known_confusion_and_macro_scores(self):
        result = classification_metrics([0, 0, 1, 1, 2], [0, 1, 1, 2, 1], ["a", "b", "c"])
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [0, 1, 1], [0, 1, 0]])
        self.assertAlmostEqual(result["accuracy"], 2 / 5)
        self.assertAlmostEqual(result["per_class"][0]["precision"], 1)
        self.assertAlmostEqual(result["per_class"][1]["precision"], 1 / 3)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 0.4) / 3)
        self.assertAlmostEqual(result["weighted_f1"], (2 * 2 / 3 + 2 * 0.4) / 5)

    def test_zero_denominator_and_invalid_labels(self):
        result = classification_metrics([0, 0], [0, 0], ["a", "b"])
        self.assertEqual(result["per_class"][1]["f1"], 0)
        self.assertEqual(result["macro_f1"], 0.5)
        for labels, predicted in [([], []), ([0.5], [0]), ([2], [0]), ([0, 1], [0])]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                classification_metrics(labels, predicted, ["a", "b"])

    def test_selection_rejects_changed_checkpoint_and_wrong_best_epoch(self):
        base = (ROOT / "reports/local/tests").resolve()
        directory = base / uuid.uuid4().hex
        directory.mkdir(parents=True)
        try:
            checkpoint = directory / "dummy.pt"
            checkpoint.write_bytes(b"synthetic-checkpoint-for-hash-test")
            report = {"status": "PASS", "purpose": "formal", "test_images_used": 0,
                      "selection_metric": "validation_loss_min", "config": {"epochs": 2},
                      "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      "best_epoch": 1, "best_validation_loss": 0.5, "train_samples": 4, "val_samples": 2}
            path = directory / "summary.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            pd.DataFrame({"purpose": ["formal"] * 2, "epoch": [1, 2], "val_loss": [0.5, 0.8],
                          "train_samples": [4, 4], "val_samples": [2, 2]}).to_csv(directory / "metrics.csv", index=False)
            self.assertEqual(verified_training_summary(checkpoint, directory)["best_epoch"], 1)
            report["best_epoch"] = 2
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "最小"):
                verified_training_summary(checkpoint, directory)
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "哈希"):
                verified_training_summary(checkpoint, directory)
        finally:
            resolved = directory.resolve()
            if not resolved.is_relative_to(base) or resolved == base:
                raise RuntimeError("拒绝清理测试目录外路径。")
            shutil.rmtree(resolved)


if __name__ == "__main__":
    unittest.main()

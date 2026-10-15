"""训练与保存加载的行为检查；运行 python -m unittest discover -s tests -v。"""

from pathlib import Path
import shutil
import unittest
import uuid

import torch
from torch import nn

from src.checkpoint import load_checkpoint, save_checkpoint
from src.data import read_json
from src.engine import run_epoch
from src.model import ResNet18


ROOT = Path(__file__).resolve().parents[1]


class TrainingTests(unittest.TestCase):
    """检查参数确实更新、验证不改变状态，以及不等长批次的统计正确性。"""

    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_loss_is_weighted_by_samples_and_eval_is_read_only(self):
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(2))
        images = torch.tensor([[5.0, 0.0], [0.0, 3.0], [4.0, 0.0]])
        labels = torch.tensor([0, 1, 1])
        batches = [(images[:2], labels[:2], ["a", "b"]), (images[2:], labels[2:], ["c"])]
        before = model.weight.detach().clone()
        metrics = run_epoch(model, batches, nn.CrossEntropyLoss(), torch.device("cpu"))
        expected_loss = float(nn.functional.cross_entropy(images, labels))
        self.assertAlmostEqual(metrics["loss"], expected_loss, places=6)
        self.assertEqual(metrics["samples"], 3)
        self.assertAlmostEqual(metrics["accuracy"], 2 / 3)
        torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
        self.assertIsNone(model.weight.grad)

    def test_training_updates_parameters_and_bn_but_validation_does_not(self):
        torch.manual_seed(42)
        model = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3), nn.ReLU(), nn.Linear(3, 2))
        batches = [(torch.randn(4, 2), torch.tensor([0, 1, 1, 0]), ["a"] * 4)]
        original = model[0].weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        run_epoch(model, batches, nn.CrossEntropyLoss(), torch.device("cpu"), optimizer)
        self.assertGreater(float((model[0].weight.detach() - original).abs().max()), 0)
        self.assertEqual(int(model[1].num_batches_tracked), 1)
        state = {name: value.clone() for name, value in model.state_dict().items()}
        run_epoch(model, batches, nn.CrossEntropyLoss(), torch.device("cpu"))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)

    def test_empty_or_nonfinite_input_cannot_report_success(self):
        model = nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "没有样本"):
            run_epoch(model, [], nn.CrossEntropyLoss(), torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            run_epoch(model, [(torch.full((2, 2), float("nan")), torch.tensor([0, 1]), [])],
                      nn.CrossEntropyLoss(), torch.device("cpu"))

    def test_checkpoint_roundtrip_and_invalid_mapping(self):
        """实际序列化完整手写模型；恢复的 logits 必须一致，错误映射不能覆盖文件。"""
        base = (ROOT / "reports/local/tests").resolve()
        directory = base / uuid.uuid4().hex
        directory.mkdir(parents=True)
        path = directory / "model.pt"
        try:
            torch.manual_seed(42)
            model = ResNet18().eval()
            settings = read_json(ROOT / "configs/model_data.json")
            summary = read_json(ROOT / "reports/v0.2.1/summary.json")
            metadata = {"purpose": "smoke", "epoch": 1, "validation_loss": 1.5,
                        "class_to_idx": summary["parameters"]["class_to_idx"],
                        "preprocessing": settings, "normalization": summary["train_rgb_statistics"]}
            image = torch.randn(1, 3, 64, 64)
            with torch.inference_mode():
                expected = model(image)
            save_checkpoint(path, model, metadata)
            restored, loaded_metadata = load_checkpoint(path, torch.device("cpu"))
            with torch.inference_mode():
                torch.testing.assert_close(restored(image), expected, rtol=0, atol=0)
            self.assertEqual(loaded_metadata, metadata)
            self.assertFalse(restored.training)
            before_size = path.stat().st_size
            metadata["class_to_idx"] = {"a": 1}
            with self.assertRaisesRegex(ValueError, "类别索引"):
                save_checkpoint(path, model, metadata)
            self.assertEqual(path.stat().st_size, before_size)
        finally:
            resolved = directory.resolve()
            if not resolved.is_relative_to(base) or resolved == base:
                raise RuntimeError("拒绝清理测试目录之外的路径。")
            shutil.rmtree(resolved)


if __name__ == "__main__":
    unittest.main()

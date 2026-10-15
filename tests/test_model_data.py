"""验证 v0.3 数据接口与防混用边界；运行 python -m unittest discover -s tests -v。

使用本地生成的人工图片测试明确行为，不改写正式数据或统计。
"""

import hashlib
import json
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.data import FlowerDataset, make_loader, seed_everything


ROOT = Path(__file__).resolve().parents[1]


class ModelDataTests(unittest.TestCase):
    """防止错误归一化、类别错位、路径越界和不稳定的验证预处理。"""

    def setUp(self):
        self.base = (ROOT / "reports/local/tests").resolve()
        self.directory = self.base / uuid.uuid4().hex
        for name in ("raw/daisy", "processed", "configs"):
            (self.directory / name).mkdir(parents=True, exist_ok=True)
        self.settings = json.loads((ROOT / "configs/model_data.json").read_text(encoding="utf-8"))
        self.settings.update(raw_dir="raw", processed_dir="processed", statistics_file="summary.json")
        self.write_json("configs/model_data.json", self.settings)
        self.mapping = {"daisy": 0}
        self.write_json("processed/class_to_idx.json", self.mapping)
        self.write_json("processed/split_config.json", {"class_to_idx": self.mapping})
        self.rows = []
        for split in ("train", "val"):
            for i in range(3):
                relative = f"daisy/{split}{i}.png"
                # 可计算的纯色像素，避免只验证 shape 而放过通道顺序或归一化错误。
                Image.new("RGB", (260, 280), color=(51, 102, 153)).save(self.directory / "raw" / relative)
                self.rows.append({"relative_path": relative, "class_name": "daisy", "label": "0",
                                  "split": split, "included": "True", "status": "valid"})
        self.save_manifests()

    def write_json(self, name, value):
        (self.directory / name).write_text(json.dumps(value), encoding="utf-8")

    def save_manifests(self):
        """写入配套的合成清单与统计，用来单独触发每一种错误。"""
        frame = pd.DataFrame(self.rows)
        path = self.directory / "processed/manifest.csv"
        frame.to_csv(path, index=False)
        for split in ("train", "val"):
            frame[frame.split == split].to_csv(self.directory / f"processed/{split}.csv", index=False)
        self.summary = {"parameters": {"class_to_idx": self.mapping},
                        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "train_rgb_statistics": {"mean": [0.1, 0.2, 0.3], "std": [0.5, 0.5, 0.5]}}
        self.write_json("summary.json", self.summary)

    def tearDown(self):
        resolved = self.directory.resolve()
        if not resolved.is_relative_to(self.base) or resolved == self.base:
            raise RuntimeError("拒绝清理测试工作区之外的路径。")
        shutil.rmtree(resolved)

    def test_eval_is_deterministic_and_uses_training_rgb(self):
        dataset = FlowerDataset(self.directory, "val")
        first, label, path = dataset[0]
        second = dataset[0][0]
        self.assertEqual(first.shape, (3, 224, 224))
        self.assertEqual(first.dtype, torch.float32)
        self.assertEqual((label, path), (0, "daisy/val0.png"))
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first[:, 100, 100], torch.tensor([0.2, 0.4, 0.6]))

    def test_loader_keeps_final_batch_and_integer_labels(self):
        batches = list(make_loader(FlowerDataset(self.directory, "val"), batch_size=2))
        self.assertEqual([len(batch[0]) for batch in batches], [2, 1])
        self.assertTrue(all(batch[1].dtype == torch.int64 for batch in batches))
        self.assertEqual(batches[-1][2][0], "daisy/val2.png")

    def test_train_seed_reproduces_crop_and_shuffle(self):
        # 使用空间变化的图案，让随机裁剪或镜像的差异可被实际观察。
        pixels = np.arange(280 * 260 * 3, dtype=np.uint32).reshape(280, 260, 3).astype(np.uint8)
        for row in self.rows:
            Image.fromarray(pixels).save(self.directory / "raw" / row["relative_path"])
        dataset = FlowerDataset(self.directory, "train")
        seed_everything(42)
        left = next(iter(make_loader(dataset, batch_size=3, seed=42)))
        seed_everything(42)
        right = next(iter(make_loader(dataset, batch_size=3, seed=42)))
        torch.testing.assert_close(left[0], right[0], rtol=0, atol=0)
        self.assertEqual(left[2], right[2])

    def test_rejects_old_statistics(self):
        self.summary["manifest_sha256"] = "0" * 64
        self.write_json("summary.json", self.summary)
        with self.assertRaisesRegex(ValueError, "统计来源不一致"):
            FlowerDataset(self.directory, "train")

    def test_rejects_mismatched_mapping_and_split(self):
        self.write_json("processed/class_to_idx.json", {"roses": 0})
        with self.assertRaisesRegex(ValueError, "类别映射"):
            FlowerDataset(self.directory, "train")
        self.write_json("processed/class_to_idx.json", self.mapping)
        path = self.directory / "processed/val.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("val0.png", "val99.png"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "总清单不一致"):
            FlowerDataset(self.directory, "val")

    def test_rejects_excluded_sample_and_unsafe_path(self):
        self.rows[0]["included"] = "False"
        self.save_manifests()
        with self.assertRaisesRegex(ValueError, "有效状态"):
            FlowerDataset(self.directory, "train")
        self.rows[0]["included"] = "True"
        self.rows[0]["relative_path"] = "../outside.png"
        self.save_manifests()
        with self.assertRaisesRegex(ValueError, "不安全"):
            FlowerDataset(self.directory, "train")

    def test_missing_and_corrupt_images_explain_path(self):
        dataset = FlowerDataset(self.directory, "val")
        image_path = self.directory / "raw/daisy/val0.png"
        image_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "图片不存在"):
            dataset[0]
        image_path.write_bytes(b"broken image")
        with self.assertRaisesRegex(ValueError, "图片无法解码"):
            dataset[0]


if __name__ == "__main__":
    unittest.main()

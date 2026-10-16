"""新图推理的输入、预处理和类别含义检查；运行 python -m unittest discover -s tests -v。"""

from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
from PIL import Image
import torch
from torch import nn

from src.data import read_json
from src.inference import load_predictor, predict_image, read_image


ROOT = Path(__file__).resolve().parents[1]


class FixedLogits(nn.Module):
    """提供已知类别分数，并记录推理是否关闭训练状态和梯度；不用于实验成绩。"""

    def __init__(self, logits):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, images):
        self.seen_input = images.cpu().clone()
        self.seen_training = self.training
        self.seen_grad_enabled = torch.is_grad_enabled()
        return self.logits


class InferenceTests(unittest.TestCase):
    """验证新图不会自行改变归一化/类别顺序，并覆盖常见文件错误。"""

    def setUp(self):
        self.base = (ROOT / "reports/local/tests").resolve()
        self.directory = self.base / uuid.uuid4().hex
        self.directory.mkdir(parents=True)
        self.metadata = {
            "purpose": "formal", "epoch": 21, "validation_loss": 0.5,
            # 故意打乱字典顺序，编号与训练时保持一致。
            "class_to_idx": {"tulips": 4, "sunflowers": 3, "roses": 2, "dandelion": 1, "daisy": 0},
            "preprocessing": read_json(ROOT / "configs/model_data.json"),
            "normalization": {"mean": [0.1, 0.2, 0.3], "std": [0.2, 0.4, 0.5]}}

    def tearDown(self):
        resolved = self.directory.resolve()
        if not resolved.is_relative_to(self.base) or resolved == self.base:
            raise RuntimeError("拒绝清理测试目录外路径。")
        shutil.rmtree(resolved)

    def test_saved_statistics_index_order_and_inference_state(self):
        model = FixedLogits([[0, 2, -2, 1, 0.5]])
        result, crop = predict_image(model, self.metadata, Image.new("RGB", (320, 280), (128, 128, 128)), 3)
        expected_pixel = (torch.tensor([128 / 255] * 3) - torch.tensor([0.1, 0.2, 0.3])) / torch.tensor([0.2, 0.4, 0.5])
        torch.testing.assert_close(model.seen_input[0, :, 100, 100], expected_pixel)
        self.assertEqual(result["input_shape"], [1, 3, 224, 224])
        self.assertEqual(crop.size, (224, 224))
        self.assertEqual(result["predicted_class"], "dandelion")
        self.assertEqual([item["class_index"] for item in result["top_k"]], [1, 3, 4])
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1, places=6)
        self.assertFalse(model.seen_training)
        self.assertFalse(model.seen_grad_enabled)
        self.assertIsNone(model.anchor.grad)

    def test_invalid_top_k_and_invalid_logits_are_rejected(self):
        rgb = Image.new("RGB", (256, 256))
        for top_k in (0, 6, True):
            with self.subTest(top_k=top_k), self.assertRaisesRegex(ValueError, "top-k"):
                predict_image(FixedLogits([[0] * 5]), self.metadata, rgb, top_k)
        for logits in ([[0, 1]], [[0, 1, 2, float("nan"), 4]]):
            with self.subTest(logits=logits), self.assertRaisesRegex(ValueError, "输出形状"):
                predict_image(FixedLogits(logits), self.metadata, rgb)

    def test_grayscale_rgba_and_unicode_paths(self):
        for mode, color in (("L", 128), ("RGBA", (20, 40, 60, 0))):
            path = self.directory / f"中文 图片 {mode}.png"
            Image.new(mode, (80, 60), color).save(path)
            rgb = read_image(path)
            self.assertEqual(rgb.mode, "RGB")
            self.assertEqual(rgb.size, (80, 60))
            self.assertEqual(rgb.getpixel((0, 0)), (128, 128, 128) if mode == "L" else (20, 40, 60))

    def test_exif_orientation_is_applied_before_crop(self):
        values = np.zeros((40, 20, 3), dtype=np.uint8)
        values[:20, :, 0] = 255
        path = self.directory / "rotated.jpg"
        exif = Image.Exif()
        exif[274] = 6  # 相机标记：显示时顺时针转 90 度。
        Image.fromarray(values).save(path, exif=exif)
        with Image.open(path) as stored:
            expected = np.rot90(np.asarray(stored.convert("RGB")), k=-1)
        actual = read_image(path)
        self.assertEqual(actual.size, (40, 20))
        np.testing.assert_array_equal(np.asarray(actual), expected)

    def test_missing_corrupt_and_animated_images_have_clear_errors(self):
        with self.assertRaisesRegex(FileNotFoundError, "图片不存在"):
            read_image(self.directory / "missing.png")
        path = self.directory / "broken.jpg"
        path.write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "图片损坏"):
            read_image(path)
        frames = [Image.new("RGB", (32, 32), color) for color in ("red", "blue")]
        path = self.directory / "animated.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:])
        with self.assertRaisesRegex(ValueError, "静态图片"):
            read_image(path)

    def test_missing_and_corrupted_checkpoint_have_clear_errors(self):
        with self.assertRaisesRegex(FileNotFoundError, "模型文件不存在"):
            load_predictor(self.directory / "missing.pt", torch.device("cpu"))
        path = self.directory / "broken.pt"
        path.write_bytes(b"not a checkpoint")
        with self.assertRaisesRegex(ValueError, "模型无法加载"):
            load_predictor(path, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()

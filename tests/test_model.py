"""v0.3 模型数学与维度检查；运行 python -m unittest discover -s tests -p test_model.py -v。"""

import unittest

import torch
from torch import nn

from src.model import BasicBlock, ResNet18


class ModelTests(unittest.TestCase):
    """验证残差相加实际生效、投影匹配奇数尺寸、完整模型具有预期接口。"""

    @classmethod
    def setUpClass(cls):
        """固定随机输入，仅使用少量 CPU 线程进行快速检查。"""
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        torch.manual_seed(42)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_zero_residual_preserves_positive_identity(self):
        """将 F(x) 置零后必须得到 ReLU(x)，能检出漏加跳跃支路的错误。"""
        block = BasicBlock(3, 3).eval()
        nn.init.zeros_(block.conv2.weight)
        x = torch.linspace(-2, 2, 2 * 3 * 9 * 11).reshape(2, 3, 9, 11)
        with torch.inference_mode():
            torch.testing.assert_close(block(x), x.relu(), rtol=0, atol=0)

    def test_projection_is_added_for_odd_spatial_size(self):
        """残差置零，手动设定投影映射；比较实际数值，不仅检查输出形状。"""
        block = BasicBlock(3, 5, stride=2).eval()
        nn.init.zeros_(block.conv2.weight)
        projection, batchnorm = block.shortcut
        with torch.no_grad():
            projection.weight.zero_()
            for channel in range(5):
                projection.weight[channel, channel % 3, 0, 0] = 1
        x = torch.arange(3 * 15 * 17, dtype=torch.float32).reshape(1, 3, 15, 17) / 100
        # 1×1 stride=2 选中 0,2,4,...；BN(eval) 用初始均值 0、方差 1 和 eps。
        expected = x[:, [0, 1, 2, 0, 1], ::2, ::2] / (1 + batchnorm.eps) ** 0.5
        with torch.inference_mode():
            torch.testing.assert_close(block(x), expected)
        self.assertEqual(list(expected.shape), [1, 5, 8, 9])

    def test_channel_only_and_stride_only_changes(self):
        """通道变化与空间变化各自都需要投影，不能只检查其中一个条件。"""
        x = torch.randn(2, 3, 13, 15)
        with torch.inference_mode():
            self.assertEqual(BasicBlock(3, 5).eval()(x).shape, (2, 5, 13, 15))
            self.assertEqual(BasicBlock(3, 3, 2).eval()(x).shape, (2, 3, 7, 8))

    def test_full_model_batch_shape_and_no_batchnorm_updates(self):
        """检查标准五类架构的参数量、真实输入尺寸及只读前向不会更新 BN 状态。"""
        model = ResNet18(5).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 11_179_077)
        before = {name: value.clone() for name, value in model.named_buffers()}
        with torch.inference_mode():
            logits = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(logits.shape, (2, 5))
        self.assertTrue(torch.isfinite(logits).all())
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_invalid_input_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "正整数"):
            ResNet18(0)
        with self.assertRaisesRegex(ValueError, "RGB"):
            ResNet18().eval()(torch.randn(1, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()

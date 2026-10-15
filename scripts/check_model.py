"""用真实训练/验证图片批次检查手写模型；随机权重不代表已经学会花卉分类。

在项目根目录运行：.venv/Scripts/python.exe scripts/check_model.py --device cuda
默认结果保存到 reports/local/model-cuda；--device cpu 可在无 GPU 环境检查。
本入口不计算准确率、不反向传播、不保存权重，也不读取测试集图片。
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import FlowerDataset, make_loader, seed_everything
from src.model import BasicBlock, ResNet18


def require(condition: bool, message: str) -> None:
    """检查失败时抛出明确异常，不能依赖可能被 python -O 关闭的 assert。"""
    if not condition:
        raise ValueError(message)


def check_residuals(device: torch.device) -> list:
    """在指定设备检查恒等/投影支路，返回实际输入输出形状与数学验证记录。"""
    records = []
    for name, cin, cout, stride in [
        ("identity", 3, 3, 1), ("channel_projection", 3, 5, 1),
        ("spatial_projection", 3, 3, 2), ("channel_and_spatial_projection", 3, 5, 2),
    ]:
        block = BasicBlock(cin, cout, stride).to(device).eval()
        # 令主支路输出为零后，完整块必须与 ReLU(shortcut(x)) 一致。
        torch.nn.init.zeros_(block.conv2.weight)
        x = torch.randn(2, cin, 15, 17, device=device)
        with torch.inference_mode():
            output = block(x)
            expected = block.shortcut(x).relu()
        torch.testing.assert_close(output, expected)
        expected_shape = (2, cout, (15 + stride - 1) // stride, (17 + stride - 1) // stride)
        require(tuple(output.shape) == expected_shape, f"{name} 的维度不正确。")
        records.append({"case": name, "input_shape": list(x.shape), "output_shape": list(output.shape),
                        "zero_residual_equals_shortcut": True})
    return records


def run_check(args) -> dict:
    """加载实际数据并运行两次只读前向；返回可追溯的设备、样本、形状及检查结论。"""
    require(args.batch_size > 0, "批量必须为正。")
    require(0 <= args.seed < 2 ** 32, "随机种子须在 0 到 2**32-1 之间。")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；检查 GPU 驱动和 PyTorch，或使用 --device cpu。")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    seed_everything(args.seed)
    # 禁止 cuDNN 自动试探算法，使当前环境下的小规模检查更容易复现。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    datasets = {split: FlowerDataset(ROOT, split, args.config) for split in ("train", "val")}
    training_data = datasets["train"]
    model = ResNet18(num_classes=len(training_data.class_to_idx)).to(device).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    require(parameter_count == 11_179_077, "当前五类 ResNet-18 参数量与预期不一致。")
    buffers_before = {name: value.clone() for name, value in model.named_buffers()}
    shapes = {}

    def capture(name):
        """前向钩子只记录形状，不修改张量或保留计算图。"""
        def hook(module, inputs, output):
            shapes[name] = list(output.shape)
        return hook

    modules = dict(model.named_modules())
    handles = [modules[name].register_forward_hook(capture(name))
               for name in ("stem.0", "stem", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc")]
    batch_reports = {}
    try:
        for split, dataset in datasets.items():
            images, labels, paths = next(iter(make_loader(dataset, args.batch_size, args.seed)))
            size = dataset.settings["image_size"]
            batch = len(images)
            require(tuple(images.shape) == (batch, 3, size, size), f"{split} 图像形状不正确。")
            require(images.dtype == torch.float32 and labels.dtype == torch.int64, "图像/标签类型不正确。")
            require(bool(torch.isfinite(images).all()), "预处理产生非有限值。")
            require(bool(((labels >= 0) & (labels < len(dataset.class_to_idx))).all()), "标签越界。")
            with torch.inference_mode():
                logits = model(images.to(device))
            require(tuple(logits.shape) == (batch, len(dataset.class_to_idx)), "模型输出形状不正确。")
            require(bool(torch.isfinite(logits).all()), "前向输出含 NaN/Inf。")
            half = (size + 1) // 2
            spatial = (half + 1) // 2
            expected = {"stem.0": [batch, 64, half, half], "stem": [batch, 64, spatial, spatial]}
            for stage, channels in enumerate((64, 128, 256, 512), start=1):
                if stage > 1:
                    spatial = (spatial + 1) // 2
                expected[f"layer{stage}"] = [batch, channels, spatial, spatial]
            expected.update(avgpool=[batch, 512, 1, 1], fc=[batch, len(dataset.class_to_idx)])
            require(shapes == expected, "逐层形状与 ResNet-18 设计不符。")
            if split == "val":
                # 验证集无随机增强：再次读同一图片，张量须逐位相同。
                torch.testing.assert_close(dataset[0][0], images[0], rtol=0, atol=0)
            batch_reports[split] = {
                "dataset_size": len(dataset), "input_shape": list(images.shape),
                "label_dtype": str(labels.dtype), "output_shape": list(logits.shape),
                "finite_input_and_logits": True, "stage_shapes": dict(shapes),
                "samples": [{"relative_path": path, "label": int(label)} for path, label in zip(paths, labels)],
            }
            print(f"{split}: images={list(images.shape)}, labels={list(labels.shape)}, logits={list(logits.shape)}")
    finally:
        for handle in handles:
            handle.remove()
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers_before[name], rtol=0, atol=0)
    require(all(parameter.grad is None for parameter in model.parameters()), "检查不应产生梯度。")
    residual_reports = check_residuals(device)
    source_paths = ("src/model.py", "src/data.py", "scripts/check_model.py", "configs/model_data.json")
    return {
        "status": "PASS", "version": "v0.3", "verified_at": datetime.now().astimezone().isoformat(),
        "purpose": "随机初始化模型的结构与数据接口检查，不是训练或分类成绩",
        "python": platform.python_version(), "torch": str(torch.__version__), "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "seed": args.seed, "num_workers": 0, "batch_size": args.batch_size,
        "parameter_count": parameter_count, "class_to_idx": training_data.class_to_idx,
        "manifest_sha256": training_data.manifest_sha256, "preprocessing": training_data.settings,
        "normalization": training_data.summary["train_rgb_statistics"],
        "batches": batch_reports, "residual_checks": residual_reports,
        "validation_preprocessing_repeatable": True, "batchnorm_buffers_unchanged": True,
        "gradients_created": False, "weights_saved": False, "test_images_used": 0,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else None,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_paths},
    }


def main() -> None:
    """解析检查参数；历史报告不覆盖，日常重跑默认进入被 Git 忽略的 local 目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8, help="结构检查批量，显存紧张可减小")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path("configs/model_data.json"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    directory = (ROOT / (args.output_dir or Path(f"reports/local/model-{args.device}"))).resolve()
    report_path = directory / "model-check.json"
    if report_path.exists() and not directory.is_relative_to((ROOT / "reports/local").resolve()):
        raise ValueError("拒绝覆盖既有验收报告；日常重跑请使用默认 reports/local 目录。")
    report = run_check(args)
    directory.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Parameters: {report['parameter_count']:,}")
    print(f"Report: {report_path}")
    print("MODEL CHECK PASSED — 随机权重结构检查通过，未训练模型。")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError, AssertionError) as error:
        print(f"MODEL CHECK FAILED: {error}", file=sys.stderr)
        sys.exit(1)

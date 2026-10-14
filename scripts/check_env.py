r"""v0.1 环境检查：验证独立环境、依赖、图表输出和 CPU / CUDA 基础运算。

在项目根目录运行：.\.venv\Scripts\python.exe scripts\check_env.py --device cuda
默认写入 reports/local/，不下载数据、不搭建 ResNet、不产生训练成绩。
"""

import argparse
import ctypes
from datetime import datetime
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_python():
    """无输入；确认解释器处于独立环境，返回版本和隔离状态。"""
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("没有使用独立环境，请用 .venv/Scripts/python.exe 运行。")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("本版本固定使用 Python 3.12，请按 README 重建环境。")
    config = (Path(sys.prefix) / "pyvenv.cfg").read_text(encoding="utf-8")
    if "include-system-site-packages = false" not in config.lower():
        raise RuntimeError("环境混用了系统包，请创建不带 --system-site-packages 的 venv。")
    return {"python": platform.python_version(), "isolated_venv": True}


def check_imports():
    """无输入；逐一导入实际依赖，返回版本，缺包或动态库错误会明确失败。"""
    versions = {}
    for name in ("numpy", "pandas", "matplotlib", "PIL", "torch", "torchvision"):
        try:
            module = importlib.import_module(name)
        except (ImportError, OSError) as exc:
            raise RuntimeError(f"无法加载 {name}：{exc}；请按 README 安装依赖。") from exc
        versions[name] = module.__version__
    return versions


def check_analysis(output_dir):
    """输入输出目录；用人工小数组检查计算、分组和绘图，返回已核对的结果。

    这里的数据只验证软件能否工作，不是花卉数据分析或模型实验。
    真正的图片统计与可视化安排在 v0.2。
    """
    import numpy as np
    import pandas as pd
    import matplotlib

    # Agg 可在没有桌面窗口时输出图片，便于命令行运行与复现。
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    row_means = values.mean(axis=1)
    np.testing.assert_allclose(row_means, [1, 4, 7, 10])
    frame = pd.DataFrame({"group": ["A", "A", "B", "B"], "value": row_means})
    means = frame.groupby("group")["value"].mean()
    np.testing.assert_allclose(means.to_numpy(), [2.5, 8.5])

    figure, axes = plt.subplots(figsize=(6, 4), layout="constrained")
    axes.bar(means.index, means.to_numpy(), color=["#238b8d", "#e6a04e"])
    axes.set(title="Synthetic dependency smoke test", xlabel="Group", ylabel="Mean")
    axes.set_ylim(0, 10)
    figure_path = output_dir / "dependency_smoke.png"
    figure.savefig(figure_path, dpi=130)
    plt.close(figure)
    # 重新打开生成文件，确认不仅完成了绘图调用，而且产物可以读取。
    with Image.open(figure_path) as picture:
        picture.load()
        if picture.size != (780, 520):
            raise RuntimeError(f"图像尺寸异常：{picture.size}")
    return {"row_means": row_means.tolist(), "group_means": means.to_dict(),
            "figure": figure_path.name, "data_kind": "synthetic_smoke_test"}


def check_image_transform():
    """无输入；将人工 RGB 图片转成张量，返回 [C,H,W] 形状。

    torchvision 仅用于基础图片变换；不调用 torchvision.models 或预训练权重。
    正式数据加载、缩放与归一化方案在 v0.3 实现。
    """
    from PIL import Image
    from torchvision.transforms import ToTensor

    picture = Image.new("RGB", (16, 16), color=(255, 0, 0))
    tensor = ToTensor()(picture)
    if list(tensor.shape) != [3, 16, 16] or tensor[0, 0, 0].item() != 1.0:
        raise RuntimeError("图片转换结果不符合 RGB 张量约定。")
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}


def check_torch(device):
    """输入 cpu 或 cuda；检查矩阵、梯度和卷积，返回设备与张量形状。

    检查真正执行计算，不能只依赖 cuda.is_available()；新显卡可能能够
    被驱动识别，却不能执行某个旧版 PyTorch 构建中的 CUDA 内核。
    """
    import torch

    torch.manual_seed(42)
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, requires_grad=True)
    product = matrix @ matrix.T
    expected = torch.tensor([[5.0, 11.0], [11.0, 25.0]], device=device)
    torch.testing.assert_close(product, expected)
    # backward 沿计算图计算梯度；此处只核对求导功能，没有训练分类器。
    product.sum().backward()
    expected_gradient = torch.tensor([[8.0, 12.0], [8.0, 12.0]], device=device)
    torch.testing.assert_close(matrix.grad, expected_gradient)

    convolution = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1).to(device)
    # [N,C,H,W] = [2,3,16,16]；padding=1 保持宽高，输出通道变为 8。
    inputs = torch.ones(2, 3, 16, 16, device=device)
    outputs = convolution(inputs)
    outputs.square().mean().backward()
    if list(outputs.shape) != [2, 8, 16, 16]:
        raise RuntimeError("卷积输出形状不正确。")
    gradient = convolution.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or gradient.abs().sum() == 0:
        raise RuntimeError("卷积梯度缺失、非有限或全为零。")
    if device == "cuda":
        # CUDA 默认异步执行；同步后才报告成功，避免隐藏设备执行错误。
        torch.cuda.synchronize()
    details = {"device": device, "matrix_and_gradient": "verified",
               "convolution_output_shape": list(outputs.shape)}
    if device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        details.update({"gpu": properties.name, "vram_gib": round(properties.total_memory / 2**30, 2),
                        "compute_capability": list(torch.cuda.get_device_capability(0)),
                        "torch_cuda_runtime": torch.version.cuda,
                        "compiled_architectures": torch.cuda.get_arch_list()})
    return details


def hardware_summary():
    """无输入；用只读系统接口收集资源，缺少可选信息时不会中断核心检查。"""
    details = {"platform": platform.platform(), "logical_cpus": os.cpu_count(),
               "disk_free_gib": round(shutil.disk_usage(PROJECT_ROOT).free / 2**30, 2)}
    if os.name == "nt":
        from ctypes import wintypes
        import winreg

        class MemoryStatus(ctypes.Structure):
            """Windows 内存状态结构；只读物理内存总量和可用量。"""
            _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total", "available", "page_total", "page_available",
                 "virtual_total", "virtual_available", "extended_available")]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            details.update({"ram_total_gib": round(status.total / 2**30, 2),
                            "ram_available_gib": round(status.available / 2**30, 2)})
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                details["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            details["cpu"] = platform.processor()
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                 "--format=csv,noheader"], capture_output=True, text=True,
                                timeout=15, check=True)
        details["nvidia_smi"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        details["nvidia_smi"] = f"未读取到（CPU 环境可忽略）：{type(exc).__name__}"
    return details


def run_check(checks, name, operation):
    """输入结果列表、检查名和无参函数；收集单项结果，让其他检查继续执行。"""
    try:
        details = operation()
        checks.append({"name": name, "status": "PASS", "details": details})
        print(f"[PASS] {name}", flush=True)
    except Exception as exc:
        checks.append({"name": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
        print(f"[FAIL] {name}: {exc}", flush=True)


def main():
    """解析命令、执行检查、保存 JSON；成功返回 0，失败返回 1。"""
    parser = argparse.ArgumentParser(description="验证 v0.1 环境；不是模型训练或正式评估。")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="auto 优先 CUDA；cuda 要求 GPU 可用；cpu 仅检查 CPU。")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports/local",
                        help="输出 JSON 和检查图；默认目录不会进入 Git。")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = PROJECT_ROOT / "reports/local/matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    checks = []
    run_check(checks, "python_venv", check_python)
    run_check(checks, "dependency_imports", check_imports)
    run_check(checks, "numpy_pandas_matplotlib", lambda: check_analysis(output_dir))
    run_check(checks, "pillow_torchvision", check_image_transform)
    run_check(checks, "torch_cpu", lambda: check_torch("cpu"))
    if args.device != "cpu":
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except (ImportError, OSError):
            cuda_available = False
        if cuda_available or args.device == "cuda":
            run_check(checks, "torch_cuda", lambda: check_torch("cuda"))
        else:
            checks.append({"name": "torch_cuda", "status": "SKIP", "details": "CUDA 不可用，已检查 CPU。"})
            print("[SKIP] CUDA 不可用；本次仅验证 CPU。", flush=True)
    passed = all(item["status"] != "FAIL" for item in checks)
    report = {"project_version": "v0.1", "checked_at": datetime.now().astimezone().isoformat(),
              "scope": "环境检查，非数据集统计、模型训练或正式实验", "requested_device": args.device,
              "status": "PASS" if passed else "FAIL", "hardware": hardware_summary(), "checks": checks}
    report_path = output_dir / "environment.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"报告：{report_path}")
    print("ENV CHECK PASSED" if passed else "ENV CHECK FAILED，请检查报告中的 FAIL 项。")
    return 0 if passed else 1


if __name__ == "__main__":
    # 控制台和保存的文件统一 UTF-8，避免 Windows 默认编码破坏中文提示。
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"ENV CHECK FAILED：无法完成检查或写入报告：{error}")
        sys.exit(1)

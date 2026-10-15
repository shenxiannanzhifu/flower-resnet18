"""保存和恢复手写模型及推理所需的元数据，不依赖当前数据集重新猜测类别。

供 scripts/train.py / scripts/check_checkpoint.py 调用。
本版保存的是推理 checkpoint；不包含优化器/随机状态，不宣称支持断点续训。
"""

import math
from pathlib import Path
import uuid

import torch

from src.data import build_transform
from src.model import ResNet18


def validate_metadata(metadata: dict) -> None:
    """核对类别索引和预处理字段，防止加载后类别错位或使用错误归一化。"""
    mapping = metadata["class_to_idx"]
    values = list(mapping.values())
    if (not values or any(type(value) is not int for value in values)
            or sorted(values) != list(range(len(values)))):
        raise ValueError("checkpoint 类别索引必须是从 0 开始的连续整数。")
    if metadata["purpose"] != "smoke":
        raise ValueError("v0.4 仅支持明确标记为 smoke 的试运行 checkpoint。")
    if not isinstance(metadata["epoch"], int) or metadata["epoch"] < 1:
        raise ValueError("checkpoint 轮数无效。")
    if not math.isfinite(metadata["validation_loss"]):
        raise ValueError("checkpoint 验证损失无效。")
    normalization = metadata["normalization"]
    build_transform(metadata["preprocessing"], normalization["mean"], normalization["std"], training=False)


def save_checkpoint(path: Path, model: ResNet18, metadata: dict) -> None:
    """写入 CPU state_dict 与 JSON 风格元数据；先写同目录临时文件再原子替换。

    CPU 张量让同一份文件可在无 GPU 的机器加载；不序列化整个 Python 模型对象。
    保存失败不会覆盖已有的完整 best.pt。临时文件只清理本函数创建的唯一文件。
    """
    validate_metadata(metadata)
    if model.fc.out_features != len(metadata["class_to_idx"]):
        raise ValueError("模型输出数量与 checkpoint 类别映射不一致。")
    payload = {"format_version": 1, "architecture": "manual_resnet18",
               "model_state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
               "metadata": metadata}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}-{uuid.uuid4().hex}.pt")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path, device: torch.device):
    """返回 (eval 模型, 元数据)；显式使用 weights_only=True，并严格检查所有参数键。"""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("architecture") != "manual_resnet18":
        raise ValueError("不支持的 checkpoint 格式或模型架构。")
    metadata = payload["metadata"]
    validate_metadata(metadata)
    model = ResNet18(num_classes=len(metadata["class_to_idx"]))
    # strict=True 不允许缺层、多层或错误尺寸被静默忽略。
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, metadata

"""按已验收清单加载花卉图像，保留类别映射及训练统计来源。

供 scripts/check_model.py 导入；运行 .venv/Scripts/python.exe scripts/check_model.py --device cpu。
这里只读取既有划分，不重新随机拆分，也不计算验证集/测试集的归一化统计。
"""

import hashlib
import json
from pathlib import Path, PurePosixPath
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def read_json(path: Path) -> dict:
    """从 UTF-8 JSON 文件读取配置；缺失时保留具体路径，便于定位准备步骤。"""
    return json.loads(path.read_text(encoding="utf-8"))


def build_transform(settings: dict, mean: list, std: list, training: bool):
    """输入配置和训练 RGB 统计，输出 PIL RGB → float32 CHW 的图像变换。

    training=True 通过裁剪/镜像增加训练变化；验证和推理采用固定缩放与中心裁剪。
    归一化公式为 (像素/255 - mean) / std，mean/std 始终来自训练集。
    """
    size = settings["image_size"]
    resize = settings["eval_resize"]
    scale, ratio = settings["train_crop_scale"], settings["train_crop_ratio"]
    flip = settings["horizontal_flip_probability"]
    if not isinstance(size, int) or size < 32 or not isinstance(resize, int) or resize < size:
        raise ValueError("image_size 须为至少 32 的整数，eval_resize 不得小于它。")
    if len(scale) != 2 or not 0 < scale[0] <= scale[1] <= 1:
        raise ValueError("train_crop_scale 须满足 0 < 下限 <= 上限 <= 1。")
    if len(ratio) != 2 or not 0 < ratio[0] <= ratio[1] or not 0 <= flip <= 1:
        raise ValueError("裁剪长宽比或镜像概率不合法。")
    if (len(mean) != 3 or len(std) != 3 or not np.isfinite(mean + std).all()
            or not all(0 <= value <= 1 for value in mean) or min(std) <= 0):
        raise ValueError("归一化须使用三个有限 RGB 均值和三个正标准差。")
    if training:
        spatial = [
            transforms.RandomResizedCrop(size, scale=tuple(scale), ratio=tuple(ratio),
                                         interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.RandomHorizontalFlip(p=flip),
        ]
    else:
        spatial = [
            transforms.Resize(resize, interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(size),
        ]
    return transforms.Compose(spatial + [transforms.ToTensor(), transforms.Normalize(mean, std)])


class FlowerDataset(Dataset):
    """读取 train/val/test CSV；每项返回 (图像张量, 整数标签, 原图相对路径)。

    project_root 是项目根目录，config_path 可选且相对根目录解析。
    初始化核对清单、统计和类别映射属于同一数据版本，避免混用旧划分。
    原图全量哈希/重复组检查仍由 scripts/verify_data.py 独立负责。
    """

    def __init__(self, project_root: Path, split: str, config_path: Path | None = None):
        if split not in ("train", "val", "test"):
            raise ValueError("split 只能是 train、val 或 test。")
        root = Path(project_root).resolve()
        self.split = split
        self.settings = read_json(root / (config_path or Path("configs/model_data.json")))
        self.raw_dir = (root / self.settings["raw_dir"]).resolve()
        processed = root / self.settings["processed_dir"]
        self.summary = read_json(root / self.settings["statistics_file"])
        self.class_to_idx = read_json(processed / "class_to_idx.json")
        values = list(self.class_to_idx.values())
        if (not values or any(type(value) is not int for value in values)
                or sorted(values) != list(range(len(values)))):
            raise ValueError("类别编号必须是从 0 开始且不重复的连续整数。")
        self.manifest_sha256 = hashlib.sha256((processed / "manifest.csv").read_bytes()).hexdigest()
        if self.manifest_sha256 != self.summary["manifest_sha256"]:
            raise ValueError("清单哈希与 RGB 统计来源不一致；请使用配套的 v0.2.1 数据。")
        split_config = read_json(processed / "split_config.json")
        if (split_config != self.summary["parameters"]
                or self.class_to_idx != split_config["class_to_idx"]):
            raise ValueError("类别映射/划分配置与统计来源不一致。")
        # 全部按字符串读，避免文件 ID、空字段或 True 被隐式转换后失去精确比较能力。
        manifest = pd.read_csv(processed / "manifest.csv", dtype=str, keep_default_na=False)
        self.frame = pd.read_csv(processed / f"{split}.csv", dtype=str, keep_default_na=False)
        expected = manifest.loc[manifest["split"] == split].reset_index(drop=True)
        if self.frame.empty or not self.frame.equals(expected):
            raise ValueError(f"{split}.csv 与总清单不一致或为空，请重新验证数据。")
        expected_labels = self.frame["class_name"].map(self.class_to_idx).map(str)
        if (not (self.frame["label"] == expected_labels).all()
                or not self.frame["relative_path"].is_unique
                or not (self.frame["included"] == "True").all()
                or not (self.frame["status"] == "valid").all()):
            raise ValueError("标签、路径唯一性或样本有效状态不正确。")
        # 数据清单中的路径只允许 类别/文件名；解析后的真实路径还必须位于原图目录。
        for row in self.frame.itertuples(index=False):
            relative = PurePosixPath(row.relative_path)
            if (relative.is_absolute() or len(relative.parts) != 2 or ".." in relative.parts
                    or "\\" in row.relative_path or ":" in row.relative_path
                    or relative.parts[0] != row.class_name):
                raise ValueError(f"不安全的图片相对路径：{row.relative_path}")
            self._image_path(row.relative_path)
        statistics = self.summary["train_rgb_statistics"]
        self.mean, self.std = statistics["mean"], statistics["std"]
        self.transform = build_transform(self.settings, self.mean, self.std, training=split == "train")

    def _image_path(self, relative: str) -> Path:
        """解析原图位置，防止路径或符号链接越过数据根目录。"""
        path = (self.raw_dir / relative).resolve()
        if not path.is_relative_to(self.raw_dir):
            raise ValueError(f"图片路径越过数据目录：{relative}")
        return path

    def __len__(self) -> int:
        """返回当前子集的样本数量。"""
        return len(self.frame)

    def __getitem__(self, index: int):
        """按索引解码原图，统一方向和 RGB，返回 (3,H,W)、标签及可追踪路径。"""
        row = self.frame.iloc[index]
        path = self._image_path(row["relative_path"])
        if not path.is_file():
            raise FileNotFoundError(f"图片不存在：{path}；请先运行 download_data.py 和 verify_data.py。")
        try:
            with Image.open(path) as image:
                # EXIF 方向来自拍摄设备；RGB 转换保证灰度/RGBA 也只有三个通道。
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                tensor = self.transform(rgb)
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError(f"图片无法解码：{path}；请运行 verify_data.py。") from error
        return tensor, int(row["label"]), row["relative_path"]


def seed_everything(seed: int) -> None:
    """固定 Python、NumPy 和 torch 随机源；不保证不同硬件/软件得到逐位相同结果。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset: FlowerDataset, batch_size: int = 8, seed: int = 42) -> DataLoader:
    """组批为 (N,3,H,W)、int64 标签与路径；仅训练集打乱，保留最后一个不足批次。

    当前受限 Windows 环境创建多进程队列时返回 WinError 5，因此本版固定 workers=0。
    增强用 seed_everything 固定；独立 generator 控制打乱顺序，避免模型初始化消耗它。
    """
    if batch_size < 1:
        raise ValueError("batch_size 必须大于 0。")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=dataset.split == "train",
                      num_workers=0, generator=generator,
                      drop_last=False)

"""单张新图推理：恢复正式模型，读取图片，并沿用训练时保存的预处理和类别顺序。

由 scripts/predict.py 调用；本模块不读取数据清单，不修改权重，也不计算新的准确率。
"""

from pathlib import Path
import pickle
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torchvision import transforms

from src.checkpoint import load_checkpoint, validate_metadata
from src.data import build_transform


CHINESE_NAMES = {"daisy": "雏菊", "dandelion": "蒲公英", "roses": "玫瑰",
                 "sunflowers": "向日葵", "tulips": "郁金香"}


def load_predictor(path: Path, device: torch.device):
    """输入模型路径和设备，返回正式模型及元数据；缺失或损坏时给出可操作的提示。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"模型文件不存在或不是文件：{path}；请按 README 生成正式模型，或使用 --checkpoint 指定。")
    try:
        model, metadata = load_checkpoint(path, device)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError,
            EOFError, pickle.UnpicklingError, IndexError) as error:
        raise ValueError(f"模型无法加载：{path}；需要本项目训练生成的完整 checkpoint（{type(error).__name__}）。") from error
    if metadata["purpose"] != "formal":
        raise ValueError("这是 smoke 试运行模型；新图入口请使用正式模型 artifacts/v0.5-formal/best.pt。")
    return model, metadata


def read_image(path: Path) -> Image.Image:
    """输入 JPEG/PNG 等静态图路径，返回已应用 EXIF 方向的独立 RGB 图像。

    灰度和 RGBA 与训练时一样转为三个通道；不重新估计输入图片的均值或标准差。
    在文件关闭前完成解码，损坏文件不会拖到后面的模型步骤才报错。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在或不是文件：{path}；请检查 --image 路径，含空格时加引号。")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as opened:
                if getattr(opened, "n_frames", 1) != 1:
                    raise ValueError("请提供单张静态图片；不自动选取动画或多页文件中的某一帧。")
                opened.load()
                return ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"图片损坏或格式不受支持：{path}；请用看图软件检查，或另存为 JPEG/PNG。") from error
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError(f"图片像素过大：{path}；请先缩小图片后再预测。") from error


def prepare_input(rgb: Image.Image, metadata: dict, view: str = "center"):
    """返回 (NCHW 张量, 实际输入图, 实际处理说明)，不改变 checkpoint 的原配置。

    center 完全沿用旧验证处理；full 等比例缩放整张图，靠模型内的自适应池化
    接收非正方形尺寸。最长边上限防止全景图片使显存占用无界增长。
    """
    normalization = metadata["normalization"]
    transform = build_transform(metadata["preprocessing"], normalization["mean"], normalization["std"], False)
    short_edge = metadata["preprocessing"]["eval_resize"]
    if view == "center":
        actual = transforms.Compose(transform.transforms[:2])(rgb)
        description = {"view": "center", "crop_applied": True, "preserves_all_image_bounds": False,
                       "resize_short_edge": short_edge, "max_long_edge": None}
    elif view == "full":
        width, height = rgb.size
        scale = min(short_edge / min(width, height), 768 / max(width, height))
        size = (round(width * scale), round(height * scale))
        if min(size) < 32:
            raise ValueError("图片长宽比过大，整图缩放后短边不足 32 像素；请提供更合适的构图。")
        actual = rgb.resize(size, Image.Resampling.BILINEAR)
        description = {"view": "full", "crop_applied": False, "preserves_all_image_bounds": True,
                       "resize_short_edge": short_edge, "max_long_edge": 768}
    else:
        raise ValueError("view 只能是 full 或 center。")
    # 两种几何方式共用 checkpoint 的 ToTensor/Normalize，绝不按新图重算 RGB 统计。
    batch = transforms.Compose(transform.transforms[-2:])(actual).unsqueeze(0)
    return batch, actual, description


def predict_image(model, metadata: dict, rgb: Image.Image, top_k: int = 3, view: str = "center"):
    """输入 eval 模型、元数据和 RGB 图，返回 (结果字典, 实际输入图)。

    center 为 (1,3,224,224)，full 为 (1,3,H,W)；两者输出均为 (1,5) logits。
    函数默认保留 center 兼容旧调用；命令行可以显式指定经过比较的模式。
    softmax 将 logits 转为五类分数；它不是经校准的正确率，也不能识别“其他类别”。
    """
    validate_metadata(metadata)
    count = len(metadata["class_to_idx"])
    if type(top_k) is not int or not 1 <= top_k <= count:
        raise ValueError(f"--top-k 须为 1 到 {count} 之间的整数。")
    batch, actual, description = prepare_input(rgb, metadata, view)
    device = next(model.parameters()).device
    model.eval()
    # eval 固定 BatchNorm；inference_mode 关闭梯度。两者作用不同，推理时都需要。
    with torch.inference_mode():
        logits = model(batch.to(device)).cpu()
    if tuple(logits.shape) != (1, count) or not bool(torch.isfinite(logits).all()):
        raise ValueError("模型输出形状错误或包含 NaN/Inf，不能生成有效预测。")
    scores = logits[0].softmax(dim=0)
    names = {index: name for name, index in metadata["class_to_idx"].items()}
    # 必须按保存的编号查类别，不能依赖 JSON 键的排列顺序；同分时按原编号稳定排序。
    ranking = torch.argsort(scores, descending=True, stable=True).tolist()
    top = [{"class_index": index, "class_name": names[index],
            "display_name": CHINESE_NAMES.get(names[index], names[index]), "score": float(scores[index])}
           for index in ranking[:top_k]]
    return {"predicted_class": top[0]["class_name"], "display_name": top[0]["display_name"],
            "score": top[0]["score"], "top_k": top,
            "probabilities": {names[index]: float(scores[index]) for index in range(count)},
            "logits": logits[0].tolist(), "input_shape": list(batch.shape),
            "original_rgb_size": list(rgb.size), "crop_size": list(actual.size),
            "model_input_size": list(actual.size), "actual_preprocessing": description,
            "class_to_idx": metadata["class_to_idx"], "preprocessing": metadata["preprocessing"],
            "normalization": metadata["normalization"],
            "note": "五类封闭分类的未校准 softmax 分数；非花卉或其他花种也会被分到其中一类，不代表真实正确率。"}, actual

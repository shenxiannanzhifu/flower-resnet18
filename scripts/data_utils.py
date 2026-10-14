"""v0.2 数据检查、重复分组与划分工具，由 prepare_data.py 调用。

不修改原始图片。各函数返回表格或数值，便于独立核对和测试；不包含模型代码。
"""

import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
SPLITS = ["train", "val", "test"]
RATIOS = np.array([0.70, 0.15, 0.15])
LICENSE_URL = "https://creativecommons.org/licenses/by/2.0/"


def read_attributions(license_path):
    """输入官方许可证路径，返回 {图片相对路径: 作者和来源}，兼容两种原文格式。"""
    attributions = {}
    for line in Path(license_path).read_text(encoding="utf-8").splitlines():
        if " CC-BY " not in line:
            continue
        filename, credit = line.split(" CC-BY ", 1)
        usual = re.fullmatch(r"by (.+?) - (https?://\S+)", credit)
        reversed_order = re.fullmatch(r"(https?://\S+) - by (.+)", credit)
        if usual:
            author, source = usual.groups()
        elif reversed_order:
            source, author = reversed_order.groups()
        else:
            raise ValueError(f"无法解析的署名行：{line}")
        if filename in attributions:
            raise ValueError(f"重复的署名路径：{filename}")
        attributions[filename] = {"author": author, "source_url": source, "license_url": LICENSE_URL}
    return attributions


def perceptual_hash(image):
    """输入 PIL 图像，返回 63 位感知哈希；用于相似候选分组，不证明图片相同。

    缩小为 32×32 灰度图，用 NumPy 计算低频 DCT 系数。去掉表示整体亮度的
    DC 项，再按中位数二值化；汉明距离越小，整体构图越接近。裁剪/旋转可能漏检。
    """
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    positions = np.arange(32) + 0.5
    frequencies = np.arange(8)[:, None]
    basis = np.cos(np.pi * frequencies * positions / 32) * np.sqrt(2 / 32)
    basis[0] /= np.sqrt(2)
    low_frequencies = (basis @ gray @ basis.T).ravel()[1:]
    bits = low_frequencies > np.median(low_frequencies)
    return sum(int(bit) << index for index, bit in enumerate(bits))


def inspect_image(path, raw_root):
    """输入图片及原图根目录，返回一条审计记录和 RGB 像素统计。

    verify 检查文件结构；重新打开并 load 才能发现部分解码错误。按 EXIF 校正
    方向后转 RGB，使像素哈希和后续读取约定一致。失败被记录，不伪造有效样本。
    """
    relative = path.relative_to(raw_root).as_posix()
    class_name = relative.split("/")[0]
    row = {"relative_path": relative, "class_name": class_name,
           "label": CLASSES.index(class_name) if class_name in CLASSES else -1,
           "width": 0, "height": 0, "mode": "", "format": "", "file_bytes": path.stat().st_size,
           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pixel_sha256": "", "phash": "",
           "source_id": path.stem.split("_")[0], "status": "valid", "error": ""}
    try:
        if class_name not in CLASSES or len(Path(relative).parts) != 2:
            raise ValueError("图片路径不能映射到五个预期类别之一")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            row.update(mode=image.mode, format=image.format)
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            row.update(width=rgb.width, height=rgb.height)
            pixels = np.asarray(rgb)
            # 把尺寸也加入哈希，避免不同形状但字节序列恰好相同的图片被当作完全相同。
            digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB:".encode() + pixels.tobytes())
            row.update(pixel_sha256=digest.hexdigest(), phash=f"{perceptual_hash(rgb):016x}")
            values = pixels.astype(np.float64) / 255
            moments = np.concatenate(([rgb.width * rgb.height], values.sum(axis=(0, 1)),
                                      np.square(values).sum(axis=(0, 1))))
        return row, moments
    except (OSError, ValueError, SyntaxError) as error:
        row.update(status="invalid", error=f"{type(error).__name__}: {error}")
        return row, np.zeros(7, dtype=np.float64)


def group_duplicates(frame, phash_distance=8, reviewed_pairs=None):
    """输入审计表，返回带保留标记/重复组的表和匹配边表。

    文件哈希、RGB 像素哈希、同一 Flickr 照片 ID、感知哈希候选均连入同组。
    A 近似 B、B 近似 C 时三者必须在同一子集；并查集把这种传递关系连起来。
    只剔除可确认的完全相同像素副本；近似候选保留，靠同组划分防止泄漏。
    """
    frame = frame.sort_values("relative_path").reset_index(drop=True).copy()
    parents = list(range(len(frame)))
    edges = []

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def connect(left, right, kind, distance=""):
        parents[find(right)] = find(left)
        edges.append({"left": frame.at[left, "relative_path"], "right": frame.at[right, "relative_path"],
                      "kind": kind, "distance": distance})

    valid = frame[frame.status == "valid"]
    for column in ("sha256", "pixel_sha256", "source_id"):
        for _, group in valid.groupby(column, sort=True):
            indices = list(group.index)
            for index in indices[1:]:
                connect(indices[0], index, column)
    hashes = [(index, int(value, 16)) for index, value in valid.phash.items()]
    for position, (left, left_hash) in enumerate(hashes):
        for right, right_hash in hashes[position + 1:]:
            distance = (left_hash ^ right_hash).bit_count()
            if distance <= phash_distance:
                connect(left, right, "phash_candidate", distance)

    # 感知哈希可能漏掉裁剪版本；人工确认的额外关系也必须可追溯、可复现。
    if reviewed_pairs is not None:
        lookup = dict(zip(frame.relative_path, frame.index))
        for pair in reviewed_pairs.itertuples(index=False):
            if pair.left not in lookup or pair.right not in lookup or not pair.reason.strip():
                raise ValueError("人工复核表存在未知图片或空的理由。")
            left, right = lookup[pair.left], lookup[pair.right]
            if frame.at[left, "status"] != "valid" or frame.at[right, "status"] != "valid":
                raise ValueError("人工复核表引用了无法解码的图片。")
            distance = (int(frame.at[left, "phash"], 16) ^ int(frame.at[right, "phash"], 16)).bit_count()
            connect(left, right, "reviewed_scene", distance)

    frame["duplicate_group"] = ""
    components = {}
    for index in valid.index:
        components.setdefault(find(index), []).append(index)
    for indices in components.values():
        first_path = min(frame.loc[indices, "relative_path"])
        group_id = "g_" + hashlib.sha256(first_path.encode()).hexdigest()[:12]
        frame.loc[indices, "duplicate_group"] = group_id
    frame["included"] = frame.status == "valid"
    frame["exclusion_reason"] = np.where(frame.included, "", "invalid_image")
    frame["duplicate_of"] = ""
    for _, group in valid.groupby("pixel_sha256", sort=True):
        if group.class_name.nunique() > 1:
            frame.loc[group.index, ["included", "exclusion_reason"]] = [False, "conflicting_exact_labels"]
        elif len(group) > 1:
            removed = group.index[1:]
            frame.loc[removed, ["included", "exclusion_reason"]] = [False, "duplicate_pixels"]
            frame.loc[removed, "duplicate_of"] = group.iloc[0].relative_path
    return frame, pd.DataFrame(edges, columns=["left", "right", "kind", "distance"])


def split_groups(frame, seed=42):
    """输入重复分组后的表，返回按类别尽量接近 70/15/15 的整组划分和目标数量。

    不能把近似组拆开以凑精确比例。使用固定种子打破同大小组的顺序，先放大组，
    每次选使各类别目标偏差增加最小的子集；支持一个近似组含多个类别。
    该过程只看类别与组大小，不看模型或测试集成绩。
    """
    frame = frame.copy()
    active = frame[frame.included]
    if active.empty:
        raise ValueError("没有可用于划分的图片。")
    group_counts = pd.crosstab(active.duplicate_group, active.label).reindex(columns=range(len(CLASSES)), fill_value=0)
    class_totals = group_counts.sum(axis=0).to_numpy()
    if (np.count_nonzero(group_counts.to_numpy(), axis=0) < 3).any():
        raise ValueError("每个类别至少需要三个独立重复组才能划分三个子集。")
    exact_targets = RATIOS[:, None] * class_totals
    targets = np.floor(exact_targets).astype(int)
    for label, total in enumerate(class_totals):
        remainder = int(total - targets[:, label].sum())
        order = np.argsort(-(exact_targets[:, label] - targets[:, label]), kind="stable")
        targets[order[:remainder], label] += 1
    rng = np.random.default_rng(seed)
    group_order = list(rng.permutation(group_counts.index.to_numpy()))
    group_order.sort(key=lambda key: -int(group_counts.loc[key].sum()))
    actual = np.zeros_like(targets)
    assignment = {}
    for group_id in group_order:
        counts = group_counts.loc[group_id].to_numpy()
        # 比较加入该组前后的平方误差增量；除以目标值，避免大类别完全主导选择。
        costs = (((actual + counts - targets) ** 2 - (actual - targets) ** 2)
                 / np.maximum(targets, 1)).sum(axis=1)
        choice = int(rng.choice(np.flatnonzero(np.isclose(costs, costs.min()))))
        assignment[group_id] = SPLITS[choice]
        actual[choice] += counts
    if (actual == 0).any():
        raise ValueError("划分后某个子集缺少类别；需要检查分组或增加数据，不能继续训练。")
    frame["split"] = "excluded"
    frame.loc[frame.included, "split"] = frame.loc[frame.included, "duplicate_group"].map(assignment)
    return frame, targets

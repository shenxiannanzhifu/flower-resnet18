r"""v0.2.1 补充数据审计：全量联系表与裁剪/旋转/镜像近重复候选。

运行：.\.venv\Scripts\python.exe scripts\audit_data.py
只生成复核材料，不自动删除、不预测类别，不替代后续手写 ResNet。
联系表保存在 reports/local/audit；每张图的编号、原始路径及署名单独记录。
"""

import argparse
from datetime import datetime
import json
from pathlib import Path, PurePosixPath
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from data_utils import CLASSES, perceptual_hash, read_attributions
from download_data import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def variants(image):
    """输入 RGB 图，返回命名变换。覆盖直角旋转、镜像、中心/角落裁剪，不穷尽所有变换。"""
    yield "original", image
    for degrees in (90, 180, 270):
        yield f"rotate{degrees}", image.rotate(degrees, expand=True)
    yield "mirror", ImageOps.mirror(image)
    # 各位置裁剪保留 80% 或 60% 宽高；用于寻找原图中子区域被另存的情况。
    for fraction in (.8, .6):
        width, height = max(1, round(image.width * fraction)), max(1, round(image.height * fraction))
        for name, x, y in [("center", .5, .5), ("tl", 0, 0), ("tr", 1, 0), ("bl", 0, 1), ("br", 1, 1)]:
            left, top = round((image.width - width) * x), round((image.height - height) * y)
            yield f"crop{int(100*fraction)}_{name}", image.crop((left, top, left+width, top+height))


def find_candidates(hashes, threshold=8):
    """输入 N×V 个 uint64 哈希，比较每个变换与所有原图，返回去重候选和最小距离。

    NumPy 按块广播，避免创建 N×N×V 的巨大数组。候选不等于重复：花朵构图可能相似，
    必须视觉复核后才写入分组配置。与原图比对可能漏掉双方都被裁剪的版本。
    """
    if hashes.ndim != 2 or hashes.shape[0] == 0 or hashes.shape[1] == 0:
        raise ValueError("哈希输入必须是非空的 N×V 数组。")
    matches = {}
    originals = hashes[:, 0]
    for start in range(0, len(hashes), 128):
        block = hashes[start:start+128]
        for variant_index in range(hashes.shape[1]):
            distances = np.bitwise_count(block[:, variant_index, None] ^ originals[None, :])
            lefts, rights = np.where(distances <= threshold)
            for local, right in zip(lefts.tolist(), rights.tolist()):
                left = start + local
                if left == right:
                    continue
                pair = tuple(sorted((left, right)))
                value = (int(distances[local, right]), left, right, variant_index)
                if pair not in matches or value < matches[pair]:
                    matches[pair] = value
    return matches


def contact_sheet(items, raw_root, output, columns=8, cell_width=180, cell_height=165):
    """把 [(编号, 图片路径, 简短说明)] 拼成带编号联系表；保留纵横比，不裁掉诊断内容。"""
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new("RGB", (columns*cell_width, rows*cell_height+25), "white")
    draw = ImageDraw.Draw(canvas)
    # 图片编号使用 ASCII；其他系统没有 Windows 字体时仍可生成复核材料。
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default(size=14)
    for index, (identifier, relative, note) in enumerate(items):
        x, y = (index % columns)*cell_width, (index // columns)*cell_height
        with Image.open(raw_root / relative) as source:
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            rgb.thumbnail((cell_width-6, cell_height-35))
            canvas.paste(rgb, (x+(cell_width-rgb.width)//2, y+(cell_height-35-rgb.height)//2))
        draw.text((x+3, y+cell_height-32), str(identifier), font=font, fill="black")
        draw.text((x+3, y+cell_height-16), note, font=font, fill="black")
    draw.text((5, rows*cell_height+4),
              "CC BY 2.0 | Pair paths: transform_candidates.csv | Authors / sources: review_index.csv",
              font=font, fill="black")
    canvas.save(output, quality=91, optimize=True)


def audit(args):
    """生成全量分页索引和变换候选；不会把未看过的图片标记为已复核。"""
    raw, output = args.raw_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.manifest, keep_default_na=False).sort_values("relative_path").reset_index(drop=True)
    if frame.empty or not frame.relative_path.is_unique:
        raise ValueError("清单必须非空且图片路径唯一。")
    credits = read_attributions(raw / "LICENSE.txt")
    records, hashes = [], []
    for index, row in enumerate(frame.itertuples(index=False)):
        relative = PurePosixPath(row.relative_path)
        path = (raw / row.relative_path).resolve()
        if relative.is_absolute() or ".." in relative.parts or "\\" in row.relative_path or not path.is_relative_to(raw):
            raise ValueError(f"清单路径超出原图目录：{row.relative_path}")
        if sha256_file(path) != row.sha256:
            raise ValueError(f"原图内容与基线不符：{row.relative_path}")
        with Image.open(path) as source:
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            named = list(variants(rgb))
            hashes.append([perceptual_hash(image) for _, image in named])
        records.append({"image_id": f"I{index+1:04d}", "relative_path": row.relative_path,
                        "class_name": row.class_name, "sha256": row.sha256,
                        **credits[row.relative_path], "modification": "等比例缩略展示，原图未修改"})
        if (index+1) % 500 == 0:
            print(f"已计算 {index+1}/{len(frame)} 张图的变换哈希", flush=True)
    index_frame = pd.DataFrame(records)
    index_frame["page"] = ""
    # 按原始类别连续分页，标签不依赖模型评分，也不使用测试集准确率。
    for name in CLASSES:
        subset = index_frame[index_frame.class_name == name]
        for start in range(0, len(subset), 64):
            page = f"labels_{name}_{start//64+1:02d}.jpg"
            selected = subset.iloc[start:start+64]
            index_frame.loc[selected.index, "page"] = page
            items = [(r.image_id, r.relative_path, name) for r in selected.itertuples()]
            contact_sheet(items, raw, output / page)
    index_frame.to_csv(output / "review_index.csv", index=False, lineterminator="\n")
    hash_array = np.asarray(hashes, dtype=np.uint64)
    np.save(output / "variant_hashes.npy", hash_array)
    names = [name for name, _ in named]
    candidates = []
    for pair, (distance, left, right, variant) in sorted(find_candidates(hash_array, args.distance).items()):
        a, b = pair
        candidates.append({"candidate_id": f"C{len(candidates)+1:04d}",
                           "left": frame.iloc[a].relative_path, "right": frame.iloc[b].relative_path,
                           "distance": distance, "transformed_path": frame.iloc[left].relative_path,
                           "variant": names[variant]})
    # 没有候选时也写表头，后续 read_csv 不会把合法空结果误报为坏文件。
    columns = ["candidate_id", "left", "right", "distance", "transformed_path", "variant"]
    pd.DataFrame(candidates, columns=columns).to_csv(output / "transform_candidates.csv", index=False, lineterminator="\n")
    for start in range(0, len(candidates), 20):
        items = []
        for pair in candidates[start:start+20]:
            for side in ("left", "right"):
                items.append((pair["candidate_id"] + side[0], pair[side], f"d={pair['distance']} {pair['variant']}"))
        contact_sheet(items, raw, output / f"pairs_{start//20+1:02d}.jpg", columns=8, cell_width=200, cell_height=180)
    summary = {"generated_at": datetime.now().astimezone().isoformat(), "images": len(frame),
               "variants": names, "distance_threshold": args.distance, "candidate_pairs": len(candidates),
               "baseline_manifest_sha256": sha256_file(args.manifest),
               "status": "materials_generated_not_reviewed",
               "limits": "有限变换候选检索，不保证穷尽裁剪、任意角旋转及不同视角；视觉复核另行记录。"}
    (output / "audit_parameters.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"生成全量 {len(frame)} 张图片联系表、{len(candidates)} 对变换候选。AUDIT MATERIALS READY")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="生成全量视觉复核与扩展近重复候选材料。")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/flower_photos")
    parser.add_argument("--manifest", type=Path, default=ROOT / "reports/v0.2/manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/local/audit")
    parser.add_argument("--distance", type=int, default=8, choices=range(0, 9),
                        help="变换哈希候选阈值；默认 8 与本次复核一致，不等于自动判定重复。")
    try:
        audit(parser.parse_args())
    except (OSError, ValueError, KeyError) as error:
        print(f"AUDIT FAILED：{error}")
        sys.exit(1)

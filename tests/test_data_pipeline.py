r"""v0.2 防回归检查：真实故障边界与防泄漏行为，而不是复述实现。

运行：.\.venv\Scripts\python.exe -m unittest discover -s tests -v
所有人工样本均位于 reports/local/tests，不读取或修改正式图片。
"""

from pathlib import Path
import shutil
import sys
import tarfile
import unittest
import uuid

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
from data_utils import CLASSES, inspect_image, read_attributions, group_duplicates, split_groups, apply_review_exclusions
from download_data import validate_member
from verify_data import verify_invariants


class DataPipelineTests(unittest.TestCase):
    """检查损坏文件、许可变体、危险路径、重复处理与整组划分。"""

    def setUp(self):
        """每个用例创建独立工作区目录；继承工作区权限，避免系统临时目录 ACL 问题。"""
        self.base = (ROOT / "reports/local/tests").resolve()
        self.directory = self.base / uuid.uuid4().hex
        self.directory.mkdir(parents=True)

    def tearDown(self):
        """只清理该用例创建的目录，先检查绝对路径仍位于测试工作区。"""
        resolved = self.directory.resolve()
        if not resolved.is_relative_to(self.base) or resolved == self.base:
            raise RuntimeError("拒绝清理测试目录之外的路径。")
        shutil.rmtree(resolved)

    def test_license_both_orders(self):
        """实际许可证有九条倒序条目；两种写法都必须保留作者和来源。"""
        path = self.directory / "LICENSE.txt"
        path.write_text("daisy/a.jpg CC-BY by Alice - https://example.org/a\n"
                        "roses/b.jpg CC-BY https://example.org/b - by Bob\n", encoding="utf-8")
        result = read_attributions(path)
        self.assertEqual(result["roses/b.jpg"]["author"], "Bob")
        self.assertEqual(result["daisy/a.jpg"]["source_url"], "https://example.org/a")

    def test_corrupted_image_is_excluded(self):
        """损坏 JPEG 不能靠扩展名冒充可训练图像。"""
        path = self.directory / "daisy/broken.jpg"
        path.parent.mkdir()
        path.write_bytes(b"not an image")
        row, _ = inspect_image(path, self.directory)
        self.assertEqual(row["status"], "invalid")
        self.assertTrue(row["error"])

    def test_rgb_statistics_known_pixels(self):
        """使用已知 RGB 像素检查缩放与平方和，避免错误的训练归一化统计。"""
        path = self.directory / "daisy/tiny.png"
        path.parent.mkdir()
        Image.fromarray(np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)).save(path)
        row, moments = inspect_image(path, self.directory)
        self.assertEqual((row["width"], row["height"]), (2, 1))
        np.testing.assert_allclose(moments, [2, 1, 1, 0, 1, 1, 0])

    def test_tar_rejects_escape_and_links(self):
        """压缩包不能写到数据目录之外，也不能借符号链接逃逸。"""
        for name in ["../outside.txt", "/absolute.txt", "flower_photos/../../outside", "flower_photos/C:stream"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_member(tarfile.TarInfo(name))
        link = tarfile.TarInfo("flower_photos/link")
        link.type = tarfile.SYMTYPE
        with self.assertRaises(ValueError):
            validate_member(link)

    def test_exact_conflicts_and_duplicates(self):
        """跨类别完全相同图片全部隔离；同类副本只留一个，近似候选不直接删除。"""
        rows = []
        for index, (label, digest, phash) in enumerate([(0, "a", "0"), (1, "a", "0"),
                                                     (2, "b", "7fffffffffffffff"), (2, "b", "7fffffffffffffff")]):
            rows.append({"relative_path": f"{CLASSES[label]}/{index}.jpg", "class_name": CLASSES[label],
                         "label": label, "status": "valid", "sha256": digest, "pixel_sha256": digest,
                         "phash": phash, "source_id": str(index)})
        frame, _ = group_duplicates(pd.DataFrame(rows))
        self.assertEqual(frame.included.sum(), 1)
        self.assertEqual((frame.exclusion_reason == "conflicting_exact_labels").sum(), 2)
        self.assertEqual((frame.exclusion_reason == "duplicate_pixels").sum(), 1)

    def fixture(self):
        """构造每类六个双样本组，用来检查划分互斥和类别覆盖。"""
        return pd.DataFrame([{"relative_path": f"{name}/{group}-{copy}.jpg", "label": label,
                              "class_name": name, "duplicate_group": f"g{label}-{group}", "included": True}
                             for label, name in enumerate(CLASSES) for group in range(6) for copy in range(2)])

    def test_reviewed_pair_joins_hash_miss_without_deleting(self):
        """视觉确认关系要能补足哈希漏检，同时不能把不同像素样本误删。"""
        frame = pd.DataFrame([{"relative_path": f"sunflowers/{i}.jpg", "class_name": "sunflowers",
                               "label": 3, "status": "valid", "sha256": str(i), "pixel_sha256": str(i),
                               "source_id": str(i), "phash": h}
                              for i, h in enumerate(["0", "7fffffffffffffff"])])
        reviewed = pd.DataFrame([{"left": "sunflowers/0.jpg", "right": "sunflowers/1.jpg", "reason": "已复核相同场景"}])
        result, _ = group_duplicates(frame, reviewed_pairs=reviewed)
        self.assertEqual(result.duplicate_group.nunique(), 1)
        self.assertEqual(result.included.sum(), 2)

    def test_review_exclusion_preserves_label_and_rejects_unknown_path(self):
        """明显噪声被隔离但不改标签；复核表拼错路径必须报错，不能默默跳过。"""
        frame = self.fixture()
        frame["exclusion_reason"] = ""
        path = frame.iloc[0].relative_path
        reviews = pd.DataFrame([{"relative_path": path, "reason": "明显非目标图片"}])
        result = apply_review_exclusions(frame, reviews)
        self.assertFalse(result.iloc[0].included)
        self.assertEqual(result.iloc[0].label, frame.iloc[0].label)
        reviews.loc[0, "relative_path"] = "unknown.jpg"
        with self.assertRaises(ValueError):
            apply_review_exclusions(frame, reviews)

    def test_split_is_grouped_stratified_and_reproducible(self):
        frame = self.fixture()
        first, _ = split_groups(frame, seed=42)
        second, _ = split_groups(frame.sample(frac=1, random_state=19), seed=42)
        self.assertEqual(first.groupby("duplicate_group").split.nunique().max(), 1)
        self.assertEqual(first.set_index("relative_path").split.to_dict(), second.set_index("relative_path").split.to_dict())
        for split in ["train", "val", "test"]:
            self.assertEqual(set(first[first.split == split].class_name), set(CLASSES))

    def test_too_few_independent_groups_fails(self):
        frame = self.fixture()
        frame["duplicate_group"] = "one"
        with self.assertRaises(ValueError):
            split_groups(frame)

    def test_verifier_rejects_cross_split_group(self):
        frame, _ = split_groups(self.fixture())
        frame["status"] = "valid"
        frame["exclusion_reason"] = ""
        frame["sha256"] = frame.relative_path
        frame["pixel_sha256"] = frame.relative_path
        frame["source_id"] = frame.relative_path
        frame["phash"] = "0"
        group = frame.iloc[0].duplicate_group
        indices = frame[frame.duplicate_group == group].index
        frame.loc[indices[0], "split"] = "train"
        frame.loc[indices[1], "split"] = "test"
        with self.assertRaisesRegex(ValueError, "跨子集泄漏"):
            verify_invariants(frame, 8)

    def test_verifier_does_not_trust_group_ids_alone(self):
        """即使已有组 ID 不跨集，独立哈希复查也要发现漏分组的近似图片。"""
        frame, _ = split_groups(self.fixture())
        frame["status"] = "valid"
        frame["exclusion_reason"] = ""
        for column in ["sha256", "pixel_sha256", "source_id"]:
            frame[column] = frame.relative_path
        frame["phash"] = "0"
        with self.assertRaisesRegex(ValueError, "pHash"):
            verify_invariants(frame, 8)


if __name__ == "__main__":
    unittest.main()

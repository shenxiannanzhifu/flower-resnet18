# 数据目录

v0.2 已完成官方下载、完整性/许可/重复检查、数据划分和统计图表。原始 3,670 张，保留 3,665 张：train=2564、val=551、test=550。

采用 TensorFlow 示例花卉图片集（tf_flowers / flower_photos）：

- 官方目录：https://www.tensorflow.org/datasets/catalog/tf_flowers
- 下载地址：https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
- 官方目录列出 3,670 张图片、5 个类别，下载包 218.21 MiB。
- 初步类别：daisy、dandelion、roses、sunflowers、tulips。
- 2026-10-14 已实际访问目录与下载地址（HEAD 200，228,813,984 字节），没有下载图片包。

2026-10-14 实际下载并校验：压缩包 228,813,984 字节，包含 3,670 个 JPG 和一个 LICENSE.txt；3,671 个解压文件均逐一与压缩包一致。全部图片可解码且为 RGB；处理时隔离 2 张跨类冲突图片、2 张同类副本和 1 张明显非花卉图片，原始文件不删除。

许可核查：压缩包 LICENSE.txt 明确声明所有图片采用 **CC BY 2.0**，并列出逐图作者和 Flickr 来源；实际解析出 3,670 条署名，与所有 JPG 一一对应。完整原文保存在 `docs/data-license.txt`。使用/分享图片时须署名、链接原始来源及许可证，并说明缩放等修改。样本图会附对应署名清单；原图和压缩包不进入 Git。许可链接：https://creativecommons.org/licenses/by/2.0/ 。

下载和离线校验（在项目根目录执行）：

```powershell
& '.\.venv\Scripts\python.exe' scripts\download_data.py
& '.\.venv\Scripts\python.exe' scripts\download_data.py --offline
```

成功标志：`DATA DOWNLOAD VERIFIED`。如网络不通，可通过上方官方下载地址获取压缩包，放到 `data/downloads/flower_photos.tgz` 后运行离线命令。

官方下载校验清单：https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/datasets/tf_flowers/checksums.tsv 。清单中的 HTTP 源与本项目 HTTPS 镜像对应同一文件，大小和 SHA-256 均匹配。预期 SHA-256 为 `4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c`，来自该独立官方清单。脚本安全解压、保留原图，发现已有原图被修改时停止，不覆盖用户文件。

划分原则：先查重并隔离噪声，再以整组分配尽量达到每类 70%/15%/15%，固定种子 42。使用 SHA-256、RGB 像素哈希、Flickr 照片 ID、63 位 pHash（阈值 8）和已记录的视觉复核关系共同构组。近似候选不直接删除；组内样本不能跨子集。验证集用于选择模型，测试集留到正式评估，不用于调参。

目录：`downloads/` 保存官方包，`raw/flower_photos/` 保存原图，`processed/` 保存可用清单和类别映射。划分只写清单，不复制三份图片。生成入口为 `scripts/prepare_data.py`，独立校验入口为 `scripts/verify_data.py`；清单快照、图表和真实日志在 `reports/v0.2/`。详细字段和复现方法见 `docs/data.md`。

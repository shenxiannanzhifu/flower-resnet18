# v0.2.1 数据流程与阅读指南

当前只准备图像分类数据，未实现 PyTorch Dataset、ResNet、训练或预测。依赖与 v0.1 相同，无新增第三方包。

## 本版修复与历史边界

2026-10-15 接续 v0.2 后的未完成审计工具，逐对查看 45 对变换候选：30 对已有分组，15 对新增候选。C0039 的两张向日葵照片属于同场景相邻拍摄，原图 pHash 距离 20，中心裁剪后距离 8；v0.2 中分别在 train/val，现均在 train。新增关系写入 `configs/duplicate_review.csv`，两张图都保留，不据此改变类别。

另在首轮修复样本图 S14 发现 `tulips/17078716890_68e0723389_n.jpg` 主体为风信子和水仙，查看原图后加入 `configs/excluded_review.csv`。最终保留 **3,664** 张、排除 6 张；训练/验证/测试为 **2,564 / 551 / 549**。首轮 3,665 张是实际中间结果，已在本地归档，不作为最终验收。

默认数据清单改为 `data/processed-v0.2.1`，日常报告为 `reports/local/data-v0.2.1`。旧 `data/processed` 与 `reports/v0.2` 保留。分组与保留集合改变后重新运行原算法，1,678 张的 split 字段变化（含新增排除样本），并重新计算训练集 RGB 统计。详见 `reports/v0.2.1/revision_comparison.json`、`split_changes.csv` 与 `summary.json`。当前没有任何训练/测试成绩参与划分决定。

复核工具仍使用 v0.2 主清单作为候选生成基线，以便复现候选 ID；这不表示后续模型继续使用 v0.2 划分。审计脚本的 `materials_generated_not_reviewed` 只说明生成材料的自动化步骤；单独的 `review_decisions.csv` 记录实际 AI 辅助复核结论和时间。完整联系表已生成，未逐张审查所有标签；其余 14 对新增候选未确认同源，记录理由并保持现状，不能宣称彻底排除潜在关系。

## 流程和文件

1. `download_data.py` 获取官方压缩包，使用独立官方清单的 SHA-256 检查完整性，安全解压，再逐文件核对。重复运行复用已有包；已有原图被改变时停止，不覆盖。
2. `prepare_data.py` 解析全部署名，检查图片和标签路径，调用 `data_utils.py` 计算哈希、建立重复组、排除已知问题并按组划分，输出统计与图表。
3. `verify_data.py` 重新读回 CSV，核对磁盘原图哈希、子集覆盖和类别映射；不只相信组 ID，还独立比较跨子集的感知哈希并检查视觉复核配置。
4. `audit_data.py` 对原图、直角旋转、镜像、80%/60% 中心和角落裁剪计算 15 种视图的哈希。阈值 8 生成候选及带编号图表；只有视觉判断及理由写入复核配置后，才由准备入口重新划分。脚本不自动删除、改标签或确认重复。

`data/processed-v0.2.1/manifest.csv` 是后续模型代码应使用的主清单，`train.csv`、`val.csv`、`test.csv` 是其可用子集。原图根目录统一为 `data/raw/flower_photos`；把它与 `relative_path` 拼接得到图片路径。

| 字段 | 意义 |
| --- | --- |
| relative_path、class_name、label | 原图相对路径、类别名、固定的 0–4 类别编号 |
| width、height、mode、format、file_bytes | EXIF 校正后的尺寸及原文件信息 |
| sha256、pixel_sha256 | 文件字节和校正后 RGB 像素的哈希；后者同时包含宽高 |
| phash、source_id | 63 位感知哈希（十六进制）及文件名中的 Flickr 照片 ID |
| status、error | 是否能正常解码及失败原因 |
| duplicate_group | 所有已知相同或近似关系的连通组 |
| included、exclusion_reason、duplicate_of | 是否保留、为什么排除、同类副本的保留对象 |
| split | train、val、test 或 excluded |

类别映射保存在 `class_to_idx.json`。`split_config.json` 记录种子、比例、阈值、复核表哈希和整个数据集指纹；不要仅凭目录名猜测类别编号。

## 重复与标签问题如何处理

完全相同像素、类别相同：按路径排序保留一份，额外副本标记为 `duplicate_pixels`。完全相同像素却有两个类别：全部标记为 `conflicting_exact_labels`，不凭猜测改标签。

近似图片：保留图片并合为同一组。同一照片的黑白版、调色版或压缩版不能通过随机划分进入不同子集。pHash 阈值 8 得到 30 条候选边，部分仅外观相近，并不证明是同一张照片；保守合组不会删除这些样本。

v0.2 的 AI 辅助视觉复核发现两张向日葵图片的 pHash 距离为 18，却有相同花朵和建筑背景，属于裁剪或相邻拍摄场景。因此在 `configs/duplicate_review.csv` 明确记录关系及理由；它们在早期试划分中曾跨 train/test，现已合组。v0.2.1 又补充 C0039 关系，现在有 27 个多图组，最大组仍含 4 张原图。

另在训练样本展示中发现 `roses/3713368809_eba7fa2fbf_m.jpg` 为历史装束人物活动场景，未见可用于该分类的玫瑰。通过 `configs/excluded_review.csv` 标记为 `visual_label_noise`，保留原图和原标签。没有让模型判断、没有查看模型成绩，也没有根据测试集效果做清洗。

v0.2 最终排除 5 张、保留 3,665 张；该历史证据不变。v0.2.1 增加上述非郁金香样本后，最终排除 6 张、保留 3,664 张。清洗策略在训练前确定；需要修复时使用新目录、新提交和补丁标签，明确记录比较条件变化。

## 分层与整组的取舍

目标为每类 70%/15%/15%。先计算整数目标，再按组大小从大到小分配；同大小组用 NumPy 固定种子 42 决定顺序，优先减少类别目标偏差。组可能含多个类别，因此不能简单按类别分别随机切片。不能为了凑比例拆组。

本版实际数目达到整数目标：训练 2,564、验证 551、测试 549。每个子集都包含五类；没有已识别的重复关系跨子集。对大幅裁剪、旋转和不同视角仍可能漏检，不能宣称彻底消除了所有数据泄漏。

## NumPy、Pandas、Matplotlib 的实际使用

Pandas 整理逐图审计表，按类别、状态、重复组和子集分组统计，导出 CSV。NumPy 计算低频感知哈希、图片尺寸/宽高比分位数、划分目标和训练集 RGB 像素加权均值/标准差。Matplotlib 根据这些真实数值绘制类别/子集分布、尺寸分布、训练样本与问题图片网格。

RGB 统计只使用训练集，范围为 0–1，且来自未缩放/裁剪/增强的原图。它们是本阶段的真实分析结果，不等于已经实现或确定了下一阶段的全部预处理策略。

## 复现与故障排查

在项目根目录运行 README 中的下载、准备、验证命令。相同输入和参数可以重跑；默认图表在 `reports/local/data-v0.2.1/`，避免覆盖版本快照。不同输入或参数不能覆盖旧数据目录或报告目录，否则会悄悄改变留出的测试集或历史证据。

验证完整重跑是否相同：

```powershell
& '.\.venv\Scripts\python.exe' scripts\prepare_data.py --output-dir data\processed-v0.2.1-final-recheck --report-dir reports\local\data-v0.2.1-final-recheck
& '.\.venv\Scripts\python.exe' scripts\verify_data.py --processed-dir data\processed-v0.2.1-final-recheck --report-dir reports\local\data-v0.2.1-final-recheck
Get-FileHash -LiteralPath 'data\processed-v0.2.1\manifest.csv','data\processed-v0.2.1-final-recheck\manifest.csv' -Algorithm SHA256
```

两份哈希应相同。实际验收还逐字节核对了三个子集 CSV、重复关系表、类别映射和配置，共七份文件；结果在 `reports/v0.2.1/reproducibility.json`。自动测试命令为 `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`，16 个用例覆盖真实发现的问题和失效边界。

旧 v0.2 的配置可在 `v0.2` 标签中查看。当前复核表已经更新，拿它核验旧清单会按设计报配置不一致；不要通过改旧哈希或覆盖旧报告消除错误，应使用新版默认清单。

下载失败：先检查网络/已启用的系统代理，或手工将官方包放入 `data/downloads/` 后使用 `--offline`。哈希不符：不要跳过校验，应检查下载来源。署名不全、缺少类别、图片被修改、复核文件路径写错、组跨子集：脚本会停止并报告原因，先处理当前问题再继续。

## 许可与边界

官方 LICENSE.txt 声明全部图片为 CC BY 2.0，原文及全部摄影者署名保存在 `docs/data-license.txt`。最初解析遗漏九条倒序格式，现已兼容并核对 3,670/3,670 全覆盖。

报告图片中的 S/P/E 编号在 `image_attributions.csv` 中对应摄影者、原图来源、[许可证](https://creativecommons.org/licenses/by/2.0/)和缩略图修改说明。分享报告图片时应同时保留署名表和许可说明；原始包及原图不入库。

扩展候选图 `pairs_01.jpg` 至 `pairs_03.jpg` 中的 C 编号对应 `transform_candidates.csv`，其图片作者和来源见同目录 `review_index.csv`。v0.2.1 仅将已复核的三页候选图及对应署名入库；60 页全量标签联系表与哈希数组留在被忽略的 `reports/local/`，并未上传。

当前验证了结构、解码、已知重复和抽样视觉问题，未人工逐张核对所有类别。部分图片含人物、背景、文字或明显编辑效果，可能形成偏差；这些局限应保留到后续错误分析，不能提前声称模型具有可靠泛化能力。

# v0.2.1 数据修复验收报告

2026-10-15，接续本仓库的 v0.2 和未完成的审计脚本。当前仅完成数据质量补丁，不包含模型或训练成绩。v0.1/v0.2 的原始提交、标签和报告保持原样。

## 修复了什么

1. 重算全部 3,670 张图的 15 种视图哈希，阈值 8 生成 45 对候选，与先前材料一致。逐对查看三页候选图，C0010 和 C0039 还单独查看原图。30 对已有分组/排除，15 对是新增候选；其中 C0039 确认是同场景相邻拍摄，其他 14 对未确认同源关系，逐项理由保存在 `review_decisions.csv`。
2. C0039 两图的原图 pHash 距离 20，中心裁剪 80% 后为 8。v0.2 将其放入 train/val；补充 `configs/duplicate_review.csv` 后整组划分为 train/train。保留两张图，不改标签。对比图见 `duplicate_pairs_4.jpg` 的 P32-L/R。
3. 首轮修复样本图 S14 显示标为 tulips 的 `17078716890_68e0723389_n.jpg` 实际主体为紫色风信子和白黄色水仙。查看原图后作为明显标签噪声隔离，原文件与 tulips 标签不改写；最终例图见 `excluded_review.jpg`。
4. 默认数据输出改为 `data/processed-v0.2.1`；同时保护旧数据和旧报告的配置。旧 `data/processed` 保留，当前模型阶段不得混用旧划分或旧 RGB 统计。

## 最终实际结果

| 项目 | 结果 |
| --- | --- |
| 原图完整解码 / SHA-256 核对 / 署名覆盖 | 3,670 / 3,670 / 3,670 |
| 保留 / 排除 | 3,664 / 6 |
| 训练 / 验证 / 测试 | 2,564 / 551 / 549 |
| 排除原因 | 跨类相同像素 2，同类副本 2，视觉标签噪声 2 |
| 多图组 / 最大组 | 27 / 4 |
| 已知重复关系跨集 | 0 |
| 回归测试 | 16 项全部通过 |
| 完整重跑 | 7 份核心文件逐字节一致 |
| 修复前后 split 字段改变 | 1,678 张，含 1 张新增排除 |

1,678 张发生变化是分组和保留集合改变后重新运行原有整组分层算法的结果；种子仍为 42。原图内容与原始类别标签完全不变，当前尚无任何训练成绩用于选择划分。`split_changes.csv` 记录每张受影响图片的前后 split/group，`revision_comparison.json` 记录基线和新版清单哈希。

首轮 3,665 张、split 改变 1,668 张是发现新标签问题之前的真实中间结果，已归档在本地 `reports/local/v0.2.1-before-label-review`，不作为本版最终验收。

## 实际运行证据

- `environment-recheck.json`：2026-10-15 在原 .venv 重新检查，六项 PASS，CPU 与 CUDA 运算通过；其中 `project_version=v0.1` 是环境检查工具的阶段标记，不是本次项目版本。
- `audit.log`、`audit_parameters.json`：全量候选生成记录。自动状态 `materials_generated_not_reviewed` 属于生成步骤，实际后续结论另存在 `review_decisions.csv`，不修改自动记录来假装人工复核。
- `prepare.log`、`summary.json`、`class_counts.csv`：最终完整准备、训练像素统计和分布。
- `verify.log`、`verification.json`：独立核查全部原图、清单及已知关系，输出 `DATA VERIFICATION PASSED`。
- `reproduce.log`、`reproduce-verify.log`、`reproducibility.json`：另一个数据目录完整重跑，主清单、三个子集、重复关系、类别映射与配置共七份文件一致。
- `tests.log`：16 个测试 PASS，包含真实 C0039 漏分组、旋转副本检索、历史报告保护和新标签噪声隔离。
- `compile.log`：对 scripts、tests、src 执行 `python -W error -m compileall -q` 通过。

在项目根目录复核：

```powershell
& '.\.venv\Scripts\python.exe' scripts\verify_data.py
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

完整重建和复核候选的命令见根目录 README 与 `docs/data.md`。候选图只展示原图，图上的变换名说明检索时使用的变换，不是已对展示图做了该变换。

## 署名与范围

`samples.jpg`、`duplicate_pairs_*.jpg`、`excluded_review.jpg` 的图号对应 `image_attributions.csv`；`pairs_01.jpg` 至 `pairs_03.jpg` 的 C 编号对应 `transform_candidates.csv`，对应摄影者、原始来源和 CC BY 2.0 许可见 `review_index.csv`。所有照片只为报告缩略展示，原图未修改。完整许可原文在 `docs/data-license.txt`。

已生成全部 3,670 张图片的 60 页联系表，但本次未逐张审查全部类别标签；这些材料留在被忽略的本地目录。AI 辅助视觉复核不是专家标注，有限变换检索也不能穷尽强裁剪、不同视角或背景偏差。不把“已知关系无跨集”表述成“彻底无泄漏”。

最终样本图 S09 是含玫瑰图案的杯碟，仍能看到原标签对应的视觉内容，因此保留原始类别；这也说明数据不限于自然花朵实拍。器物图案、人物背景和编辑效果等分布偏差需要在后续模型错误分析中讨论，不能提前宣称对实物花卉的泛化可靠。

下一阶段为 v0.3：使用本版清单，实现带中文注释的 BasicBlock、手写 ResNet-18、加载与预处理、实际批次前向和维度检查。等待用户确认后再开始。

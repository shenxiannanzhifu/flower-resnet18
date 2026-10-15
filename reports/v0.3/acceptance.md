# v0.3 模型与数据接口验收

2026-10-15，沿用原仓库及 Python 3.12.14、PyTorch 2.8.0+cu128，无新增第三方依赖。
本版已通过结构验收；**权重随机初始化，尚未训练，不报告准确率。**

## 实际验证及证据

| 检查 | 实际结果 | 文件 |
| --- | --- | --- |
| 全量数据复核 | 3,670 原图哈希/署名通过；保留 3,664；train/val/test = 2,564/551/549；已知重复跨集 0 | `verification.json`、`data-verification.log` |
| 第一轮残差及模型检查 | 5 个检查通过后作第一笔实现提交 `21a2086` | `model-unit-tests.log` |
| 完整回归 | 16 个原有数据检查 + 5 个模型检查 + 7 个新数据接口检查，共 28 个全部通过 | `all-tests.log` |
| CPU 实图前向 | 训练/验证各 8 张，输入 `[8,3,224,224]` → logits `[8,5]`；逐层形状、有限值通过 | `cpu/model-check.json`、`cpu-check.log` |
| CUDA 实图前向 | RTX 5060 Laptop GPU，同样的两个实际批次及检查通过 | `cuda/model-check.json`、`cuda-check.log` |
| 残差设备检查 | CPU/CUDA 各验证恒等、仅通道变化、仅尺寸变化、通道+尺寸变化；15×17 奇数输入通过 | 两份 `model-check.json` 的 `residual_checks` |
| 检查模式 | BN 缓冲值不变，无梯度产生；同一验证图片重复预处理逐位一致 | 两份 `model-check.json` |
| 语法 | `python -W error -m compileall -q src scripts tests`，退出码 0 | 开发中实际执行，未生成虚构日志 |

五类模型参数量 11,179,077。CPU/CUDA 的最终记录分别生成于北京时间 21:32:07，JSON 包含实际样本路径/标签、逐层形状、设备、预处理设置、训练 RGB 统计及源码 SHA-256。两设备处理的训练样本顺序和验证样本一致；未进行两设备 logits 逐位相等测试，不宣称此项。

CUDA 报告中 PyTorch 峰值 allocated memory 为约 109.965 MiB，只表示这次 eval/inference_mode 检查中该进程的张量分配，不是整机显存占用、速度基准或未来训练显存估计。

## 失败尝试及处理

核心 CPU/CUDA 单进程流程首次已通过。随后尝试 `num_workers=2` 时发生 WinError 5，失败日志为 `windows-workers.log`；独立最小复现 `multiprocessing.get_context('spawn').Queue()` 同样在 Windows 管道 `CreateFile` 失败，见 `multiprocessing-diagnostic.log`。DataLoader 构造未完成还触发了清理时的附带 AttributeError，不属于网络模型计算失败。

处理方式：最终 `make_loader` 固定 `num_workers=0`，删除多进程参数和未能验证的 worker 初始化路径，保留简单的随机种子控制。随后完整重跑 28 项检查及 CPU/CUDA 前向均通过，当前验收 JSON 对应修正后的源码。失败日志保留作过程证据，不把失败尝试列为成功能力。

首次结构检查 JSON 已移至被忽略的 `reports/local/v0.3-initial-checks`，避免将旧源码哈希对应的结果误当最终结果；v0.1/v0.2/v0.2.1 历史报告未改写。

## 复核命令

在项目根目录执行：

```powershell
& '.\.venv\Scripts\python.exe' scripts\verify_data.py
& '.\.venv\Scripts\python.exe' scripts\check_model.py --device cpu
& '.\.venv\Scripts\python.exe' scripts\check_model.py --device cuda
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

模型检查默认写 `reports/local`，不会覆盖本报告。CPU/CUDA 两次各取 train/val 一个批次；数据总清单已全量复核，但本版没有让全部数据通过模型。测试集图片使用数为 0，没有损失/优化器、训练成绩、模型文件、正式预测或视频。本阶段验收不等于整个考核项目完成。

本阶段源码/数据接入分两次实际提交，随后整理验收文档并创建 v0.3 标签；版本快照以标签为准，远端同步结果在成功读回后另作记录。下一阶段仅 v0.4 小规模训练试运行，等待用户确认。

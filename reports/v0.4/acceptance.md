# v0.4 验收：训练流程试运行

2026-10-15，使用原项目目录、独立环境、手写 ResNet-18 和 v0.2.1 数据。
本版已完成训练流程验收，**结果全部属于试运行，不是正式成绩**。

## 真实执行与证据

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 全量数据复核 | 3,670 张原图哈希/署名通过；可用 3,664 张，已知重复跨集 0 | verification.json、data-verification.log |
| 单进程加载判断 | 128 张增强图约 0.347 秒；继续使用 num_workers=0 | loading-assessment.json |
| 新训练核心检查 | 4 项通过后完成第一笔实现提交 e36fdde | training-unit-tests.log |
| 全部回归 | 原有 28 项 + 新增 4 项，共 32 项通过 | all-tests.log |
| 真实花卉训练 | CUDA、200 张训练/100 张验证、3 轮、39 个训练批次，分类层参数实际变化 | smoke-training.log、smoke/metrics.csv、smoke/summary.json |
| 配置及样本来源 | 种子 42、每类固定抽样、映射/数据/源码哈希记录齐全 | smoke/run.json、smoke/selected_samples.csv |
| 保存最佳模型 | 按验证损失最小保存第 3 轮，best.pt 为 44,797,707 字节 | smoke/summary.json；本地 artifacts/v0.4-smoke |
| 独立 CPU 重载预测 | PASS，参考 logits 最大绝对差约 8.05e-7 | reload-cpu/checkpoint-check.json、reload-cpu.log |
| 独立 CUDA 重载预测 | PASS，参考 logits 最大绝对差 0 | reload-cuda/checkpoint-check.json、reload-cuda.log |
| 历史产物保护 | 相同非空目录重跑返回退出码 1；随后核实旧 checkpoint SHA-256 不变 | overwrite-protection.log 及实际哈希复核 |
| 严格语法编译 | `python -W error -m compileall -q src scripts tests`，退出码 0 | 实际执行；不补造日志 |

模型 SHA-256 为 `f34fab2c2012eb9871c8baf58decbb96e397d90a6f0f18e474915de16196660b`，未加入 Git。模型获取和重跑步骤见 [训练说明](../../docs/training.md)。训练开始时 Git HEAD 为 e36fdde，工作区包含正在开发的入口，报告如实记为 dirty；对应源码与结果随后提交为 8767b01，源码哈希可独立核实，没有改造提交时间或过程。

## 结果解释

按最小验证损失选择的第 3 轮损失约 1.7123、验证准确率 44%；只针对这次 100 张验证子集。第 2 轮损失 10.9457 的波动完整保留，尚未用对照实验证明原因。用于重载检查的参考图片真实类别是 daisy，预测为 sunflowers；它的作用是核对保存前后输出一致，错误预测不能称作分类任务已合格。

训练三轮累计加载约 1.522 秒、计算约 1.642 秒，现有单进程已满足试运行。多进程可能降低加载等待，但当前没有值得修改系统权限的工作阻碍；没有宣称 Windows 多进程权限根因已修复。

## 本版边界与下一步

- 测试集图片读取数量为 0，不根据测试结果挑模型或调参；没有改变 v0.2.1 的划分、原图、统计或既有报告。
- 只有少量样本和 3 轮；没有全量正式训练、测试集评估、混淆矩阵、新图预测展示或视频。
- 保存模型可做推理，但不含优化器和随机状态，不能精确断点续训。
- 本次已实际验证 GPU 花卉训练、CPU/CUDA 重载；完整 CPU 花卉训练和其他机器尚未验证。
- 沿用已有第三方依赖；权重、原图、环境与凭据不进入 Git。源码、配置、说明、真实日志和样本清单进入仓库。

下一阶段为 v0.5 正式训练与评估，依据实际硬件选择配置、用验证集选择模型，再进行独立测试评估。本版验收后停止，等待用户确认。

# v0.4：小规模训练、验证与保存加载

本文保留 v0.4 的试运行配置和历史结果，用于学习基础训练流程。当前 v0.5 已完成正式训练与测试，见 [正式实验说明](experiments.md)；下文 32 项测试是 v0.4 的历史数量，当前完整测试为 38 项。

本版把真实图片经过手写 ResNet-18 的训练、验证、保存、重载和参考样本预测串联起来。所有产物都标记 `purpose=smoke`，这是检查代码能否完整工作的试运行，不是正式实验成绩。正式训练、测试集评估和曲线/混淆矩阵留给 v0.5；面向新图片的预测命令和展示留给 v0.6。

## 已有模型如何检查

本机已有 `artifacts/v0.4-smoke/best.pt`。在项目根目录执行：

```powershell
& '.\.venv\Scripts\python.exe' scripts\check_checkpoint.py --device cpu
& '.\.venv\Scripts\python.exe' scripts\check_checkpoint.py --device cuda
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

成功标志为 `CHECKPOINT CHECK PASSED` 和 `Ran 32 tests ... OK`。独立进程从磁盘恢复模型，用 checkpoint 内保存的预处理与类别映射重新预测参考图片，再与保存前的 logits 比较，容差为 `rtol=atol=1e-4`。GPU 模型可在 CPU 加载，已实际验证。

参考图片来自试运行的验证子集，真实类别为 daisy，本次预测为 sunflowers。预测错误不影响“保存和重载保持同一结果”的流程检查，但明确说明这个试运行模型不能作为完成花卉识别任务的成果。没有使用测试集图片，也没有挑选正确预测案例展示。

## 重新训练与模型获取

本机模型约 44.8 MB，按现有规则保留在 `artifacts`、不加入 Git。仓库保存源代码、配置、样本清单、日志与校验值；从 GitHub 新克隆仓库后，先按 README 配置环境并准备数据，再运行：

```powershell
& '.\.venv\Scripts\python.exe' scripts\train.py --device cuda
```

默认模型写到 `artifacts/v0.4-smoke`，报告写到 `reports/local/train-smoke`。本机已经有模型，重跑请改用新的空目录，避免覆盖已验收的历史：

```powershell
& '.\.venv\Scripts\python.exe' scripts\train.py --device cuda --run-dir artifacts\v0.4-smoke-rerun --report-dir reports\local\train-smoke-rerun
& '.\.venv\Scripts\python.exe' scripts\check_checkpoint.py --device cpu --checkpoint artifacts\v0.4-smoke-rerun\best.pt
```

成功标志为 `SMOKE TRAINING PASSED`，随后重载检查通过。再次重跑时需要继续选择未使用的目录名。入口拒绝覆盖非空模型/报告目录，已实际验证拒绝后原模型 SHA-256 不变。训练失败会保留 `failure.json` 和已生成日志，不能仅因存在 `best.pt` 就认定整次运行成功。

无 CUDA 的机器可指定 `--device cpu`；本版实际完成的是 GPU 花卉训练及 CPU/CUDA 重载，完整花卉训练的 CPU 耗时和跨机器复现仍未验证。可通过 `--epochs`、`--batch-size`、`--learning-rate`、`--seed`、`--train-per-class`、`--val-per-class` 覆盖配置；结果仍标记试运行，不会因此变成正式实验。

本机已生成模型的 SHA-256：

```text
f34fab2c2012eb9871c8baf58decbb96e397d90a6f0f18e474915de16196660b
```

这是本次文件的校验值，不保证重新训练后的文件字节相同。checkpoint 序列化、软件/硬件差异和训练数值都可能影响新文件；以新运行实际生成的 `summary.json` 为准。没有另行公开发布模型或上传原始图片。

## 本次配置及数据边界

| 参数 | 实际值与作用 |
| --- | --- |
| 数据版本 | `data/processed-v0.2.1`，训练/验证总池分别为 2,564/551 张 |
| 抽样 | 在原训练集中每类取 40 张、原验证集中每类取 20 张，共 200/100 张；按 seed+类别编号固定抽样 |
| 随机种子 | 42；固定 Python、NumPy、torch，DataLoader 使用独立生成器 |
| 输入与精度 | 224×224，普通 FP32，关闭 TF32，未使用混合精度 |
| epoch / batch | 3 轮 / 16 张；每轮 13 个训练批次，最后一批 8 张也保留 |
| 损失函数 | CrossEntropyLoss，接收原始 logits 和 int64 类别索引 |
| 优化器 | SGD，初始学习率 0.01、动量 0.9、权重衰减 0.0005 |
| 学习率调度 | StepLR，每 2 轮乘 0.1；三轮实际学习率 0.01、0.01、0.001 |
| 模型选择 | 只按当前验证子集的最小平均交叉熵保存 best.pt，未查看测试集 |
| 设备与加载 | RTX 5060 Laptop GPU，PyTorch 2.8.0+cu128，4 个 CPU 计算线程，num_workers=0 |

抽样只用于试运行，没有重新划分原数据，不依据模型结果挑样本。准确图片列表、类别、split、哈希与重复组保存在 `reports/v0.4/smoke/selected_samples.csv`；训练统计仍是 v0.2.1 完整训练子集的原始 RGB 像素统计，没有用这次验证子集重算均值。

## 训练循环为什么这样写

`src/engine.py` 的 `run_epoch` 根据是否传入 optimizer 区分训练/验证。训练时先 `model.train()`，然后每批执行：清除旧梯度 → 前向得到 logits → 交叉熵 → `loss.backward()` → `optimizer.step()`。如果不清除旧梯度，PyTorch 会累加梯度，更新含义就会改变。SGD 的学习率决定每步改变幅度，动量积累历史更新方向，权重衰减约束参数幅度；这些是当前初始选择，没有通过正式实验调优。

验证时调用 `model.eval()` 和 `torch.inference_mode()`，不反向传播、不更新参数，BN 使用训练获得的运行统计。本次每轮还检查验证前后 BN 缓冲值完全一致。损失按每批样本数加权求平均，正确预测数量按样本累计，避免最后一个小批次被赋予与大批次相同权重。日志中的训练指标来自训练过程中不断变化的模型，并不等同于用最终模型重新评估训练集。

学习率调度在整轮训练/验证后调用；CSV 记录的是这一轮实际使用的学习率。为了确认训练真的发生，还对比训练前后的分类层参数，最大绝对变化实测为约 0.07230。没有只跑前向却称为训练。

## 保存内容与独立重载

`src/checkpoint.py` 保存基础张量 `state_dict`，不序列化整个模型对象。参数先转成 CPU 张量，写入同目录唯一临时文件，完成后原子替换 `best.pt`。文件中包含模型架构、格式版本、类别映射、预处理和归一化参数、训练配置、最优轮数/验证损失、数据清单哈希、源码哈希与参考图片 logits。

加载时显式使用 `weights_only=True`，重新构建手写 ResNet18，再以 `strict=True` 校验所有参数键和尺寸。推理使用 checkpoint 中保存的类别/预处理，不从当前配置猜测；避免改了配置后解释错已有模型的输出。`best_metadata.json`、`config.json` 和 `class_to_idx.json` 另存本地，方便阅读。

本版只提供推理 checkpoint：没有保存优化器、学习率调度器和各随机源的状态，因此**不支持精确断点续训**。不把恢复模型权重称作完整恢复训练。

## 本次真实结果与局限

| 轮数 | 学习率 | 训练损失 | 训练准确率 | 验证损失 | 验证准确率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.01 | 1.6583 | 25.0% | 1.9290 | 24.0% |
| 2 | 0.01 | 1.5034 | 49.5% | 10.9457 | 30.0% |
| 3 | 0.001 | 1.4480 | 44.5% | 1.7123 | 44.0% |

表中所有数值都是 **200/100 张、3 轮的试运行结果**。第二轮验证损失明显增大；小样本、随机增强、参数更新及尚不稳定的 BN 运行统计都可能影响结果，但本次没有做对照实验来确认原因。没有隐去这一轮，也没有为得到好看的数字追加调参。正式训练需在 v0.5 进一步观察稳定性。

最佳轮为第 3 轮。独立 CPU 重载参考 logits 最大绝对差约 `8.05e-7`，CUDA 重载为 0，均小于约定容差。训练进程 PyTorch 峰值 allocated memory 约 504.38 MiB，不代表整机显存或以后正式配置的需求。本阶段不产生正式测试指标，不制作正式评估曲线或混淆矩阵。

## Windows 多进程加载是否需要修复

结论：**当前没有必要阻塞开发去修复。**多进程是可选吞吐优化，不是模型训练或考核的必要条件。此前最小复现确认 Python 多进程队列在当前受限环境的 Windows 管道 CreateFile 阶段返回 WinError 5；这不等于原图没有读取权限，也不能据此判断所有 Windows 终端都无法加载多进程。系统权限策略的最终根因尚未确认。

本次单进程短测加载 128 张增强图约 0.347 秒；真实三轮训练累计加载约 1.522 秒、计算约 1.642 秒。加载占这两项时间约 48%，说明未来存在优化空间，但本次绝对耗时很小，已经顺利完成阶段目标。计时在 GPU 运算后显式同步，是诊断记录，不是最大吞吐基准，不能直接线性推算其他机器或正式训练速度。

因此保留 `num_workers=0`，不修改系统权限。多进程是否能在普通本机终端恢复仍待单独验证；如果 v0.5 测量发现加载明显影响正式训练总耗时，再评估处理环境或采用并行加载的收益。

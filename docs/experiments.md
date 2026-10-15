# v0.5：正式训练与独立评估

本阶段使用同一手写 ResNet-18，从随机权重开始，完整训练 2,564 张图片，用 551 张验证图片选模型，选择固定后再评估 549 张测试图片。实验方案先提交为 `7034fa0`，见 [训练前方案](../reports/v0.5/protocol.md)。v0.4 的 200/100 张小规模记录继续保留，不替代正式实验。

## 实验配置与选择理由

配置在 `configs/train_formal.json`：30 轮，batch=32，SGD 初始学习率 0.01、动量 0.9、权重衰减 0.0005；StepLR 每 10 轮乘 0.2，实际学习率按 0.01 → 0.002 → 0.0004 三段运行；固定 seed=42。图像 224×224，沿用 v0.3 的随机裁剪/镜像训练增强和固定验证处理，归一化只使用 v0.2.1 的完整训练集原始 RGB 像素统计。

设备为 RTX 5060 Laptop GPU（约 8 GB 显存），PyTorch 2.8.0+cu128；普通 FP32、关闭 TF32、4 个 CPU 计算线程、DataLoader num_workers=0。batch=32 先通过 40/20 张一轮 smoke 容量检查，PyTorch 峰值 allocated memory 约 807.62 MiB。预检查不是正式成绩。

正式配置给初始学习率 10 轮学习时间，然后减小更新幅度，比 v0.4 每 2 轮下降更适合全量实验。只实施一组预先确定的配置，没有预训练、没有反复筛随机种子，也不宣称这组超参数最优。没有改变数据划分或根据预测结果删改图片。

每轮使用全部 train/val，损失按样本数加权，最后不足一批的图片保留。训练日志中的准确率来自该轮更新过程中的不同权重，而验证指标来自该轮结束的固定模型；两条曲线的统计条件不同，不能把两者差值简单当成严格泛化差距。

## 运行正式训练

以下命令从项目根目录执行，默认环境仍使用项目 `.venv`。本机正式产物已占用 `artifacts/v0.5-formal` 和 `reports/v0.5/formal`，重跑用新的空目录：

```powershell
& '.\.venv\Scripts\python.exe' scripts\train.py --config configs\train_formal.json --device cuda --run-dir artifacts\v0.5-rerun --report-dir reports\local\formal-rerun
```

成功标志 `FORMAL TRAINING PASSED`。配置中的 `purpose=formal` 强制使用完整 train/val；同时设置 per_class 抽样会被拒绝，避免只把 smoke 改名当正式结果。未显式指定正式配置时，旧 `train.py` 默认仍是 smoke。非空目录拒绝覆盖，重跑必须选择未使用的名字。

训练进程永远不构造 test Dataset。每轮只根据验证交叉熵是否降低来保存最佳模型；选择准确率最高轮或最后一轮可能得到不同模型，本项目事先固定为**最小验证损失**。不支持精确断点续训，失败要记录并用新目录重新开始，不能将不完整日志冒充完成。

## 固定模型后再测试

`freeze_model.py` 先核对正式训练已完成、日志轮数与样本数、验证损失最小轮、checkpoint SHA-256，再保存模型选择记录。记录中包括最佳轮次、验证指标、训练配置和训练报告 SHA-256；已有记录不能静默覆盖。

为新的重跑结果生成记录和曲线：

```powershell
& '.\.venv\Scripts\python.exe' scripts\plot_training.py --training-report reports\local\formal-rerun --output reports\local\formal-rerun-curves.png
& '.\.venv\Scripts\python.exe' scripts\freeze_model.py --checkpoint artifacts\v0.5-rerun\best.pt --training-report reports\local\formal-rerun --output reports\local\rerun-selection.json
& '.\.venv\Scripts\python.exe' scripts\evaluate.py --selection reports\local\rerun-selection.json --device cuda --output-dir reports\local\test-rerun
```

成功标志依次为 `TRAINING CURVES SAVED`、`MODEL SELECTION FROZEN`、`FORMAL EVALUATION PASSED`。这些命令用于复现预先固定的配置；已经看过测试结果后，不能利用它筛选更好的一次重跑或修改超参数。重新训练后文件哈希和数值不承诺跨机器完全一致。

本次实验的选模记录在测试评估前独立提交，从 Git 历史中能检查先后顺序。评估入口核对模型/训练报告未在选模后改变，并核对类别映射、预处理、归一化、数据清单哈希以及三个子集完整数量。测试阶段使用 eval/inference_mode，只前向、不更新参数或 BN。

收尾检查发现旧版本按原始文本字节校验报告，Windows CRLF 经 Git 转为 LF 后会误报。已修复：新选择记录明确标注 `canonical-json-v1`，解析 JSON 后按固定键序和编码求摘要；历史记录仅兼容 LF/CRLF 转换，数值变化仍被拒绝。模型二进制始终严格校验原字节。原 `model_selection.json` 与测试结果没有重写；测试报告中源码哈希对应测试时的 `98b4018` 源码，后续兼容修复另作真实提交，详见 [换行修复验证](../reports/v0.5/newline-compatibility.json)。

## 指标与输出含义

本次实际结果：选中第 21 轮，验证准确率 82.2142%；独立测试 446/549 正确，accuracy=81.2386%，macro-F1=0.808692，weighted-F1=0.811325。完整表格、曲线和错误分析见 [正式结果报告](../reports/v0.5/analysis.md)。这是一项单次正式实验，不代表多次运行平均或外部图片表现。

正式模型位于 `artifacts/v0.5-formal/best.pt`，44,797,643 字节；本次 SHA-256 为 `c836ddcb3053e88b8d7c92b51decdcea49f6f9adfc62a8aa6d4ffa7260de812b`。已有模型可运行 `.venv/Scripts/python.exe scripts/check_checkpoint.py --checkpoint artifacts/v0.5-formal/best.pt --device cpu` 检查，成功标志 `CHECKPOINT CHECK PASSED`。

`predictions.csv` 保存每一张测试图像的原始路径、真实标签、预测标签和五类概率。`confusion_matrix.csv` 的行是真实类别，列是预测类别；对角线是正确预测，非对角线显示混淆方向。PNG 图由相同矩阵绘制。

每类 precision 表示“预测为该类的图片里有多少正确”，recall 表示“真实该类被找回多少”，F1 是两者的调和平均。macro-F1 对五类等权，weighted-F1 按真实支持数加权；零分母明确定义为 0。计算只依赖 NumPy/Pandas，没有新增 sklearn 依赖；手算矩阵、零分母和不合法标签均有回归检查。

错误展示事先固定为预测概率最高的六个错误，用来发现模型自信但错误的情形，而不是随机样本或整体错误分布。每个例子并列展示原图和实际中心裁剪视图，配套 `image_attributions.csv` 包含摄影者、原始来源、CC BY 2.0 许可及变换说明。图像上的概率是 softmax 分数，未做置信度校准，不能解释为真实可靠性保证。

## 适用边界

这是一项固定种子、单数据集的实验，不是多次重复实验均值，也没有外部数据集验证。数据中可能仍有漏检的近重复、标签噪声和非典型花卉图像；单一中心裁剪可能遗漏边缘目标。错误分析只用于描述当前局限，测试评估后不调整本版训练配置、划分或标签。

正式训练中的加载耗时和显存以实际报告为准。本次保持单进程，未修复 Windows 多进程管道权限；后续若优化加载，应另记实际改动和验证，不能修改这次实验的历史配置。指标、日志和图表按真实结果提交；PDF 没有规定准确率门槛，不因此给出“保证达到生产应用效果”的结论。

模型保留在本机 `artifacts`，不上传原始图片、虚拟环境或大模型文件；仓库提供完整代码、依赖和重建步骤。推理 checkpoint 包含类别和预处理，但不包含精确续训所需的优化器/随机状态。单张新图片入口及展示体验由 v0.6 完善，最终视频与个人学习材料在 v1.0 整理。

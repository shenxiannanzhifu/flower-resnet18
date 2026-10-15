# 模型与运行产物目录

当前正式模型为 `v0.5-formal/best.pt`，44,797,643 字节，SHA-256：`c836ddcb3053e88b8d7c92b51decdcea49f6f9adfc62a8aa6d4ffa7260de812b`。完整训练 30 轮后按最小验证损失选第 21 轮，测试正确 446/549，准确率 81.2386%。该文件实际保存在本机，未上传 GitHub，没有公开下载地址。

在项目根目录检查正式模型：

```powershell
& '.\.venv\Scripts\python.exe' scripts\check_checkpoint.py --checkpoint artifacts\v0.5-formal\best.pt --device cpu
```

新克隆仓库先按 README 配置环境并准备数据，再生成正式模型：

```powershell
& '.\.venv\Scripts\python.exe' scripts\train.py --config configs\train_formal.json --device cuda --run-dir artifacts\v0.5-rerun --report-dir reports\local\formal-rerun
```

成功标志分别为 `CHECKPOINT CHECK PASSED`、`FORMAL TRAINING PASSED`。产物目录须为空；复现步骤、选模/测试命令和限制见 [正式实验说明](../docs/experiments.md)。重新训练不是获取完全相同的二进制文件，不保证跨机器指标完全一致。

## 保留的 v0.4 试运行模型

本机 v0.4 已生成 `v0.4-smoke/best.pt`，以及 `best_metadata.json`、`config.json`、`class_to_idx.json`。这是 200 张训练/100 张验证、3 轮试运行选出的模型，不是正式训练成果。

模型 44,797,707 字节，SHA-256：`f34fab2c2012eb9871c8baf58decbb96e397d90a6f0f18e474915de16196660b`。本目录除本说明以外默认不进入 Git，没有另行上传模型。

新克隆仓库须先配置环境和准备数据，在项目根目录运行 `.venv/Scripts/python.exe scripts/train.py --device cuda` 生成模型；已有产物时必须指定新的空 `--run-dir` 与 `--report-dir`。重新训练文件不保证字节级相同，以各次实际报告哈希为准。

检查已有模型：`.venv/Scripts/python.exe scripts/check_checkpoint.py --device cpu`。完整步骤、配置及限制见 [训练说明](../docs/training.md)。本版文件用于推理，未保存优化器状态，不支持精确断点续训。

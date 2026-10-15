# 模型与运行产物目录

本机 v0.4 已生成 `v0.4-smoke/best.pt`，以及 `best_metadata.json`、`config.json`、`class_to_idx.json`。这是 200 张训练/100 张验证、3 轮试运行选出的模型，不是正式训练成果。

模型 44,797,707 字节，SHA-256：`f34fab2c2012eb9871c8baf58decbb96e397d90a6f0f18e474915de16196660b`。本目录除本说明以外默认不进入 Git，没有另行上传模型。

新克隆仓库须先配置环境和准备数据，在项目根目录运行 `.venv/Scripts/python.exe scripts/train.py --device cuda` 生成模型；已有产物时必须指定新的空 `--run-dir` 与 `--report-dir`。重新训练文件不保证字节级相同，以各次实际报告哈希为准。

检查已有模型：`.venv/Scripts/python.exe scripts/check_checkpoint.py --device cpu`。完整步骤、配置及限制见 [训练说明](../docs/training.md)。本版文件用于推理，未保存优化器状态，不支持精确断点续训。

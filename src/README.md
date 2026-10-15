# 模型与数据代码目录

v0.3 已实现：`model.py` 手写 BasicBlock/ResNet18，`data.py` 按清单加载图片、增强、归一化和组批。

v0.4 已增加 `engine.py` 的训练/验证循环、`checkpoint.py` 的保存加载与参考预测检查；只有小规模试运行模型，尚无正式实验结果。

从项目根目录运行 `.venv/Scripts/python.exe scripts/check_checkpoint.py --device cpu` 可检查本机已有模型。首次训练用 `scripts/train.py --device cuda`；重跑需要指定新的空产物目录。完整说明见 `docs/training.md`。

旧入口 `scripts/check_model.py --device cpu` 仍只做随机模型的结构检查。网络及数据原理见 `docs/model.md`。

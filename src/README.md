# 模型与数据代码目录

v0.3 已实现：`model.py` 手写 BasicBlock/ResNet18，`data.py` 按清单加载图片、增强、归一化和组批。

从项目根目录运行 `.venv/Scripts/python.exe scripts/check_model.py --device cpu`。
模型及数据原理见 `docs/model.md`；v0.4 才增加训练流程。当前没有训练后的权重。

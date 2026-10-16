# 模型与数据代码目录

v0.3 已实现：`model.py` 手写 BasicBlock/ResNet18，`data.py` 按清单加载图片、增强、归一化和组批。

v0.4 已增加 `engine.py` 的训练/验证循环、`checkpoint.py` 的保存加载与参考预测检查。v0.5 已完成正式训练；`evaluation.py` 用 NumPy 计算矩阵和分类指标，并核对选模规则及训练报告。旧模型及报告继续保留。

从项目根目录运行 `.venv/Scripts/python.exe scripts/check_checkpoint.py --checkpoint artifacts/v0.5-formal/best.pt --device cpu` 检查正式模型。正式训练须显式指定 `scripts/train.py --config configs/train_formal.json --device cuda`，默认配置仍是 smoke；重跑需要指定新的空产物目录。正式实验与完整重跑命令见 [实验说明](../docs/experiments.md)，基础训练原理见 [训练说明](../docs/training.md)。

旧入口 `scripts/check_model.py --device cpu` 仍只做随机模型的结构检查。网络及数据原理见 `docs/model.md`。

v0.6 增加 `inference.py`：从正式 checkpoint 恢复预处理和类别顺序，读取任意单张静态图片并预测，不依赖原数据目录。运行 `.venv/Scripts/python.exe scripts/predict.py --image examples/sunflower.jpg`，查看中文 Top-K 和生成的 PNG。使用方法与边界见 [新图预测说明](../docs/inference.md)。

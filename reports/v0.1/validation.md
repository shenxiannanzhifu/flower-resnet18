# v0.1 真实验证记录

日期：2026-10-14，Asia/Shanghai。以下检查在本项目 `.venv` 中执行。环境报告内有实际运行时间与设备采样；没有训练指标。

工作目录：

```text
C:\Users\Cipher\Documents\Codex\2026-10-14\pdf-ai-git-1-pdf-pdf\outputs\flower-resnet18
```

| 实际命令/检查 | 结果 | 证据 |
| --- | --- | --- |
| `.\.venv\Scripts\python.exe scripts\check_env.py --device cuda --output-dir reports\v0.1` | 退出码 0，六项 PASS | environment.json、environment-check.log |
| `.\.venv\Scripts\python.exe scripts\check_env.py --device cpu` | 退出码 0，五项 PASS | cpu-check.log；此命令另写入被忽略的 reports/local |
| `.\.venv\Scripts\python.exe -m pip check` | 退出码 0，No broken requirements found. | pip-check.log |
| `.\.venv\Scripts\python.exe -W error -m py_compile scripts/check_env.py` | 退出码 0，无语法错误或警告 | 开发阶段实际执行；未单独生成编译日志 |
| 查看依赖检查图；Pillow 重开生成文件 | 图表清晰，780×520 像素 | dependency_smoke.png；检查程序核对尺寸 |
| `git diff --check` 与 `git diff --cached --check` | 无空白错误 | 本地实际检查 |
| `git check-ignore` 检查环境、数据、模型、本地报告、.env 示例路径 | 均被忽略 | .gitignore；未创建任何真实密钥 |

CUDA 检查不是只查询显卡名称：已执行矩阵乘法并核对数值，反向传播并核对梯度，执行卷积前向/反向，检查 `[2,8,16,16]` 输出形状和有限非零梯度，最后同步设备。PyTorch 构建包含 `sm_120`，本机 RTX 5060 Laptop GPU 实测通过。

人工检查数据：NumPy 构造 `[0,1,...,11]` 并变形为 4×3，行均值为 `[1,4,7,10]`；Pandas 将前两行与后两行分别分组，均值为 A=2.5、B=8.5；Matplotlib 将该结果绘成柱状图。**这些值不是花卉数据统计，也不是模型实验成绩。**

完整锁定依赖由 `python -m pip freeze` 实际生成，见项目根目录 requirements-lock.txt；不含本地文件 URL、账号或认证信息。

尚未验证：真实数据及其许可、手写网络、训练/验证/保存/预测、正式评估、异机重建环境、远程仓库、运行视频。下一阶段必须等待用户确认。

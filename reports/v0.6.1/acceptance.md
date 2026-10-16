# v0.6.1：整图输入与未知类别问题的实测判断

检查日期：2026-10-16。接续 v0.6，不重训、不改划分或原测试成绩。比较前先提交方案为 `637afaa`，实现与针对性检查提交为 `d27ca0e`，之后才运行比较。

## 实现与采用结论

`predict.py --view full` 等比例缩放完整图片，正常情况下短边 256，长边最多 768；不裁剪、不补边。沿用 checkpoint 的 RGB 归一化与类别映射，模型内已有自适应池化可接受非正方形输入。缩放后短边不足 32 的极端长宽比会被明确拒绝。

这解决了**整图模式中因中心裁剪而丢失边缘范围**的问题，不保证保留缩小后的细微纹理，更不保证模型关注正确的花朵。显示图和 JSON 记录实际输入方式及尺寸，原 checkpoint 配置不被伪装成已改变。

| 同一 551 张验证图 | 正确数 | 准确率 | 交叉熵 |
| --- | ---: | ---: | ---: |
| 中心裁剪 | 453 | 82.2142% | 0.500067 |
| 整图输入 | 446 | 80.9437% | 0.535928 |

整图准确率下降 1.2704 个百分点，超过事先固定的 1 个百分点容忍范围（同时要求 loss 增加不超过 0.05）。因此**默认仍为 center，full 仅供明确选择**。不使用测试集筛选模式；本次 test_images_read=0。验证集此前已参与选模，这次比较属于开发验证，不是独立泛化成绩。

## 未知类别没有被可靠解决

按两种模式各自验证最高分的第 5 百分位设置诊断阈值，保留约 94.92% 已知验证图。阈值分别为 0.474516 和 0.452727。八张固定非花卉合成图（黑、白、灰、红、绿、蓝、seed=42 噪声、棋盘）在两种模式下均有 **8/8 通过阈值**。

这些是“高分仍可能不是花”的具体反例，不是未知类别基准或未知准确率。没有将阈值加入用户预测入口，也没有加上缺乏证据的“未知识别”承诺。要可靠评估拒识需要有代表性的未知数据、独立阈值选择/评估集与明确误拒/漏拒代价，超出本次有限修正。用户允许不能解决时直接进入下一版本，因此继续 v1.0 材料准备。

## 已实际验证

- 9 项推理针对性测试、47 项完整回归通过；严格 SyntaxWarning 编译通过。测试包括宽/高图四边保留、最大尺寸、归一化、非法模式和极端比例拒绝。
- 正式 checkpoint 完整比较 551 张验证图的两种输入；保存 1,102 行逐图结果、比较 JSON、8 张合成探针记录和几何示意图。
- 外部示例 CPU/CUDA 整图预测均为向日葵，分数 0.824018；两设备 logits 最大差 1.073e-6。默认中心裁剪分数 0.969357，与 v0.6 CPU logits 完全一致。单图分数不代表准确率。
- 整图/中心示例图、几何图与探针图已目视检查；示例展示图沿用 Till Westermayer 的 CC BY-SA 2.0，来源与许可在各预测 JSON 和 examples/attribution.json。
- 正式模型和数据清单 SHA-256 未变；v0.1–v0.6 报告、模型/数据/保存模块、配置相对 a071b44 无改动。具体数值在 consistency.json。

复现命令（项目根目录；比较输出目录必须未被使用）：

```powershell
& '.\.venv\Scripts\python.exe' scripts\predict.py --image examples\sunflower.jpg --view full
& '.\.venv\Scripts\python.exe' scripts\compare_input_views.py --device cuda --output-dir reports\local\view-comparison
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

对应成功标志为 PREDICTION PASSED、INPUT VIEW COMPARISON PASSED、Ran 47 tests / OK。比较工具需原始数据及清单；单张预测只需图片和正式模型。

## 证据

方案 protocol.md；比较 comparison/comparison.json、validation_predictions.csv、unknown_probes.csv；演示 full-cpu、full-cuda、default-center；检查 all-tests.log、compileall.log、consistency.json。几何示意中的中心裁剪丢失左右边界，整图保留四边。

补丁验收通过代表可选功能与诊断过程通过，不表示未知类别问题已解决或模型精度提高。v1.0 最终交付仍需真实视频、用户个人感悟与理解确认。

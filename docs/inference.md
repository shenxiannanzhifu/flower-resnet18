# 单张新图片预测（v0.6 / v0.6.1）

使用 v0.5 已训练好的手写 ResNet-18，对文件路径指定的单张图片分类，输出中文 Top-K、JSON 和展示图。v0.6.1 增加可选整图输入，默认仍保持旧中心裁剪；没有重新训练或改写 v0.5 测试结果。

## 本机直接运行

独立环境和正式模型已经存在，无需重新安装或训练。在 PowerShell 运行：

```powershell
Set-Location -LiteralPath 'C:\Users\Cipher\Documents\Codex\2026-10-14\pdf-ai-git-1-pdf-pdf\outputs\flower-resnet18'
& '.\.venv\Scripts\python.exe' scripts\predict.py --image examples\sunflower.jpg
```

成功标志为 `PREDICTION PASSED`，终端显示中文类别、分数、结果 JSON 路径和 PNG 路径。默认 CPU、Top-3，默认 checkpoint 是项目内 `artifacts/v0.5-formal/best.pt`。输出每次写入新的 `reports/local/predictions/时间戳/`，可以重复运行。

打开最近一次默认输出：

```powershell
$predictionFolder = Get-ChildItem -LiteralPath '.\reports\local\predictions' -Directory | Sort-Object Name -Descending | Select-Object -First 1
Get-Content -LiteralPath (Join-Path $predictionFolder.FullName 'prediction.json') -Encoding utf8
Start-Process -FilePath (Join-Path $predictionFolder.FullName 'prediction.png')
```

给自己的图片分类时，把下方示例路径替换为真实文件路径；含空格的路径保留引号：

```powershell
& '.\.venv\Scripts\python.exe' scripts\predict.py --image 'C:\你的图片\花朵.jpg'
```

可选择 CUDA、显示全部五类或指定新的空目录：

```powershell
& '.\.venv\Scripts\python.exe' scripts\predict.py --image examples\sunflower.jpg --device cuda --top-k 5 --output-dir reports\local\my-prediction
```

第二次使用同一非空输出目录会明确拒绝。日常直接省略 `--output-dir` 即可；验收报告目录不要覆盖。显式提供的图片、checkpoint 和输出相对路径均相对当前命令目录；默认模型和默认结果目录则固定在项目内。

## 可选整图输入

花朵靠近边缘时，可以显式选择整图：

```powershell
& '.\.venv\Scripts\python.exe' scripts\predict.py --image examples\sunflower.jpg --view full
```

full 等比例缩放完整图片，不裁剪或补边，短边目标来自 checkpoint（当前 256），最长边限制为 768；极端长宽比导致短边不足 32 时拒绝。模型的自适应池化接受可变 H/W，但整图不保证优于中心裁剪，也不能保证缩小后的小花仍可识别。

同一 551 张验证图实测：center 82.2142%、full 80.9437%，full 未通过事先约定的采用条件，所以默认不变。此比较属于开发验证，不是独立测试。示例图两种模式均为向日葵，分数分别约 96.94%/82.40%；分数不能用来证明哪种模式准确率更高。

JSON 的 preprocessing 保留 checkpoint 原配置，actual_preprocessing 记录本次实际模式，model_input_size 为 [宽,高]，input_shape 为 [1,3,高,宽]。旧 crop_size 字段仅兼容旧读取程序；full 下它不表示发生裁剪。诊断与证据见 [补丁验收](../reports/v0.6.1/acceptance.md)。

## 新克隆仓库如何准备模型

GitHub 保存代码、依赖清单、带许可的演示缩图和真实报告；不包含虚拟环境、训练数据或约 44.8 MB 的 checkpoint，没有模型下载地址。

先按 [README 环境配置](../README.md#环境与依赖) 建立 Python 3.12 独立环境并安装依赖；然后从项目根目录运行：

```powershell
& '.\.venv\Scripts\python.exe' scripts\download_data.py
& '.\.venv\Scripts\python.exe' scripts\prepare_data.py
& '.\.venv\Scripts\python.exe' scripts\verify_data.py
& '.\.venv\Scripts\python.exe' scripts\train.py --config configs\train_formal.json --device cuda --run-dir artifacts\v0.5-formal --report-dir reports\local\formal-first-run
& '.\.venv\Scripts\python.exe' scripts\predict.py --image examples\sunflower.jpg
```

这些训练目录用于**首次克隆且尚无模型**的情况。本机已有相同目录，请勿重复覆盖。已有其他正式模型则使用 `--checkpoint '实际模型路径'`；仅需推理时，不必保留原训练数据、数据清单或统计 JSON，预处理和类别信息已保存在模型中。无 GPU 的电脑可以用 CPU 推理；完整 CPU 训练耗时未验证。

重新训练不保证模型文件哈希或跨机器指标完全一致，训练/选模方法见 [正式实验说明](experiments.md)。v0.6 只使用已固定的正式模型，不根据新图结果重新调参。

## 如何理解结果

图中左侧为 EXIF 方向修正后的 RGB 原图，中间是**真正送给模型的图像（按所选模式）**，右侧为 Top-K 分数。图中文字使用英文以减少字体兼容问题，终端和 JSON 提供中文类别。

| 编号 | 类别名 | 中文 |
| --- | --- | --- |
| 0 | daisy | 雏菊 |
| 1 | dandelion | 蒲公英 |
| 2 | roses | 玫瑰 |
| 3 | sunflowers | 向日葵 |
| 4 | tulips | 郁金香 |

上表解释当前模型；代码实际按 checkpoint 的编号查类别，不依赖字典键顺序。JSON 记录全部五类分数、logits、Top-K、图像/模型 SHA-256、输入尺寸、类别映射、预处理、设备和源码摘要。它不要求输入真实标签，也不为单张图片计算准确率。

公开示例在 CPU/CUDA 上均预测为向日葵，softmax 分数约 96.94%；这是一次演示，不是“新图准确率 96.94%”。v0.5 的独立测试准确率仍是 446/549=81.2386%。模型只能在五类中选择，非花卉、其他花种也会被归入其中一类；没有训练“未知类”检测器或进行概率校准。

外部图片在任何推理前选定，来源为 Till Westermayer 的 2024 年照片，许可 **CC BY-SA 2.0**。图片及含图的展示产物按相同许可提供，署名、来源、许可链接和改动记录见 [示例许可](../examples/README.md) 与 [来源记录](../examples/attribution.json)。示例未进入原有 train/val/test。任意用户图片不会被自动套用示例署名；默认预测产物位于 Git 忽略目录，不会自动上传。

## 常见提示

| 提示 | 处理方式 |
| --- | --- |
| 图片不存在或不是文件 | 检查路径与扩展名，含空格的路径加引号，不能传文件夹 |
| 图片损坏或格式不受支持 | 用看图软件检查，必要时另存为 JPEG/PNG；不能只修改扩展名 |
| 请提供单张静态图片 | 将动画或多页文件另存为一张静态图 |
| 模型文件不存在 | 本机使用默认正式模型；新克隆按上方步骤训练，或用 --checkpoint 指定已有模型 |
| 模型无法加载 | 使用本项目生成的完整 checkpoint，不传图片、JSON 或不匹配的权重 |
| 这是 smoke 试运行模型 | 改用正式模型；v0.4 模型仍可通过旧 check_checkpoint.py 检查 |
| top-k 范围不对 | 当前五类模型使用 1–5 |
| 输出目录已存在且非空 | 省略 --output-dir 自动生成新目录，或换一个空目录 |
| CUDA 不可用 | 使用 --device cpu |

灰度/RGBA 图片统一转 RGB，RGBA 的透明通道按既有处理规则丢弃；EXIF 方向先修正再处理。默认中心裁剪仍可能遗漏边缘花朵；full 避免几何裁剪，但有背景干扰和缩小细节的代价。图片过大时保留 Pillow 的像素限制并给出缩小图片的提示，没有关闭保护。

未知类别尚不能可靠识别：本次八张非花卉合成图全部通过了保留约 95% 已知图的诊断分数阈值。高 softmax 分数不能证明属于五类花卉，因此入口没有部署该阈值或声称“自动识别未知”。八个反例也不等于全面未知数据集评估。

## 建议阅读的代码

`src/inference.py` 负责图片解码、正式模型检查和单图预测；`scripts/predict.py` 负责命令行、结果文件和图表；`tests/test_inference.py` 用已知像素和分数检查关键行为。

理解三个重点：预处理与类别映射必须随模型一起恢复；`eval()` 固定 BatchNorm 而 `inference_mode()` 关闭梯度；softmax 排名与真实分类准确率的区别。实际阶段证据见 [v0.6 验收报告](../reports/v0.6/acceptance.md)。

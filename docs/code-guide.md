# 代码阅读与理解练习

本说明帮助本人解释和修改项目，不代表已经完成理解验收。先运行预测再读相应函数，比按文件名顺序通读更容易建立联系。所有练习使用 reports/local 下的新输出，保留正式实验。

## 阅读路线

| 顺序 | 文件/函数 | 要理解的输入与输出 |
| --- | --- | --- |
| 1 | scripts/predict.py: main | 图片/模型路径 → 中文排名、JSON、展示图 |
| 2 | src/inference.py: read_image、prepare_input、predict_image | 静态图 → RGB → NCHW → 五类 logits/分数 |
| 3 | src/model.py: BasicBlock、ResNet18 | [N,3,H,W] → [N,5]，看两支路相加 |
| 4 | src/data.py: FlowerDataset、build_transform、make_loader | CSV 的某个划分 → 图片张量、标签和路径 |
| 5 | src/engine.py: run_epoch；scripts/train.py | 训练批次/优化器 → 一轮指标，验证损失决定保存 |
| 6 | src/checkpoint.py；src/evaluation.py | 参数+元数据 → 可复用模型；真实标签+预测 → 指标 |
| 7 | scripts/data_utils.py、prepare_data.py | 原图与重复关系 → 分组划分、统计、图表 |

## 三个核心问题

### 为什么残差块要有两条支路？

主支路学习 F(x)，跳跃支路直接传递信息或投影，输出为 ReLU(F(x)+shortcut(x))。相加前 N、C、H、W 必须一致。例如 [N,64,56,56] 到 [N,128,28,28] 时，主支路第一层 stride=2；跳跃支路用 1×1、stride=2 将通道改为 128、空间减半。

18 层按首个卷积、八块各两卷积和一个全连接层计数；不把 BatchNorm、ReLU 或投影卷积加入惯例计数。AdaptiveAvgPool2d(1) 把任意有效 H/W 聚成一个空间位置，所以整图模式仍能输出五类。结构允许变尺寸不等于对任意构图都同样准确。

### 训练、验证、测试分别做什么？

训练从增强图片算 logits，CrossEntropyLoss 内部处理稳定的 log-softmax，所以不要先对模型输出做 softmax 再送入此损失。zero_grad 清除旧梯度，backward 计算梯度，optimizer.step 真正更新权重。

验证用于选择模型，不能更新权重或 BatchNorm 统计；eval 改模块行为，inference_mode 关闭梯度，两者不同。测试在模型选择固定后估计表现，不能再用它筛超参数。本项目先提交选择记录再测试；v0.6.1 的输入比较只用验证集，不能冒充新的独立测试。

重复图像必须成组分到同一个子集，否则模型可能记住近似图片而产生偏高成绩。归一化统计只来自训练集，推理从 checkpoint 恢复，不用一张新图片重新算统计。

### 为什么高分不能证明模型认识这朵花？

softmax 只在五个输出之间分配总和为 1 的分数，没有自动新增“未知”项。非花卉也可能有很高分。准确率需要一批有真实标签的数据才能计算；单图 96.94% 是分数，正式测试 81.24% 才是对应测试集准确率。

把低分都拒绝会丢掉一部分已知花，高分未知又可能漏过。本项目用八张合成反例发现简单阈值不可靠，因此没有把它包装成未知检测功能。

## 实际动手练习

以下从项目根目录执行。无需重新训练，也无需修改正式文件。

```powershell
& '.\.venv\Scripts\python.exe' scripts\predict.py --image examples\sunflower.jpg --top-k 5
& '.\.venv\Scripts\python.exe' scripts\predict.py --image examples\sunflower.jpg --view full
& '.\.venv\Scripts\python.exe' scripts\predict.py --image examples\this-file-does-not-exist.jpg
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -p test_model.py -v
```

前两条应成功；比较显示图中输入范围和 JSON 的 actual_preprocessing/input_shape。第三条**预期失败**，检查提示与非零退出码；不要把演示失败当成项目故障。最后一条验证残差结构边界；阅读 tests/test_model.py，尝试用自己的话解释恒等支路和投影检查。

可选代码练习：在自己的未发布练习分支里改变 Top-K 默认值，检查命令帮助与结果，然后决定是否保留为新提交。不要改已有标签或以 reset --hard 清除不理解的改动。改变 num_classes 并不能让旧五类模型识别新类；还需要新数据、映射和重新训练。

## 其他应能解释的问题

- 为什么用 venv？隔离依赖；它不是可随意复制到另一机器的 Python 安装。实际版本在 requirements-lock.txt。
- NumPy 和 Pandas 是否只安装没用？指出 RGB 统计、groupby 计数、混淆矩阵代码与对应报告。
- 最后一批较小时为什么损失乘样本数？避免直接平均批次均值导致小批次权重过大。
- 为什么 checkpoint 还要保存类别和预处理？相同 logits 编号若映射错会显示错类别，归一化不一致也会改变输出。
- 迁移学习是什么？利用预训练权重，冻结部分层或微调；本项目只学习概念，实际模型随机初始化，手写要求未被替代。
- Windows num_workers=0 是否影响正确性？它是已验证的单进程加载方式，影响速度；本次正式训练已经完成。多进程管道权限根因尚未修复。

请在 [项目说明的个人感悟](project-report.md#ai-辅助范围与个人感悟本人待填写) 写真实理解与疑问。未亲自运行的内容不要写成亲身经历。

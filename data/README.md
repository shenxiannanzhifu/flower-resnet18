# 数据目录

v0.1 只确定数据来源，尚未下载、检查或划分数据。

拟采用 TensorFlow 示例花卉图片集（tf_flowers / flower_photos）：

- 官方目录：https://www.tensorflow.org/datasets/catalog/tf_flowers
- 下载地址：https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
- 官方目录列出 3,670 张图片、5 个类别，下载包 218.21 MiB。
- 初步类别：daisy、dandelion、roses、sunflowers、tulips。
- 2026-10-14 已实际访问目录与下载地址（HEAD 200，228,813,984 字节），没有下载图片包。

这些数字来自官方目录，**不是本项目的实测统计**。v0.2 会检查图片、标签、重复样本与损坏文件，并记录实际结果。

使用条件：公开可下载不等于没有限制。v0.2 必须读取压缩包中的 `LICENSE.txt` 和逐图来源/署名信息，记录图片许可与再分发条件。官网页面底部的文档许可不能直接当作图片许可。在完成核查前，不将图片、压缩包或示例拼图上传远程仓库。

v0.2 的划分原则：先查重，再按类别分层；同一重复样本组只进入一个子集；固定随机种子。精确比例根据实际清洗结果确定。验证集用于选择模型，测试集在正式评估时使用。

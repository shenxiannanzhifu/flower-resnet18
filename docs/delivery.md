# v1.0 最终交付与提交步骤

项目技术成果、本人审阅的感悟及定稿运行视频已经齐备。最终包在本地制作；仓库持续保存源码、依赖、文档和真实开发历史。最后向考核接收人提交由本人执行，AI 不自动发送成果。

## 最终包在哪里

原项目路径：

```text
C:\Users\Cipher\Documents\Codex\2026-10-14\pdf-ai-git-1-pdf-pdf\outputs\flower-resnet18
```

最终 ZIP：项目下 `artifacts/delivery-v1.0/flower-resnet18-v1.0-delivery.zip`。

它是交付快照，开发仍只在原项目仓库进行。原 `artifacts/delivery-preparation/20261016-preparation` 是旧准备包，继续保留，但不要将它当作最终交付。最终包的来源提交、文件清单和 SHA-256 在随包 manifest.json / checksums.sha256；ZIP 本身摘要在旁边同名 .sha256 文件。

## 解压后有什么

```text
flower-resnet18-v1.0/
├── README-DELIVERY.md          # 接收者先阅读
├── repository-url.txt         # 原 GitHub 仓库与真实历史
├── PROJECT-REPORT.html        # 双击可读的完整说明、感悟与图表
├── manifest.json              # 源提交、文件大小/哈希与核验范围
├── checksums.sha256            # 包内文件校验清单
├── delivery-verification.json # 归档、模型/视频及解包运行核验
├── flower-resnet18/            # 从真实 Git 提交导出的源码、文档与报告
├── model/best.pt               # 原正式模型，约 44.8 MB
└── video/demo.mp4              # 本人定稿原视频，约 281.9 MB
```

源码目录含 README、requirements*.txt、src/scripts/configs/tests、数据分析与实验报告、示例及许可。项目说明原文为 flower-resnet18/docs/project-report.md，HTML 由该原文生成，便于接收者直接阅读；相对图片和链接指向包内对应文件。

不会把 .git、.venv、原训练图片或下载缓存放入最终包；Git 历史请在原仓库查看。模型和视频属于本地交付材料，均不上传 GitHub，也没有创建公开模型下载链接。

## 接收者如何运行

先把整个 ZIP 解压，在 flower-resnet18 子目录按 README 建立 Python 3.12 环境，再运行：

```powershell
& '.\.venv\Scripts\python.exe' scripts\predict.py --checkpoint ..\model\best.pt --image examples\sunflower.jpg --device cpu
```

成功标志 PREDICTION PASSED；结果 JSON/PNG 保存到 reports/local/predictions 的新目录。整图模式可加 --view full，但不保证更准确。只预测时不需要原训练数据。

也可将单独模型复制到子项目 artifacts/v0.5-formal/best.pt，使用 README 的默认命令。若只克隆 GitHub、未收到交付包，按 [模型获取说明](inference.md#新克隆仓库如何准备模型) 准备数据并重新训练。旧 check_checkpoint.py 需要训练数据中的参考图，不能替代此处无数据的新图入口。

实际打包预验收已在不含 .venv/原始数据的快照中，借用本机已有环境，以包内相对路径加载 model/best.pt 完成两种模式预测，结果与原项目一致。这不等于另一台机器重新安装环境成功；跨机器安装尚未实测。

正式模型 SHA-256：

```text
c836ddcb3053e88b8d7c92b51decdcea49f6f9adfc62a8aa6d4ffa7260de812b
```

定稿视频 SHA-256：

```text
c01bd8eba8b2fc41d77d31cb8aed73336f6e036a0411c573fbac149fc69ad353
```

## 本人接下来怎样提交

1. 打开最终 ZIP 旁的同名 .sha256 文件；需要检查传输完整性时，运行 `Get-FileHash -Algorithm SHA256 'ZIP 的完整路径'` 并核对。
2. 解压包，打开 PROJECT-REPORT.html、video/demo.mp4，确认拿到的是最终说明和已选定的视频。
3. 按考核指定渠道发送最终 ZIP，并附上仓库链接：https://github.com/shenxiannanzhifu/flower-resnet18 。包内已有说明、结果、模型、视频及仓库链接。
4. 若渠道限制文件大小，使用该渠道支持的大文件传输方式；不必为此重新训练或把视频强行加入 Git。接收人和发送范围由本人确认。

可用提交说明：

> 后端 AI 方向第一阶段考核项目：基于手写 ResNet-18 的五类花卉分类。附件包含源代码、依赖说明、数据分析与正式实验结果、项目说明与个人感悟、正式模型及运行视频。仓库：https://github.com/shenxiannanzhifu/flower-resnet18 ，版本 v1.0。复现和预测步骤见包内 README-DELIVERY.md。

## 验收范围与已知局限

本人已审阅修改感悟并要求提交；确认完成阅读和运行；提供视频并指定定稿。自动检查和视频展示不等于教师评分，也不替代本人继续学习。

正式测试保持 446/549、81.2386%、macro-F1 0.808692。默认中心裁剪可能遗漏边缘；整图可选输入保留范围但验证稍降；未知类别识别未可靠解决；Windows 多进程权限和原生 Git HTTPS 崩溃根因未永久修复。上述局限不阻碍已核对的 PDF 核心项目要求，不伪装成已解决。

视频完整音视频解码及关键帧视觉检查已通过，未逐字听审旁白。图像来源与 CC BY/CC BY-SA 署名随包保留。最终验收证据在 [reports/v1.0](../reports/v1.0/acceptance.md)，打包和远端核验以随包记录为准。**尚未向考核接收人发送；这是最后一项必须由本人完成的操作。**

# 运行视频录制脚本（准备材料，尚未录制）

PDF 要求运行项目并展示主要结果，无时长限制。建议按下面顺序录制约 6–10 分钟，时长只是建议。本人实际操作、按自己理解讲解，不需逐字朗读。

录屏可使用 Windows 截图工具的录制功能或已有录屏软件；打开要录的终端、代码与图表，收起登录/个人消息窗口。保存为本机 MP4，观看一次确认画面、声音和结果可辨认。不会自动上传或发送。

## 录制前

本机无需重新安装环境和训练。PowerShell 先执行：

```powershell
Set-Location -LiteralPath 'C:\Users\Cipher\Documents\Codex\2026-10-14\pdf-ai-git-1-pdf-pdf\outputs\flower-resnet18'
& '.\.venv\Scripts\python.exe' scripts\predict.py --help
```

成功看到 --image、--checkpoint、--device、--view 等参数。后面省略输出目录的预测可重复执行；已有的正式报告不覆盖。本人先阅读 [代码说明](code-guide.md)，把不理解的内容列出来。

## 镜头与操作

| 镜头 | 屏幕与操作 | 应解释的事实 |
| --- | --- | --- |
| 1：题目 | README、仓库 main 与标签；运行下方 Git 命令 | 五类花卉，分版本实际开发，AI 辅助范围；最终状态如实说明 |
| 2：环境 | 运行下方环境命令 | Python 3.12 venv，PyTorch/NumPy/Pandas/Matplotlib，GPU 检查真实通过 |
| 3：数据 | 打开 reports/v0.2.1 的类别分布和数据说明 | 原 3,670，保留 3,664；三划分 2564/551/549，重复组不跨集，训练统计不读验证/测试 |
| 4：模型 | src/model.py 的 BasicBlock 和四组层 | F(x)+shortcut(x)，维度不同用投影，随机初始化手写 ResNet-18 |
| 5：训练与评估 | 查看真实日志、曲线、混淆矩阵 | 30 轮，按验证 loss 选第 21 轮；测试 446/549、81.24%、macro-F1 0.8087 |
| 6：现场预测 | 实际运行默认预测，打开刚生成的 JSON/PNG | 新图读取、模型加载、实际输入、中文 Top-K；分数不等于准确率 |
| 7：整图与错误提示 | 实际运行 full 和不存在文件命令 | 整图可保留边缘但验证稍降；模型不能可靠识别未知；缺失文件会明确提示 |
| 8：理解与总结 | 项目说明中的本人感悟、交付清单 | 只讲自己真实运行/理解的内容；承认局限和仍待改进处 |

Git 历史与环境命令：

```powershell
git log --oneline --decorate -15
git tag --list
& '.\.venv\Scripts\python.exe' scripts\check_env.py --device cuda --output-dir reports\local\video-env
```

环境成功标志 ENV CHECK PASSED。video-env 只写本地检查产物，可重复检查；不要换成 reports/v0.1 覆盖原证据。没有 CUDA 的其他机器改 --device cpu 并如实说明设备。

展示真实训练日志和图表（不必在视频中重新等待训练）：

```powershell
Import-Csv -LiteralPath '.\reports\v0.5\formal\metrics.csv' | Format-Table epoch,learning_rate,train_loss,val_loss,val_accuracy
Get-Content -LiteralPath '.\reports\v0.5\test\metrics.json' -Encoding utf8
Start-Process -FilePath '.\reports\v0.5\training_curves.png'
Start-Process -FilePath '.\reports\v0.5\test\confusion_matrix.png'
```

讲清楚这是之前实际训练保存的日志，不声称录屏时重新训练。若展示错误图，保留图上署名和旁边的 image_attributions.csv；不要为了好看只留下正确案例。

现场新图预测及查看：

```powershell
& '.\.venv\Scripts\python.exe' scripts\predict.py --image examples\sunflower.jpg --top-k 5
$predictionFolder = Get-ChildItem -LiteralPath '.\reports\local\predictions' -Directory | Sort-Object Name -Descending | Select-Object -First 1
Get-Content -LiteralPath (Join-Path $predictionFolder.FullName 'prediction.json') -Encoding utf8
Start-Process -FilePath (Join-Path $predictionFolder.FullName 'prediction.png')
```

出现 PREDICTION PASSED。默认示例为向日葵，分数约 96.94%；图中显示实际中心裁剪。图片由 Till Westermayer 摄影，CC BY-SA 2.0，保留图上署名。

整图与异常演示：

```powershell
& '.\.venv\Scripts\python.exe' scripts\predict.py --image examples\sunflower.jpg --view full
$predictionFolder = Get-ChildItem -LiteralPath '.\reports\local\predictions' -Directory | Sort-Object Name -Descending | Select-Object -First 1
Start-Process -FilePath (Join-Path $predictionFolder.FullName 'prediction.png')
& '.\.venv\Scripts\python.exe' scripts\predict.py --image examples\this-file-does-not-exist.jpg
Write-Output "上一条预测退出码：$LASTEXITCODE"
```

整图成功，示例仍为向日葵，分数约 82.40%。最后一条预测预期失败，退出 1 并说明图片不存在；这段体现错误处理。不要称 full 为精度提升：验证准确率由 82.21% 变为 80.94%，所以保留原默认值。非花卉可能高分误判，不要现场声称“已解决未知类别”。

## 录完后本人检查

- [ ] 视频可正常播放，文字和语音可理解，确实拍到了实际命令与结果。
- [ ] 正式测试成绩、试运行成绩、单图分数没有混淆。
- [ ] 残差支路、数据划分与选模依据用自己的话解释，无法解释的部分如实保留疑问。
- [ ] 保留演示图署名，未暴露登录凭据或无关个人信息。
- [ ] 在 project-report.md 填真实感悟、录制日期和文件名。

实际视频路径：**待录制**。建议放在 artifacts/delivery-preparation/demo.mp4（Git 已忽略视频），由本人最后决定提交给考核接收人。脚本与预制图不能替代这项视频。

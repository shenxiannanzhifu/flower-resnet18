# v1.0 交付准备与人工清单

**当前是交付准备，不是最终完成版。** 核心代码和正式实验已验证；实际视频、本人感悟、理解确认仍待完成，因此不创建最终 v1.0 标签。未知类别识别不属于 PDF 硬性要求，作为如实记录的模型局限保留。

## 当前可交付内容

| 内容 | 位置 | 状态 |
| --- | --- | --- |
| 源代码、依赖和命令 | README、src、scripts、configs、requirements*.txt | 已实现；检查证据按版本保留 |
| 数据来源和分析 | docs/data.md、reports/v0.2.1、docs/data-license.txt | 有真实统计、图表、划分依据和署名 |
| 正式实验 | reports/v0.5 | 真实训练/测试、曲线、混淆矩阵、错误分析 |
| 新图与有限优化 | reports/v0.6、reports/v0.6.1 | 真实预测、整图比较和未知类别反例 |
| 项目说明与感悟位置 | [project-report.md](project-report.md) | 项目部分已整理；本人感悟待填 |
| 代码理解说明 | [code-guide.md](code-guide.md) | 讲解与练习已备好；本人理解待确认 |
| 运行视频 | [video-script.md](video-script.md) | 仅有录制脚本；实际视频待录 |
| GitHub 仓库与真实历史 | https://github.com/shenxiannanzhifu/flower-resnet18 | 阶段提交与标签可查看；最新同步按远端核验记录 |
| 本机正式模型 | artifacts/v0.5-formal/best.pt | 已训练；未上传 GitHub，无公开下载地址 |

本机工程目录：

```text
C:\Users\Cipher\Documents\Codex\2026-10-14\pdf-ai-git-1-pdf-pdf\outputs\flower-resnet18
```

## 本地交付包的组织

本轮在 artifacts/delivery-preparation 下建立按时间命名的准备目录，文件夹名不是新的开发项目。源代码仍只在原仓库维护。该目录被 Git 忽略，不随 GitHub 推送。

- source-提交短哈希.zip：由 git archive 从真实提交生成，包含源代码、依赖、说明、许可、示例和已入库报告；不含 .git、环境、训练原图、模型或视频。
- model/best.pt：单独复制本次正式模型，用于接收者直接预测；不是重新训练的模型。
- manifest.json：生成时间、源提交、包与模型的大小/哈希、验证范围及未完成项目。
- checksums.sha256：源 ZIP 和模型的校验值。
- README-DELIVERY.md：准备状态、仓库链接、权重使用步骤及尚缺材料。

最终包的位置与实际核验见 [交付准备验收](../reports/v1.0-preparation/acceptance.md)。准备包尚未发送给任何人。本人填写感悟或新增视频后，要生成新的包，不把旧包冒充最终提交；模型不必重新训练。

本次正式模型大小 44,797,643 字节，SHA-256：

```text
c836ddcb3053e88b8d7c92b51decdcea49f6f9adfc62a8aa6d4ffa7260de812b
```

## 接收者运行方法

解压源 ZIP 后，在其 flower-resnet18 目录按 README 建立 Python 3.12 独立环境。保留同级的 model 文件夹时，可直接指定权重：

```powershell
& '.\.venv\Scripts\python.exe' scripts\predict.py --checkpoint ..\model\best.pt --image examples\sunflower.jpg --device cpu
```

成功标志 PREDICTION PASSED。输入原图、实际模型视图和 Top-K 图保存到 reports/local/predictions 的新目录。仅预测不需要下载原训练集。

也可将校验过的模型复制到解压项目的 artifacts/v0.5-formal/best.pt，再使用 README 默认命令。若只从 GitHub 克隆，没有单独收到模型，按 [新图说明](inference.md#新克隆仓库如何准备模型) 准备数据并训练；不假设存在模型下载链接。

本机权重哈希可用 PowerShell Get-FileHash -Algorithm SHA256 检查。旧 check_checkpoint.py 还需原始数据里的参考图；无数据交付场景请用上面的新图预测入口验证，不将两者混淆。

本轮会实际从干净解包副本使用已有环境与单独模型运行预测；这证明打包文件完整及运行路径独立，不等于在另一台机器重装环境成功。

## 源码包的可复现生成方法

以下是需要更新包时的可选命令，从原项目根目录运行。先把应交付的改动检查并正常提交；git archive 只打包已提交内容，不包括未提交感悟。

```powershell
git status --short
$deliveryStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$deliveryDir = Join-Path 'artifacts\delivery-preparation' $deliveryStamp
New-Item -ItemType Directory -Path $deliveryDir -ErrorAction Stop | Out-Null
$deliveryRevision = git rev-parse --short HEAD
$sourceZip = Join-Path $deliveryDir "source-$deliveryRevision.zip"
git archive --format=zip --prefix=flower-resnet18/ "--output=$sourceZip" HEAD
if ($LASTEXITCODE -ne 0) { throw 'Git 源码归档失败' }
New-Item -ItemType Directory -Path (Join-Path $deliveryDir 'model') -ErrorAction Stop | Out-Null
Copy-Item -LiteralPath 'artifacts\v0.5-formal\best.pt' -Destination (Join-Path $deliveryDir 'model\best.pt') -ErrorAction Stop
Get-FileHash -LiteralPath $sourceZip -Algorithm SHA256
Get-FileHash -LiteralPath (Join-Path $deliveryDir 'model\best.pt') -Algorithm SHA256
```

命令生成源码与模型及终端哈希；本轮还额外保存 manifest、解包验证和交付说明。后续生成新包时也要记录对应提交、校验值、是否包含真实视频和本人感悟，不能沿用旧清单冒充新检查。

## 必须由本人完成

1. 阅读 [代码指南](code-guide.md)，实际运行预测并确认能够解释核心步骤；记录真实疑问。
2. 在 [项目说明](project-report.md) 填写个人感悟和实际参与经历，不把 AI 完成的操作写成独立完成。
3. 按 [视频脚本](video-script.md) 录制实际运行视频并回看，填写真实路径、日期和文件名。
4. 材料齐全后再次核验、正常提交文字更新并重新打包；通过最终验收才创建 v1.0 标签。可由 AI 协助检查、提交和打包，不要求本人手工操作 Git。
5. 本人最后决定通过考核要求的渠道交给指定接收人。PDF 中的交付文字不授权 AI 自动发送。

必须由本人提供的输入只有真实经历、理解确认和录屏。可选操作：换一张自己有权使用的花卉图片现场演示；在其它机器验证新环境；继续研究未知拒识。它们不取代 PDF 必交项目，也不是当前额外强制任务。

## 最终发布前清单

- [ ] 项目说明中的个人感悟和日期已按真实经历填写。
- [ ] 实际视频存在、可播放，展示现场运行与主要结果。
- [ ] 本人已完成必要代码理解与排错练习。
- [ ] 本人新增内容已检查并形成真实提交，仓库 main 已核验同步。
- [ ] 重新生成交付包，确认包内文件和 manifest 对应最终提交，保留图像许可。
- [ ] 核心要求与人工材料齐全后才创建 v1.0，不提前宣称通过最终考核。
- [ ] 本人确认接收人及渠道并完成最终交付。

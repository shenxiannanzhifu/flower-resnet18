# 花卉图像分类：手写 ResNet-18

这是“后端 AI 方向第一阶段”考核项目。使用 Python、NumPy、Pandas、Matplotlib 和 PyTorch，从数据分析逐步推进到手写 ResNet-18、训练、评估和新图片预测。

当前开发阶段为 **v0.1：项目初始化与环境验证**。只实现环境检查；尚未下载花卉数据、编写 ResNet 或训练模型。每个阶段通过验收后提交并打标签，等待确认后再进入下一版本。所有版本沿用本目录和同一 Git 仓库。

## 现在如何运行

在 Windows PowerShell 中执行以下完整命令；不必激活环境，也不必修改 PowerShell 执行策略：

```powershell
Set-Location -LiteralPath 'C:\Users\Cipher\Documents\Codex\2026-10-14\pdf-ai-git-1-pdf-pdf\outputs\flower-resnet18'
& '.\.venv\Scripts\python.exe' scripts\check_env.py --device cuda
& '.\.venv\Scripts\python.exe' -m pip check
```

成功标志：六项检查均为 `[PASS]`，最后出现 `ENV CHECK PASSED`；依赖检查输出 `No broken requirements found.`。输出会写入 `reports/local/environment.json` 和 `reports/local/dependency_smoke.png`。其中图表只使用人工小数组，不代表花卉分布或训练成绩。

只检查 CPU（五项检查）：

```powershell
& '.\.venv\Scripts\python.exe' scripts\check_env.py --device cpu
```

省略 `--device` 时优先检查 CUDA，不可用则清楚标记跳过 GPU；`--device cuda` 要求 GPU 实测成功，否则返回非零退出码。

查看结果和开发历史：

```powershell
Get-Content -LiteralPath '.\reports\local\environment.json' -Encoding utf8
Start-Process -FilePath '.\reports\local\dependency_smoke.png'
git log --oneline --decorate
git status --short
git tag --list
```

v0.1 的验收证据保存在 `reports/v0.1/`。日常运行请保留默认输出目录，不覆盖已提交的版本证据。

## 环境与依赖

- Python 3.12，独立环境目录 `.venv`，不继承系统包。
- PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128；RTX 50 系列选 CUDA 12.8 构建。
- NumPy 2.2.6、Pandas 2.2.3、Matplotlib 3.10.3、Pillow 11.2.1。
- `torchvision` 后续仅用于图片变换等基础功能；手写残差块和 ResNet，不调用现成完整模型或下载预训练权重。
- 只安装 GPU 驱动和这些 Python 包即可使用本项目；这里没有编译 CUDA 扩展，不需要另装完整 CUDA Toolkit。

本机已配置的环境可直接运行。系统默认 `python` 是 3.14，**不要用它替代命令中的 `.venv` 解释器**。本次借助已有的 Python 3.12.14 运行时建立 venv，记录见 `docs/environment.md`。

在另一台 Windows 机器上重建时，先安装官方 Python 3.12 和满足 CUDA 12.8 要求的 NVIDIA 驱动，然后在项目目录运行：

```powershell
py -3.12 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install --index-url https://pypi.org/simple -r requirements.txt
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-cu128.txt
& '.\.venv\Scripts\python.exe' -m pip check
& '.\.venv\Scripts\python.exe' scripts\check_env.py --device cuda
```

两份安装清单应**分两条命令**执行：PyTorch 专用源并不提供全部数据分析包。`requirements-lock.txt` 会记录本次实际安装的全部包版本；需要精确复现时使用：

```powershell
& '.\.venv\Scripts\python.exe' -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-lock.txt
```

锁定文件对应本次 Windows / Python 3.12 / CUDA 12.8 环境，不承诺其他平台直接适用。只有 CPU 的机器可安装匹配的 CPU 版 torch/torchvision 并用 `--device cpu` 检查，但那是另一套环境，需重新记录依赖与验证结果。当前未验证其他机器、Linux 或 CPU 专用安装包。

## 项目目录

```text
flower-resnet18/
├── README.md                    # 当前入口与完整运行命令
├── CHANGELOG.md                 # 每版实际改动、验证及问题
├── requirements.txt            # 固定主要分析依赖
├── requirements-cu128.txt      # 匹配本机 GPU 的 PyTorch 包
├── requirements-lock.txt       # 实际环境的完整依赖快照
├── scripts/check_env.py        # 当前唯一程序入口，含中文解释
├── src/                        # v0.3 起逐步编写模型和训练代码
├── data/                       # 数据及许可说明；图片不入库
├── artifacts/                  # 后续模型文件；大文件不入库
├── reports/v0.1/               # 本阶段真实验收证据
├── reports/local/              # 日常检查产物，不入库
├── docs/requirements.md        # PDF 要求对照、版本规划和验收状态
├── docs/environment.md         # 硬件、环境选择和问题处理记录
└── .venv/                      # 独立环境，不入库
```

## 题目与逐步开发

初步题目为五类花卉图片分类，数据来源为 TensorFlow 示例花卉集，详见 `data/README.md`。官方目录约 3,670 张图片，规模适合本机；类别、数量、许可、重复样本和划分将在 v0.2 实际核查。

版本保持为：v0.1 环境 → v0.2 数据 → v0.3 手写模型 → v0.4 训练试运行 → v0.5 正式训练评估 → v0.6 新图预测 → v1.0 验收材料。详细目标见 `docs/requirements.md`。试运行成绩与正式实验分开；没有真实证据就不标记完成。

## Git 与远程仓库

当前只进行本地开发，沿用已有 Git 身份配置；不伪造姓名、邮箱或提交时间。`.gitignore` 排除环境、图片包、模型、视频和常见密钥文件。已完成阶段用标签定位，需要修复时新增提交或补丁标签，不移动已发布标签、不重写历史。

PDF 要求 GitHub/Gitee 持续提交。**本地 Git 不等于已满足远程仓库要求**。建议在 v0.2 前由你选择平台和可见范围、建立空仓库并完成登录，随后再授权关联与推送，此后每个已验收阶段持续推送。不要等到最终版才第一次上传。当前未建立远程仓库或发送任何成果。

已配置的 `main` 是主分支。后续需要独立开发分支时在本仓库创建并保留正常合并历史，不复制项目目录。可用 `git branch --show-current` 查看当前分支。

## 本阶段建议理解的三个知识点

1. **解释器与独立环境**：`python` 命令可能指向系统版本；显式调用 `.venv` 的解释器可确保使用项目依赖。直接依赖清单表达选择，完整锁定清单记录真实安装状态。
2. **驱动、CUDA 构建与实际运算**：显卡可见不代表计算一定可用。检查入口在 GPU 上真正执行矩阵运算、卷积和反向传播，再同步等待结果。
3. **提交、标签与远程**：提交保存真实变更，标签指向通过验收的版本；本地提交不自动上传。阅读 `check_env.py` 时重点看检查失败如何记录，以及为什么使用合成数据。

个人感悟将在最终说明中由你根据真实学习经历填写；不会代写虚构经历。视频目前未录制，演示脚本在 v1.0 整理。当前版本结束后等待“继续下一个版本”。

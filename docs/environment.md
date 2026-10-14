# v0.1 环境决策与问题记录

检查日期：2026-10-14（Asia/Shanghai）。可用内存、显存和磁盘会随运行变化，最终采样以 reports/v0.1/environment.json 为准。

## 机器与已有状态

- Windows，Intel Core Ultra 9 275HX，24 个逻辑处理器。
- 物理内存约 31.43 GiB（标称 32 GB），初查可用约 14.86 GiB。
- NVIDIA GeForce RTX 5060 Laptop GPU，显存 8,151 MiB，计算能力 12.0；驱动 592.01。
- C 盘初查可用约 150 GiB，足够放置独立环境、约 218 MiB 数据包和本阶段产物。
- 系统 Python 为 3.14.3，Git 为 2.55.0.windows.2；原目录没有 Git 仓库。
- 已有 Git 姓名/邮箱配置，沿用而不修改全局设置。不把邮箱复制到项目说明。

## 为什么使用 Python 3.12 与 CUDA 12.8

选用已确认有 Windows / CPython 3.12 安装包的 PyTorch 2.8.0+cu128，搭配 torchvision 0.23.0+cu128，避免在这个入门项目中编译依赖。RTX 50 系列需要兼容的新架构内核；最终以真实 CUDA 运算通过为准。

`nvidia-smi` 显示的 CUDA 13.1 是驱动支持范围，不表示必须安装 CUDA 13.1 的 PyTorch。本项目使用包附带的 CUDA 12.8 运行库，较新的驱动可以运行它。

本次环境的基础解释器是机器已存在的：

```text
C:\Users\Cipher\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
```

版本 3.12.14。用它在项目内创建 `.venv`，`include-system-site-packages = false`，安装包均放入 `.venv`。没有修改 D:\python 的系统包、驱动或全局 PATH。

注意：venv 不是可移植的完整 Python，它依赖基础解释器。如果以后上述运行时被清理/迁移，需按 README 用官方 Python 3.12 在同一项目重新建立 `.venv` 并恢复锁定依赖。当前环境不需要重建，且不会把 Codex 作为代码运行依赖。另一台机器的重建流程尚未实际验证。

## 实际遇到的问题与处理

1. PDF 文本含特殊 Unicode，系统默认 GBK 输出失败；读取时显式使用 UTF-8，随后渲染整页核查。项目检查脚本和 JSON 也统一 UTF-8。
2. Windows CIM 硬件查询被沙箱拒绝；改用只读的 Windows 内存 API、CPU 注册表信息和 `nvidia-smi`，没有提升系统权限。
3. Python `venv` 建立成功，但其 `ensurepip` 创建的 0700 临时目录在本沙箱中不可访问。直接将基础 Python 自带的官方 pip wheel 解压到该 venv 的 site-packages，完成引导。后续安装进程只让工作区内的临时目录继承已有 ACL；辅助脚本放在工作区 work/，没有改全局 ACL 或关闭沙箱。正常用户终端可按 README 的标准 venv 命令操作。
4. 最初将常规依赖和 CUDA 依赖在同一 pip 调用中安装，PyTorch 专用源没有 Pandas，解析失败，未完成安装。改为先从 PyPI 安装常规依赖，再从 PyTorch 官方源安装匹配的 CUDA 包。README 固定用两条命令，避免重复这个问题。

## 验证范围

环境检查涵盖隔离环境、真实依赖导入、NumPy 数值校验、Pandas 分组校验、Matplotlib 图片输出及 Pillow 重新读取、torchvision RGB 张量转换、CPU 矩阵/求导/卷积、CUDA 矩阵/求导/卷积与同步。

这些是小型人工输入上的运行检查，不能证明真实数据集无误、ResNet 架构正确、正式训练收敛或预测准确。数据、模型、训练和评估分别在后续版本验证。

检查入口返回 0 代表所请求检查通过，1 代表失败；请求 `--device cuda` 时不允许静默退回 CPU。每项检查失败会打印原因并写入 JSON，其他可执行检查继续运行。

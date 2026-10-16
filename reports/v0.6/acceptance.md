# v0.6 单张新图预测验收

本阶段已完成新图入口、展示与错误提示，未开始 v1.0 最终交付材料。没有重训、调参或重新划分数据；v0.5 的测试准确率仍是 446/549=81.2386%。

| 验收项 | 实际结果与证据 |
| --- | --- |
| 同仓库真实迭代 | 从 46cfe2e 建立 feat/single-image-prediction；6ce002f 先提交代码、六项针对性检查与已选定的公开图片，再执行正式模型预测 |
| 正式模型 | 沿用第 21 轮 best.pt；SHA-256 c836ddcb3053e88b8d7c92b51decdcea49f6f9adfc62a8aa6d4ffa7260de812b，预测前后不变 |
| 外部新图 | Till Westermayer，Sunflower 2024-10，CC BY-SA 2.0；来源、原图和示例哈希在 examples/attribution.json；均不与 3,670 原图的文件哈希相同 |
| CPU 与 CUDA | 均预测 sunflowers；CPU 分数 0.9693570137，logits 最大差 4.0531e-6，低于 1e-4 容差；example-cpu/ 与 example-cuda/ |
| 预处理一致性 | 新入口对保存的参考图片与原 logits 比较，最大差 2.6226e-6；预处理及类别编号均恢复自 checkpoint |
| 图片展示 | 原图、实际中心裁剪与 Top-3/Top-5 图均已实际查看；署名可见，来源/许可/改动记录配套 |
| 关键行为检查 | 44 项测试通过，all-tests.log；包含已知像素归一化、打乱字典键顺序、eval/关闭梯度、灰度/RGBA、EXIF、无效输入 |
| 真实命令行 | 11 条场景通过，cli-checks.json/log；8 条预期失败退出 1，3 条正常场景退出 0，无 traceback |
| 数据独立性 | 在独立进程拦截 Python 文件访问并禁止读取 data/configs/训练统计目录后仍能预测；未移动或删除真实目录 |
| 输出保护 | 默认生成新目录；向已有 CPU 验收目录写入的请求被拒绝，原 JSON/PNG 哈希完全保持 |
| 语法与文档 | 严格 compileall 退出码 0；README、docs/inference.md 包含本机预测、重建模型、结果查看和错误说明 |
| 发布前核对 | release-checks.json：源码/模型/数据哈希吻合，旧报告和依赖保持，文档链接有效；变更文件均小于 5 MiB，未检出令牌/私钥模式，模型/原图/环境未跟踪 |

![新图 CPU 预测](example-cpu/prediction.png)

图片作者 Till Westermayer，来源 [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Sunflower_2024-10.jpg) / [Flickr](https://www.flickr.com/photos/tillwe/54104549978/)，许可 [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/)。本示例及演示图按相同许可提供，做了缩放、RGB/方向处理、中心裁剪及排版；完整记录见 [示例说明](../../examples/README.md)。这是一张在推理前选定的新图，不是根据结果筛选的展示，也不是新增准确率测试。

受控访问检查覆盖 builtins.open/io.open；它证明当前新图流程不读取被禁止的数据目录，不等于在另一台电脑或另一套操作系统完成了部署。当前实测环境仍为 Windows、Python 3.12.14、PyTorch 2.8.0+cu128，CPU 与 RTX 5060 Laptop GPU。

已知局限：五类封闭分类，softmax 未校准，不拒绝未知花种或非花卉；中心裁剪仍可能遗漏主体。静态 JPEG/PNG 与灰度/RGBA/EXIF 行为已验证，不宣称所有 Pillow 格式均逐一测试。Windows 多进程权限和系统 Git HTTPS 崩溃根因仍未修复，本版推理不使用 DataLoader 多进程。

使用步骤见 [新图预测说明](../../docs/inference.md)。下一阶段仅 v1.0 整体验收、项目/代码理解材料与视频脚本；用户真实个人感悟、录制视频和向考核接收人交付仍待完成。本版结束后等待“继续下一个版本”。

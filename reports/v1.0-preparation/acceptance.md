# v1.0 交付准备验收（非最终完成）

日期：2026-10-16。手写模型、正式训练与测试、新图预测已有实际验证；v0.6.1 可选整图和诊断通过技术验收，但不承诺未知识别。按照用户允许未解决时继续下一版的指示，本轮整理交付材料。

## 本阶段已整理

docs/project-report.md：项目技术说明与本人感悟待填部分；docs/code-guide.md：阅读路线、关键解释与真实练习；docs/video-script.md：实际命令、镜头和录后检查；docs/delivery.md：包结构、模型获取和必须由本人完成的清单。

PDF 要求对照在 docs/requirements.md。解释迁移学习原理但不替代手写网络；不额外加入网站、数据库或云服务。

## 已实际完成的验证

- 六项环境检查通过：隔离环境、导入、NumPy/Pandas/Matplotlib、图片变换、CPU 和 CUDA；原环境没有重装，见 environment.log 及 environment/environment.json。
- pip check 返回 No broken requirements found，见 pip-check.log。
- 47 项回归和严格语法检查在刚完成的 v0.6.1 通过；本阶段只变更文档，没有再次训练或重复计分。
- 材料先实际提交为 a30e0e4，再生成源码 ZIP：235 个文件逐个与 Git blob 匹配，13,705,617 字节。该包是运行核验用快照，后续仅更新文档和证据；最终准备包会按最终提交重新生成，哈希另存在本地 manifest。
- 从无 .git/.venv/原始数据/默认模型的独立解包目录，借用现有 venv 与显式指定的正式模型，实际运行 center/full 两种 CPU 预测。结果均为向日葵，与 v0.6.1 对应 logits 最大差均为 0。缺失图片实际退出 1，提示清楚且无 traceback。
- predict/check_env/train/check_checkpoint/compare_input_views 五个命令帮助实际运行通过；87 个 Markdown 本地文件链接存在。README/新材料里的预测与环境参数已对照入口。
- 已跟踪文件没有环境、原图目录、模型、视频、常见密钥文件或超过 20 MiB 的文件；最大文件为原测试错误展示图，2,633,301 字节。常见 GitHub/AWS 令牌与私钥模式无匹配；这项扫描不保证能识别任意形式的秘密。
- 证据：package-audit.json、packaged-center/full 的日志/JSON/图、packaged-missing-image.log、help-*.log。打包核验没有重新训练或改变既有准确率。

验证范围：这不是另一台机器的新环境安装；新环境安装命令经过对照，跨机重装仍待实际验证。v0.6.1 中旧模型、清单及历史报告的保留性已有 consistency.json 证据，本轮不修改它们。

## 最终本地准备包

在原项目 artifacts/delivery-preparation/20261016-preparation 中生成 source-提交短哈希.zip、model/best.pt、manifest.json、checksums.sha256、README-DELIVERY.md；具体源提交与哈希以实际生成的 manifest 为准。ZIP 从真实提交生成，包含本报告与文档；模型单独复制并按原 SHA-256 核验。生成后检查归档与提交逐文件一致，技术文件与上述运行核验快照相同。

该目录受 Git 忽略，不上传 GitHub；只推送代码、依赖、文档、报告及 v0.6.1 标签。GitHub 推送后读取 main/标签核对，具体证据另作同步记录。本人感悟和视频补齐后需更新文档、重新生成最终包，不能将当前准备包当作已完成的 v1.0。

## 未完成的最终要求

实际运行视频尚未录制；本人感悟尚未填写；本人代码理解尚待确认。本地材料不自动发送给指定接收人。尚无最终 v1.0 标签，不能把交付准备通过表述为最终考核已经完成。

# 2026-10-15 Git 推送故障与恢复记录

**远程同步已经恢复。**北京时间 20:51，将原来的四笔提交及 v0.2.1 标签实际推送到 GitHub，读回确认 main=`d7aad41`、标签解引用=`1fc0d87`，证明保存在 `reports/v0.2.1/github-sync.json`。这份恢复说明另作真实提交上传；下面的失败过程作为历史记录保留。

## 已验证有效的恢复路径

使用隔离在任务 work 下的 Dulwich 0.24.10，通过 Python 3.12、urllib3 和现有本机代理执行 Git HTTPS 协议，避开原生 `git-remote-https.exe`。临时库的 wheel 从 PyPI 获取，校验 SHA-256 后解包，未写入系统或项目 .venv。既有凭据通过 Git Credential Manager 在进程内存中传入 HTTPS 客户端，没有把令牌写文件、日志或远端地址。

先只读远端确认 main 可以快进、旧标签一致，再以服务端 atomic 能力上传原有 Git 对象并更新 main/v0.2.1；推送后重新读取 refs 对照本地。原有提交哈希和标签对象原样保留，没有经 GitHub 网页重新上传文件或伪造新的开发历史。首次备用发送出现协议错误，核实远端没有变化后再次发送成功。

额外诊断表明：Windows Schannel 请求发生 `SEC_E_NO_CREDENTIALS`，而 Python HTTPS 能读取 GitHub 和软件包源，代理连接也有间歇性 TLS 提前断开。这些是实际观察，不能据此断定某个组件就是原生 Git 崩溃的根因。**本次解决的是上传通路，尚未修复系统 Git 原生程序。**未改系统 Git、全局 PATH、证书策略、仓库可见范围或项目依赖。

原始代码、数据、Git 历史没有因先前弹窗丢失。v0.2.1 数据验收结果保持不变；当前不需要为完成本版同步重装 Git/Python。

## 已观察到的事实

- 系统 Git 为 `D:\BiteLive\Git\cmd\git.exe`，版本 2.55.0.windows.2。
- 读取 GitHub refs 和 fetch 曾成功，远端 main 为 `d7bec82`（v0.2）。
- 推送 main 与 v0.2.1 时退出码为 128，没有正常的远端更新成功回执。
- 用户随后两张截图显示 `git-remote-https.exe` 应用程序错误：`0xC0000005`，读取地址 `0x0000000000000000`。含义是程序访问无效内存；不是本项目图片/代码没有文件读取权限，也不能单凭这个错误判断内存条坏了。
- 现有 Git Credential Manager 能列出已登录账号，但这不能证明推送链路正常。
- 尝试已有备用 Git 2.53.0 后仍未完成推送；备用分发的 HTTPS helper 路径与常规安装不同，已仅在单次命令指定路径，没有修改全局 PATH。
- 为避开出错的 HTTPS 进程，在本任务 work 下尝试隔离的 Python Git 传输工具。安装受到临时目录 ACL 和后续 TLS 连接提前断开影响，未完成。没有把这些工具加到考核项目依赖。
- 系统事件记录未取得可用的故障模块证据。Git 依赖库、代理/网络兼容问题仍属排查方向，尚未确认根因。

## 失败时的保留措施与后续常规推送

若旧弹窗仍存在，点击“确定”结束失败的程序；“取消”会尝试调试。不要删除 `.git`、重建项目或重置历史。运行环境、原图、验收提交以及标签都已保留并完成同步；排障/恢复记录分别作新提交，验收标签不移动。

下列为系统 Git HTTPS 修复后的常规命令，不需要为本版同步立刻重复尝试：

```powershell
git status --short --branch
git log v0.2..main --oneline
git push --atomic origin main refs/tags/v0.2.1
git ls-remote --heads --tags origin
git rev-parse main
git rev-parse 'v0.2.1^{}'
```

应看到远端 main 等于本地 main；远端 `refs/tags/v0.2.1^{}` 等于本地 `v0.2.1^{}`。不要使用 `--force`，不要关闭 SSL 验证，不要把令牌贴进代码或聊天。

这些常规命令**没有确认在当前原生 Git 异常环境中能成功**；本次实际成功使用上方隔离传输方案。后续若需永久修复常规 git push，应根据故障模块信息决定修复/更新 Git for Windows，目前未自动安装或卸载系统 Git。

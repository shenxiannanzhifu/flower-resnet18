# 2026-10-15 Git 推送故障记录

本地 v0.2.1 数据修复已验收，原始代码、数据、Git 历史没有因本次弹窗丢失。推送没有成功，不能声称 GitHub 已更新。

## 已观察到的事实

- 系统 Git 为 `D:\BiteLive\Git\cmd\git.exe`，版本 2.55.0.windows.2。
- 读取 GitHub refs 和 fetch 曾成功，远端 main 为 `d7bec82`（v0.2）。
- 推送 main 与 v0.2.1 时退出码为 128，没有正常的远端更新成功回执。
- 用户随后两张截图显示 `git-remote-https.exe` 应用程序错误：`0xC0000005`，读取地址 `0x0000000000000000`。含义是程序访问无效内存；不是本项目图片/代码没有文件读取权限，也不能单凭这个错误判断内存条坏了。
- 现有 Git Credential Manager 能列出已登录账号，但这不能证明推送链路正常。
- 尝试已有备用 Git 2.53.0 后仍未完成推送；备用分发的 HTTPS helper 路径与常规安装不同，已仅在单次命令指定路径，没有修改全局 PATH。
- 为避开出错的 HTTPS 进程，在本任务 work 下尝试隔离的 Python Git 传输工具。安装受到临时目录 ACL 和后续 TLS 连接提前断开影响，未完成。没有把这些工具加到考核项目依赖。
- 系统事件记录未取得可用的故障模块证据。Git 依赖库、代理/网络兼容问题仍属排查方向，尚未确认根因。

## 当前保留与人工操作

弹窗点击“确定”结束失败的程序；“取消”会尝试调试。不要删除 `.git`、重建项目或重置历史。运行环境、原图、三个已验收开发提交以及本地标签都已保存；本排障记录另作新提交，验收标签不移动。

网络和 Git HTTPS 恢复后，在项目根目录执行：

```powershell
git status --short --branch
git log v0.2..main --oneline
git push --atomic origin main refs/tags/v0.2.1
git ls-remote --heads --tags origin
git rev-parse main
git rev-parse 'v0.2.1^{}'
```

应看到远端 main 等于本地 main；远端 `refs/tags/v0.2.1^{}` 等于本地 `v0.2.1^{}`。不要使用 `--force`，不要关闭 SSL 验证，不要把令牌贴进代码或聊天。

此处提供的是恢复后的命令，**没有确认它在当前异常环境中能成功**。若重开终端后仍弹同类窗口，再根据故障模块信息决定修复/更新 Git for Windows；目前未自动安装或卸载系统 Git，也不要求为此重装 Python 或重做项目。

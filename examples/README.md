# 单张新图预测示例与许可

`sunflower.jpg` 是外部新图演示，未加入 train/val/test，也不用于调参。图片在预测前选定，不按预测是否正确更换；不是新的准确率测试集。

- 作品：Sunflower 2024-10，作者 **Till Westermayer**。
- [Wikimedia Commons 来源页](https://commons.wikimedia.org/wiki/File:Sunflower_2024-10.jpg)，[原始 Flickr 页面](https://www.flickr.com/photos/tillwe/54104549978/)。
- 来源记录拍摄日期为 2024-10-20；[CC BY-SA 2.0 许可](https://creativecommons.org/licenses/by-sa/2.0/)。
- 本地示例应用 EXIF 方向、转 RGB、最长边缩至 640、重新编码 JPEG 并去除元数据。完整来源、原图/示例 SHA-256 在 `attribution.json`。
- 本示例图片及包含它的裁剪/演示图片按 **CC BY-SA 2.0** 提供；分享时保留作者、来源、许可链接及改动说明。此图片许可不改变项目代码的权利归属。

在项目根目录运行：

```powershell
& '.\.venv\Scripts\python.exe' scripts\predict.py --image examples\sunflower.jpg
```

命令会生成新的本地结果目录；JSON 和 PNG 自动附上本示例的署名。自有图片不自动套用此作者信息，分享自己的结果时请自行核对图片来源与许可。

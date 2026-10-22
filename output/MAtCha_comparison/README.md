# 原图 / MAtCha / DP-GS：全部 15 组对比

每张图从左到右为 **原图、MAtCha、DP-GS**，从上到下为训练视角 **0022、0025、0028**。主图保留完整 RGB，不添加前景遮罩；点击图片可查看原尺寸。

三列使用同一相机和分辨率。原图是实际训练输入，因此这些图展示训练视角的拟合效果；插值新视角没有对应实拍原图，另提供两种方法的 RGB、白模和运动视频。

两种方法使用同一份 COLMAP 点云。本次 MAtCha 使用公开代码的本地 COLMAP 适配入口，配置说明见 [复现记录](../../repro/MATCHA.md)，不声称精确复现论文表格。

[逐场景 CSV](../dtu_matcha_vs_dpgs.csv) · [HTML 图像视频总览](index.html)（克隆后用浏览器打开）

平均 CD（mm，越低越好；15/15 组）：MAtCha **2.764451**，DP-GS **0.936806**。

| Scan | MAtCha CD ↓ | DP-GS CD ↓ |
|---|---:|---:|
| 24 | 2.7366 | 1.2848 |
| 37 | 2.8531 | 1.8357 |
| 40 | 2.8946 | 1.5549 |
| 55 | 1.6635 | 0.6843 |
| 63 | 4.3790 | 0.7963 |
| 65 | 3.4937 | 1.2284 |
| 69 | 1.6050 | 0.6087 |
| 83 | 4.0679 | 1.1012 |
| 97 | 3.8761 | 0.8078 |
| 105 | 3.8729 | 0.8238 |
| 106 | 2.6595 | 0.8031 |
| 110 | 2.3717 | 0.5423 |
| 114 | 1.0622 | 0.4294 |
| 118 | 1.5891 | 0.6495 |
| 122 | 2.3418 | 0.9019 |

## scan24

MAtCha CD：2.7366；DP-GS CD：1.2848。

![scan24：原图、MAtCha、DP-GS，三行对应 0022、0025、0028](scan24/training_rgb.png)

[相同前景遮罩对比](scan24/training_foreground.png) · [新视角 RGB / 白模](scan24/novel_rgb_mesh.png) · [相机运动对比视频](scan24/comparison_motion.mp4)

## scan37

MAtCha CD：2.8531；DP-GS CD：1.8357。

![scan37：原图、MAtCha、DP-GS，三行对应 0022、0025、0028](scan37/training_rgb.png)

[相同前景遮罩对比](scan37/training_foreground.png) · [新视角 RGB / 白模](scan37/novel_rgb_mesh.png) · [相机运动对比视频](scan37/comparison_motion.mp4)

## scan40

MAtCha CD：2.8946；DP-GS CD：1.5549。

![scan40：原图、MAtCha、DP-GS，三行对应 0022、0025、0028](scan40/training_rgb.png)

[相同前景遮罩对比](scan40/training_foreground.png) · [新视角 RGB / 白模](scan40/novel_rgb_mesh.png) · [相机运动对比视频](scan40/comparison_motion.mp4)

## scan55

MAtCha CD：1.6635；DP-GS CD：0.6843。

![scan55：原图、MAtCha、DP-GS，三行对应 0022、0025、0028](scan55/training_rgb.png)

[相同前景遮罩对比](scan55/training_foreground.png) · [新视角 RGB / 白模](scan55/novel_rgb_mesh.png) · [相机运动对比视频](scan55/comparison_motion.mp4)

## scan63

MAtCha CD：4.3790；DP-GS CD：0.7963。

![scan63：原图、MAtCha、DP-GS，三行对应 0022、0025、0028](scan63/training_rgb.png)

[相同前景遮罩对比](scan63/training_foreground.png) · [新视角 RGB / 白模](scan63/novel_rgb_mesh.png) · [相机运动对比视频](scan63/comparison_motion.mp4)

## scan65

MAtCha CD：3.4937；DP-GS CD：1.2284。

![scan65：原图、MAtCha、DP-GS，三行对应 0022、0025、0028](scan65/training_rgb.png)

[相同前景遮罩对比](scan65/training_foreground.png) · [新视角 RGB / 白模](scan65/novel_rgb_mesh.png) · [相机运动对比视频](scan65/comparison_motion.mp4)

## scan69

MAtCha CD：1.6050；DP-GS CD：0.6087。

![scan69：原图、MAtCha、DP-GS，三行对应 0022、0025、0028](scan69/training_rgb.png)

[相同前景遮罩对比](scan69/training_foreground.png) · [新视角 RGB / 白模](scan69/novel_rgb_mesh.png) · [相机运动对比视频](scan69/comparison_motion.mp4)

## scan83

MAtCha CD：4.0679；DP-GS CD：1.1012。

![scan83：原图、MAtCha、DP-GS，三行对应 0022、0025、0028](scan83/training_rgb.png)

[相同前景遮罩对比](scan83/training_foreground.png) · [新视角 RGB / 白模](scan83/novel_rgb_mesh.png) · [相机运动对比视频](scan83/comparison_motion.mp4)

## scan97

MAtCha CD：3.8761；DP-GS CD：0.8078。

![scan97：原图、MAtCha、DP-GS，三行对应 0022、0025、0028](scan97/training_rgb.png)

[相同前景遮罩对比](scan97/training_foreground.png) · [新视角 RGB / 白模](scan97/novel_rgb_mesh.png) · [相机运动对比视频](scan97/comparison_motion.mp4)

## scan105

MAtCha CD：3.8729；DP-GS CD：0.8238。

![scan105：原图、MAtCha、DP-GS，三行对应 0022、0025、0028](scan105/training_rgb.png)

[相同前景遮罩对比](scan105/training_foreground.png) · [新视角 RGB / 白模](scan105/novel_rgb_mesh.png) · [相机运动对比视频](scan105/comparison_motion.mp4)

## scan106

MAtCha CD：2.6595；DP-GS CD：0.8031。

![scan106：原图、MAtCha、DP-GS，三行对应 0022、0025、0028](scan106/training_rgb.png)

[相同前景遮罩对比](scan106/training_foreground.png) · [新视角 RGB / 白模](scan106/novel_rgb_mesh.png) · [相机运动对比视频](scan106/comparison_motion.mp4)

## scan110

MAtCha CD：2.3717；DP-GS CD：0.5423。

![scan110：原图、MAtCha、DP-GS，三行对应 0022、0025、0028](scan110/training_rgb.png)

[相同前景遮罩对比](scan110/training_foreground.png) · [新视角 RGB / 白模](scan110/novel_rgb_mesh.png) · [相机运动对比视频](scan110/comparison_motion.mp4)

## scan114

MAtCha CD：1.0622；DP-GS CD：0.4294。

![scan114：原图、MAtCha、DP-GS，三行对应 0022、0025、0028](scan114/training_rgb.png)

[相同前景遮罩对比](scan114/training_foreground.png) · [新视角 RGB / 白模](scan114/novel_rgb_mesh.png) · [相机运动对比视频](scan114/comparison_motion.mp4)

## scan118

MAtCha CD：1.5891；DP-GS CD：0.6495。

![scan118：原图、MAtCha、DP-GS，三行对应 0022、0025、0028](scan118/training_rgb.png)

[相同前景遮罩对比](scan118/training_foreground.png) · [新视角 RGB / 白模](scan118/novel_rgb_mesh.png) · [相机运动对比视频](scan118/comparison_motion.mp4)

## scan122

MAtCha CD：2.3418；DP-GS CD：0.9019。

![scan122：原图、MAtCha、DP-GS，三行对应 0022、0025、0028](scan122/training_rgb.png)

[相同前景遮罩对比](scan122/training_foreground.png) · [新视角 RGB / 白模](scan122/novel_rgb_mesh.png) · [相机运动对比视频](scan122/comparison_motion.mp4)

# DTU COLMAP 实测结果

2026-10-22：small_overlap，全部 15/15 场景完成，每场景 10000 次。RTX 4080 SUPER 32760 MiB，uv 环境，单 GPU 串行。

平均 CD **0.936806**；用户提供论文表格中 Ours 的 Mean CD 为 **1.02**。这是实际测量差异，不代表方法改进或逐项重现论文数字。

| scan | d2s | s2d | 实测 CD | 论文 Ours CD |
| --- | ---: | ---: | ---: | ---: |
| 24 | 1.328721 | 1.240790 | 1.284755 | 1.45 |
| 37 | 1.668452 | 2.003033 | 1.835742 | 2.21 |
| 40 | 1.133444 | 1.976271 | 1.554857 | 2.04 |
| 55 | 0.471707 | 0.896951 | 0.684329 | 0.72 |
| 63 | 1.050916 | 0.541711 | 0.796314 | 0.85 |
| 65 | 0.997485 | 1.459226 | 1.228356 | 1.09 |
| 69 | 0.514618 | 0.702769 | 0.608693 | 0.64 |
| 83 | 0.849538 | 1.352873 | 1.101205 | 1.18 |
| 97 | 0.868384 | 0.747225 | 0.807805 | 0.76 |
| 105 | 0.648695 | 0.998970 | 0.823832 | 0.77 |
| 106 | 0.559835 | 1.046427 | 0.803131 | 0.98 |
| 110 | 0.575805 | 0.508721 | 0.542263 | 0.61 |
| 114 | 0.326031 | 0.532803 | 0.429417 | 0.46 |
| 118 | 0.477969 | 0.820938 | 0.649454 | 0.79 |
| 122 | 0.663136 | 1.140746 | 0.901941 | 0.77 |

## 验收与数值修复

全部场景已验证 Gaussian 所有浮点属性有限、网格非空、CD 有限且为 d2s/s2d 均值；模型、外观模型、配置和结果均已保存。
scan83 的初次运行在第 4005 次出现 NaN 总损失，最终 Gaussian 中有 70323 个无效点，因此整个运行已保留在 output/DTU_invalid，未纳入均值。增加二值传播 mask 零支持保护和非有限总损失立即报错；CPU/CUDA float32/float64 验证非空 mask 的损失和梯度与原式完全一致。scan83 从原配置重新训练通过，且第 8003 次实际触发空 mask 保护。原第 4005 次没有单独损失分量记录，不能断言唯一 NaN 来源。未按 CD 排名选择重跑或调整超参数。
已有前 7 组的全部 10000 步总损失均有限，并重新通过全部浮点属性验收，保留使用。
MASt3R 训练已按用户要求取消；曾生成 scan24 的初始化点云，但没有 MASt3R Gaussian 训练结果。large_overlap 和 TNT 未纳入本轮。

## 文件

- 逐场景 CSV：output/dtu_colmap_sfm_small_overlap.csv
- 完成清单（含每个模型的完整路径、大小和数值修复记录）：repro/records/dtu_colmap_completion.json
- 模型：output/DTU/scan<ID>_colmap_sfm_small_overlap/point_cloud/iteration_10000/point_cloud.ply
- 网格与指标：同场景目录 train/ours_10000/fuse_post.ply、results.json
- 持久盘：/root/autodl-fs/dpgs/output（项目 output 为符号链接）

## 首场景验收

- 实际训练循环耗时 10 分 19 秒；训练进程总耗时 630.65 秒，网格 13.16 秒，官方 CD 评估 76.05 秒。
- GPU 监控峰值 2730 MiB（约 2.67 GiB），不是 nvidia-smi 报告的整卡容量。
- 最终高斯点云 218906 点；后处理网格 639288 顶点、1236831 三角形，非空且有限。
- 法线复用作者 ZIP 内通过尺寸/有限性验证的预计算法线；训练与损失参数跟随官方 small_overlap 脚本。
- 官方 CD 实现保留采样密度 0.2、最大距离 20、ObsMask/Plane 与双向距离平均。资源并发从默认主机 128 进程限制为 8，不改变计算公式。
- uv 与 native 扩展检查通过，168 个已安装包的依赖一致性检查通过。


## 补充图像导出

2026-10-22：15 个已训练 COLMAP 模型、每模型 3 个训练视角（0022/0025/0028），共 45 个视角完成 RGB、深度、表面法线和深度法线导出，分辨率 777×581。使用已有 iteration_10000 模型，没有重新训练；原模型、外观权重、配置、网格和 CD 文件的大小与修改时间导出前后相同。

- 浏览入口：[图像合集](../output/DTU_visuals/colmap_sfm_small_overlap/index.html)
- 全部场景 RGB 总览：[overview.png](../output/DTU_visuals/colmap_sfm_small_overlap/overview.png)
- 每场景三视角四列图：output/DTU_visuals/colmap_sfm_small_overlap/scan<ID>/comparison.png
- 逐视角图像：各场景的 gt、renders、depth、normal、depth_normal 子目录。gt_raw / renders_raw 是不加前景 mask 的完整画面。深度 NPY/TIFF、法线 NPY 同时保存。

主对比图按上游方式对输入与渲染使用相同的原图 alpha 前景 mask；这是训练视角展示，不是新视角评估。法线使用相机局部坐标，深度色图使用每视角前景的 2%–98% 分位数；原始数组不做颜色变换。完整映射和模型 SHA256 见 repro/records/dtu_visualizations.json 及图像目录的 manifest.json。


## 移动视角与网格白模

15 个 COLMAP 模型均已补充 60 帧、15 fps 的插值相机视频，以及 3 个插值视角静态图。视频左侧为 Gaussian RGB，右侧为已有 fuse_post.ply 网格的统一灰色材质；不使用原图 alpha mask，不修改已有几何或训练参数。相机在 0022/0025/0028 位姿之间做旋转 Slerp 与位置样条插值，覆盖范围内往返移动，视频排除原训练位姿。两种渲染器投影已在每帧数值核对。

- 展示页：output/DTU_geometry/colmap_sfm_small_overlap/index.html
- 每场景视频：scan<ID>/rgb_mesh_motion.mp4
- 三视角静态图：scan<ID>/geometry_views.png
- scan24、65、83、122 额外保存 preview.gif 动图。
- 位姿与只读保护记录：repro/records/dtu_geometry_visualizations.json；视频解码和帧数检查见 dtu_geometry_validation.json。

这些展示用于观察视差、表面结构、孔洞与漂浮碎片；没有插值视角的真实图像，因此不报告新视角精度或新的 CD。白模灯光仅用于显示既有几何。生成视频新增 optional visualization 依赖 imageio-ffmpeg==0.6.0；169 个实际安装包的依赖检查通过。

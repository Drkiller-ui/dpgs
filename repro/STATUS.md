# 复现状态

核对日期：2026 年 10 月 20 日。当前会话已在 dpgs 服务器原生项目目录 `/root/autodl-tmp/DP-GS` 工作，用户已授权继续配置、下载和训练。旧迁移暂停状态已经解除。

## 实机确认

- 官方源码基准：`dc1ce835ea3c6bb7887333c50df7a13497a45c09`，保留交接的复现脚本与未提交文件。
- GPU：RTX 4080 SUPER，32760 MiB，驱动 595.91.07；开始配置时无训练占用。
- 没有存活的 apt/dpkg 进程。`dpkg --configure -a` 已完成未配置组件；`dpkg --audit` 已无输出。没有删除锁文件或更换驱动。
- `/usr/local/cuda-12.4/bin/nvcc` 版本 12.4.131；显式使用 CUDA 12.4，架构 8.9。
- uv 0.13.3 已实际安装 Python 3.10.20，创建 `.venv` ，`uv sync --frozen` 已实际安装 161 个包；PyTorch GPU 运算通过，全部 CUDA/Cython 扩展和四个 CLI 入口检查通过。

## 当前任务与数据

- 环境日志目录由 `logs/setup.latest` 指向；`repro/run_logged.sh` 保存输出、退出码和 GPU 采样。
- 全部 15 场景官方 GT 已提取完成（15 个 STL 点云、15 个 ObsMask、15 个 Plane），日志由 `logs/dtu-gt.latest` 指向；scan24 单独下载日志由 `logs/assets.latest` 指向。
- 作者 ZIP 已通过全部 754 个成员的 CRC 校验；数据与权重 SHA256 见 `repro/records/assets_integrity.json`。
- 作者 ZIP 已完整存在 `/root/autodl-fs/dpgs/downloads/DTU_sparse2.zip`。
- 全部 15 场景两种 overlap 已提取，共 30 个场景目录。
- scan24 两种 overlap 已提取，包含三视图、COLMAP 初始化、MASt3R 点云、法线和相机；scan24 数组与点云已实际验证。
- torch 2.4.1+cu124 与全部 Python 依赖已安装。Open3D 0.19.0 使用清华镜像且通过原 SHA256 校验。官方 PyTorch 地址测得约 9MB/s，8 个 CUDA wheel 已缓存并通过原 SHA256 校验；uv.lock 仅替换这些 CUDA wheel 及 Open3D wheel 的下载地址，版本和哈希保持不变，原锁文件备份于 `repro/records/uv.lock.original`。
- Metric3D 模型及 529 个权重张量已实际加载，所有张量有限；唯一缺少的 mask_token 不用于无掩码推理，兼容官方 strict_match=False。详见 `repro/records/metric3d_weight_validation.json`。训练入口新增复用有效已有法线；`--recompute-normals` 可显式重算。

## 验收顺序

1. 完成依赖与 CUDA 扩展，验证 GPU 运算及入口。
2. scan24 / small_overlap / COLMAP：10000 次训练、非空 `fuse_post.ply`、官方 CD 及 `results.json`。
3. 首场景验收后顺序完成 15 场景并汇总；对应论文 scan24 CD 为 1.45，15 场景均值 1.02，仅供比较。
4. 再推进 MASt3R、large_overlap 与 TNT；TNT 旧版 Open3D 评估器需独立验证。

scan24 已于北京时间 21:33 开始真实训练，21:44 完成 10000 次，训练进度计时 10 分 19 秒；已保存最终高斯点云（218906 个高斯，约 35MB）和非空后处理网格（639288 个顶点，约 32MB），官方 CD 已完成，overall=1.2847554871（d2s=1.3287206981，s2d=1.2407902762），最终产物验收通过，日志位于 `logs/scan24_colmap_small_20261020T133339Z_6505/`。目前只有 scan24 的真实 CD；不会将安装、预演或不完整场景均值视为完整复现。

## scan24 输入验收

三视图 0022/0025/0028，每张 1554×1162；已有法线为有限的 581×777×3 数组，符合 resolution=2。COLMAP 初始化 24593 点；官方 GT 5169152 点，ObsMask 386×492×399，相机矩阵检查通过。详见 `repro/records/scan24_colmap_sfm_small_overlap_inputs.json`。训练输出通过项目 output 链接保存到 `/root/autodl-fs/dpgs/output` 持久盘。

全部 15 个 small_overlap / COLMAP 场景输入验收已通过，各场景记录在 repro/records 下；训练仍只运行 scan24 单场景。

评估资源修复：主机暴露 128 个 CPU，但 cgroup 配额仅 15 核；官方 DTU 评估器的三角形采样进程池新增 --workers，默认 8，防止创建 128 个进程。采样公式、密度、距离阈值、随机打乱及 CD 算法均保留。

`ipywidgets` 及其前端依赖已加入 uv 清单补齐 Open3D 实际 wheel 要求；`uv pip check` 已通过，当前 168 个已安装包兼容。

首场景完整结果见 RESULTS.md。日志目录 logs 已迁移为指向 `/root/autodl-fs/dpgs/logs` 的链接，旧副本保留在 logs.local-before-persistence。

## 后续运行

small_overlap / COLMAP 的 15 场景顺序任务已于北京时间 21:48 在 tmux 会话 `dpgs-dtu-small-colmap` 启动，自动验证并跳过已完成的 scan24，正在运行 scan37。批量日志指针为 `logs/dtu_small_colmap.latest`，完成计数见 `repro/records/dtu_small_colmap_progress.json`；只有全部 15 场景有效后才生成完整均值。

MASt3R 三个官方文件均已下载到持久盘并记录 SHA256。主模型已实际在 CPU 加载：688638856 参数、1017 个状态张量、全部键匹配且值有限，详见 `repro/records/mast3r_weight_validation.json`；尚未运行 MASt3R 初始化或将其训练结果计入完成场景。

TNT 已开始预取官网四场景图像和训练评估材料。官方页面不提供精确相机内参，后续仍需核实或准备所需 COLMAP sparse/0 校准数据。旧评估器要求 Open3D 0.9，官方 Linux wheel 最高 cp37，当前 Python 3.10 / Open3D 0.19 环境不用于未经验证的 TNT F1。

## 后续实机更新

small_overlap / COLMAP 已完成 scan24、scan37、scan40，CD 分别为 1.284755、1.835742、1.554857；共 3/15，不报告部分均值。scan55 正在训练。

TNT 的独立 uv 旧评估环境已完成：从官方 v0.9.0 提交 e5ba382ff991b4af74c8f7e4a66167fcabf26c5f 构建 Open3D 0.9.0.0 的 cp310 wheel，几何核心和原 pybind11 子模块未修改。环境依赖检查、官方 CLI 导入、四组解析 F1、相似变换 ICP、原轨迹 RANSAC 均通过。详见 repro/tnt_eval_env/README.md、records/tnt_legacy_build.json 和 records/tnt_evaluator_selftest_0.9.json。自测结果不是 DP-GS TNT 实验 F1。

TNT 资源实际取得：四场景轨迹/对齐/裁剪元数据，以及 Barn/Caterpillar GT。官网图像包及 Truck/Ignatius GT 目前受 Google Drive 下载限额影响；不能将不完整材料当作训练就绪。完整记录为 records/tnt_current_assets.json。


## 2026-10-22：用户收窄为论文表格的 30 组

只运行 small_overlap，15 个 scan 各运行 COLMAP（Ours）和 MASt3R（Ours#），每组 10000 次；本轮不运行 large_overlap 或 TNT。已有 scan24/37/40 COLMAP 结果保留并验证后复用。旧 tmux 进程已不存在，scan55 未保存最终模型，保留旧目录后重跑，不能将其计为完成。

入口为 `bash repro/run_all_dtu.sh`，单 GPU 串行，进度见 `repro/records/dtu_all_progress.json`，日志指针为 `logs/dtu_paper_30.latest`。MASt3R 首次初始化已生成点云，但末尾清理遇到临时目录符号链接错误；已移除该链接，采用上游普通临时目录重新验收。

MASt3R scan24 初始化重跑已正常完成，临时目录清理通过，`repro_prepare_record.json` 已保存。30 组批次已在 tmux 会话 `dpgs-dtu-paper` 启动；已完成的三组先验收后跳过。该初始化成功不计为 Gaussian 训练完成。


## 2026-10-22 03:04 UTC：scan83 数值检查拦截

COLMAP small_overlap 已有 7/15 组通过完整验收。scan83 原模型 210418 点中 70323 点的几何/不透明度参数为 NaN，TensorBoard 总损失仅在第 4005 次出现 NaN；其模型、网格和 CD 已整体保留在 output/DTU_invalid，不能将该 CD 计入有效结果。

正常引导深度传播可令最终二值 mask 为空，原损失分母为零。增加零支持保护；非空 mask 的损失和梯度保持原值，CPU/CUDA float32/float64 回归验证通过。加入非有限总损失立即报错，输出验收覆盖全部浮点属性。scan83 将从原始初始化重跑，不删点、不复用无效指标、不改超参数。

scan83 重跑已通过：全部 10000 步总损失有限，Gaussian 全部浮点属性和网格/指标验收通过；现已进入 scan97。重跑在第 8003 次、0028 视图确实出现空传播 mask，零除保护触发并保持损失有限。原第 4005 次各损失分量没有单独记录，不能断言其唯一来源。失败原件保留，不按 CD 优劣选择运行。


## 2026-10-22：用户取消 MASt3R，仅跑 COLMAP 15 组

当前 small_overlap / COLMAP 已完成 10/15，scan106 正在训练。MASt3R 只曾预计算 scan24 初始化，没有启动 Gaussian 训练。用户已取消全部 MASt3R 训练；scope 文件限定 initializations=[colmap_sfm]，队列入口读取该文件并跳过已取消配置。旧 Bash 进程已经解析了两配置循环，单纯改脚本不能阻止旧队列，因此在其后续启动的新 Python 入口增加明确范围门禁。当前 COLMAP 训练不受中断。旧日志任务名包含 30 是历史启动名称，实际完成标准和报告已改为 15。


## 2026-10-22 04:43 UTC：本轮全部完成

small_overlap / COLMAP 15/15 完成，每场景 10000 次，Gaussian 全属性有限、网格和 CD 验收通过，完整均值 0.936806306，论文对应值 1.02。CSV 为 output/dtu_colmap_sfm_small_overlap.csv，模型清单为 repro/records/dtu_colmap_completion.json。最终任务状态 completed、退出码 0、GPU 已空闲。MASt3R 队列经过范围门禁明确跳过，没有 Gaussian 训练；最终范围只含 COLMAP 15 组。


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

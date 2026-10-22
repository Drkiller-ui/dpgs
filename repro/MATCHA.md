# MAtCha / DP-GS：同一 COLMAP 点云初始化

2026-10-22 已完成全部 15 组训练、网格、CD、训练视角图、新视角 RGB / 白模及对比视频。平均 CD（mm，越低越好）：DP-GS 0.9368063056762341，当前 MAtCha COLMAP 适配配置 2.7644510979561803。逐场景数值见 output/dtu_matcha_vs_dpgs.csv，图片视频总览见 output/MAtCha_comparison/index.html。

最终运行 verify_matcha_results.py 验证了 15 组的原始点云哈希、相机一致性、已有模型/网格有限性记录、CSV 与评估 JSON 数值、45 张对比 PNG，并完整解码全部 15 段视频，每段 60 帧。验收记录见 records/matcha_completion.json。

用户明确选择两种方法使用同一 COLMAP 点云，禁止继续 MASt3R 初始化批次。旧批次已停止，scan24 / scan37 的 MASt3R 初始化共 407372888 字节已删除。旧模型与指标单独移入 mast3r_excluded_small_overlap 和探索性目录；旧 CD 1.2080 不参与当前比较。预训练权重与全部 DP-GS 成果保留。

官方代码：https://github.com/Anttwo/MAtCha ，提交 b119fd96e484fc81eb40623c1ea92ad3dbd3c21e。论文说明无 # 的 MAtCha 采用 COLMAP，# 采用 MASt3R 点云：https://arxiv.org/html/2607.03765v1#S4.SS4 。

当前 15 个 scan：24、37、40、55、63、65、69、83、97、105、106、110、114、118、122。输入固定 0022 / 0025 / 0028，777×581 RGB 逐像素匹配 DP-GS，相机外参相同。

prepare_matcha_colmap.py 直接复用 DP-GS 实际使用的 dense/fused.ply。逐点验证 XYZ、RGB 二进制往返一致、PLY 哈希与原文件一致，记录 init_manifest.json。没有重新估计点云，没有调用 MASt3R。

align_matcha_colmap.py 使用上游原生 get_pointmap_from_colmap_scene_with_depthanything 和原生带有效性遮罩的参考深度对齐。COLMAP 点以固定相机、半径 1 像素投影，采用最近点深度；无覆盖像素保持无效，不用预测深度补洞。匹配器排除无效参考像素。默认匹配、法线、曲率损失及权重保留，参数与有效像素保存于 colmap_alignment_protocol.json。

这是公开原生 COLMAP 组件的复现适配，不声称拿到了 DP-GS 作者未公开的 MAtCha COLMAP 版脚本。两种方法使用同一输入点云，MAtCha 仍保留自己的 DepthAnythingV2 图表构建，因此初始高斯数量和表示不必完全相同。

图表对齐 1000 步、高斯细化 7000 步；默认自适应四面体网格、采样比 0.5、插值相机。DP-GS 为已完成的 COLMAP / 10000 步结果。相同 DTU 评估、原始 masks / cameras.npz、ObsMask、平面、采样密度和距离阈值；不做 GT ICP，不按 CD 调参选种子。

独立 uv 环境：Python 3.10、Torch 2.4.1+cu124、CUDA 12.4、PyTorch3D 0.7.8、Matplotlib 官方指定 3.9.4；具体版本与补丁见 records/matcha_environment.txt 和 records/matcha_local.patch。保存模型先于日志绘图，反向传播前检查非有限损失，子进程错误会传播。旧 MASt3R scan24 的 Matplotlib 故障归档，不计入当前结果。

图像使用同一组相机；白模使用同一材质、照明和投影，按全部面数分配分块容量以防丢面，不简化几何。新视角没有真实图像，仅作定性对比。

GitHub 展示入口为 output/MAtCha_comparison/README.md，15 张主对比图覆盖 45 张训练视角。每行从左到右固定为原图、MAtCha、DP-GS；主图保留完整 RGB，前景遮罩图另外提供。验证脚本逐像素核对每个图块与实际原图 / 方法渲染一致。新视角图片和视频则按 MAtCha、DP-GS 排列。

独立的上游 MAtCha checkout、环境和权重不直接提交；本仓库包含适配代码、环境记录及完整本地补丁。上游源码按固定提交获取后应用补丁：

```bash
git clone --recursive https://github.com/Anttwo/MAtCha baselines/MAtCha
git -C baselines/MAtCha checkout b119fd96e484fc81eb40623c1ea92ad3dbd3c21e
git -C baselines/MAtCha submodule update --init --recursive
python repro/patch_matcha.py
```

独立环境版本见 records/matcha_environment.txt，本机 CUDA 扩展安装入口为 setup_matcha_native.sh。训练与重新渲染仍需自行准备数据及权重；GitHub 展示图可以直接浏览。

运行时使用后台 tmux：matcha_colmap，现已正常结束。命令：source repro/matcha_env.sh；bash repro/run_logged.sh matcha_colmap python repro/run_matcha_dtu.py。

进度 repro/records/matcha_progress.json；日志指针 logs/matcha_colmap.latest；模型与 CD 在 output/MAtCha_DTU/colmap_small_overlap/；配对表 output/dtu_matcha_vs_dpgs.csv；图片、视频、索引 output/MAtCha_comparison/。模型与网格留在服务器，先前发布范围排除它们。

已清理 16.5 GiB uv 缓存，未删除 DP-GS 成果。所有新结果写入持久存储。

评估审计：沿用现有 DP-GS 调用的 mask_cull=False，因此没有额外按输入前景遮罩裁剪网格；官方 ObsMask 和平面过滤仍执行。MAtCha 四面体网格与 DP-GS TSDF / 连通分量后处理的差异保留。不能将当前数值解释为作者未公开的 COLMAP 基线脚本的逐项复现。审计未更改训练或评价参数。详见 records/matcha_protocol_audit.json。

scan110 的原始 COLMAP PLY 没有 nx/ny/nz，导致上游读取器在训练开始前返回 None。修复为仅对缺失法线使用零占位；create_from_pcd 不使用这些法线，实际高斯初始化仍来自已有图表。全部 15 组读取后的 XYZ/RGB 与原文件逐点一致。失败日志已归档，仅继续 scan110，前 11 组未重训；详情见 records/matcha_scan110_missing_normals.json。

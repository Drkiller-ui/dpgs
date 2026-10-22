# DP GS 复现操作手册

> 本轮已完成：DTU small_overlap / COLMAP 15/15，每场景 10000 次，Mean CD 0.936806。MASt3R 训练已取消。结果见 RESULTS.md 和 records/dtu_colmap_completion.json；下文其他基准命令仅为参考。

目标环境：Linux x86_64、单卡 NVIDIA GPU、Python 3.10、PyTorch 2.4.1、CUDA 12.4。Python 环境统一使用 uv。官方源码基准提交为 `dc1ce835ea3c6bb7887333c50df7a13497a45c09`。

截至 2026 年 10 月 20 日，已在 dpgs 服务器原生工作区继续配置。实际 GPU 为 RTX 4080 SUPER 32760 MiB，CUDA 12.4 工具链已修复；uv 依赖与扩展已实际安装通过，scan24 训练和 CD 验收已完成，结果见 RESULTS.md。最新训练、评估和产物状态以 STATUS.md 及 logs 为准，命令预演不代表实验成功。

## 一 服务器环境

32GB 显存预计足够顺序复现。论文报告的约 3GB 是训练阶段的用量，不能代表 MASt3R、法线预测或网格提取峰值。先在 scan24 测量，再安排全量场景。实机型号和 32760 MiB 显存已核对。

建议 Ubuntu 22.04，配备 CUDA 12.4 **devel** 工具链，确保有 nvcc、gcc、g++、Git。uv 不负责安装驱动或完整 CUDA 编译器；nvidia-smi 的 CUDA 版本也不代表系统已经有 nvcc。

当前服务器先在项目根目录执行 `source repro/env.sh` 选择 CUDA 12.4、uv 与线程限额，然后执行：

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvcc --version
uv --version
bash repro/setup_uv.sh
source .venv/bin/activate
```

pyproject.toml 声明依赖，uv.lock 固定解析结果。仅 torch 与 torchvision 使用官方 cu124 索引，其他包使用 PyPI。先同步基础环境，再编译 diff-plane-rasterization、simple-knn、PyTorch3D、ASMK 与 CroCo 扩展。安装成功后 repro/records 保存实际依赖与硬件记录。

兼容处理：使用 mmcv-lite 2.2.0，避免编译未使用的 MMCV 算子；将 Metric3D 的 collect_env 导入迁移到 MMEngine；PyTorch3D 0.7.8 用官方版本提交固定源码。训练损失和优化代码没有修改。不要混装 Metric3D 子目录的旧依赖，它固定了另一套 PyTorch 2.0.1 环境。

扩展通过 `uv pip install --no-build-isolation` 安装，随后使用已激活环境的 python 运行。若重新执行 uv sync，需再次运行 setup_uv.sh 以恢复项目清单外安装的本地扩展。不要把安装日志中的成功当成 CUDA 核函数已经通过训练验证。

## 二 权重与数据

第一场景选择 DTU scan24、small_overlap、COLMAP 初始化。训练路径不需先运行 MASt3R；环境检查仍覆盖完整工具链。

将 [Metric3D 权重 metric_depth_vit_large_800k.pth](https://drive.google.com/file/d/1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr/view) 放入 Metric3D/weight。下载作者的 [DTU_sparse2.zip](https://drive.google.com/file/d/11NBPiVAnjB7HxMIE56rl_2Zk1it0IkWp/view)，并从 [DTU 官方站点](https://roboimagedata.compute.dtu.dk/) 准备几何真值。评估不仅需要点云，还需要 ObsMask 与 Plane 文件。

```text
/data/DTU/
  small_overlap/scan24/
    images/
    sparse/0/cameras.bin
    sparse/0/images.bin
    dense/fused.ply
    cameras.npz
    split.json              相机包含完整视图时指定三个训练视图
    normals_metric3d/*.npy   prepare 阶段生成
  large_overlap/...
/data/DTU_GT/
  ObsMask/ObsMask24_10.mat
  ObsMask/Plane24.mat
  Points/stl/stl024_total.ply
```

MASt3R 初始化和 TNT 还需从 [官方 README](https://github.com/hanl2010/DP-GS#usage) 下载以下文件，放到 mast3r/checkpoints：

- MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
- MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth
- MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl

## 三 首个场景

```bash
python repro/run_dtu.py --data-root /data/DTU --gt-root /data/DTU_GT --scans 24 --dry-run
python repro/run_dtu.py --data-root /data/DTU --gt-root /data/DTU_GT --scans 24
```

dry-run 只打印命令，不检查数据或 GPU。真实执行按顺序检查或生成法线、训练 10000 次、提取网格、计算 CD；失败即停止。默认拒绝覆盖已有训练或评估结果，并检查恰好三个有效训练视图。

有效的作者预计算法线默认复用，`--recompute-normals` 可强制预测。

可用 `--stage prepare`、`--stage train`、`--stage mesh`、`--stage eval` 分段运行。输出位于 output/DTU/scan24_colmap_sfm_small_overlap，最终应包含高斯点云、train/ours_10000/fuse_post.ply 和同目录的 results.json。

验收时检查法线为有限的 H×W×3 数组、训练损失实际启用、网格非空且合理、CD 计算成功。论文 Table 1 的 scan24 COLMAP 结果为 1.45，MASt3R 结果为 0.57，应对照相应行。若偏差明显，先核对视图、尺度、初始化、网格后处理和评估协议。

## 四 全部公开基准

DTU 使用 15 个扫描。先完成 small_overlap 的 COLMAP 版本，再切换 MASt3R。

```bash
python repro/run_dtu.py --data-root /data/DTU --gt-root /data/DTU_GT \
  --scans 24 37 40 55 63 65 69 83 97 105 106 110 114 118 122
python repro/summarize_dtu.py

python repro/run_dtu.py --data-root /data/DTU --gt-root /data/DTU_GT \
  --sfm-type mast3r_sfm --scans 24 37 40 55 63 65 69 83 97 105 106 110 114 118 122
python repro/summarize_dtu.py --sfm-type mast3r_sfm
```

对应论文均值分别为 1.02 与 0.85。汇总入口要求全部 15 场景都有有效指标才报告完整均值。进一步复现可用 `--data-type large_overlap`，它沿用官方对应的传播次数与平滑参数。MASt3R 入口仅选择训练视图进行初始化，避免把其他视图用于几何估计。

TNT 使用 Truck、Caterpillar、Barn、Ignatius，每个测试 5、10、20 视图，共 12 个实验。原始场景目录须具有 images 和已校准的 COLMAP sparse/0，不应只放图片。数据与评估材料来自 [Tanks and Temples](https://www.tanksandtemples.org/)。

```bash
python repro/run_tnt.py --data-root /data/TNT --scenes Truck --views 5 --dry-run
python repro/run_tnt.py --data-root /data/TNT --scenes Truck --views 5
python repro/run_tnt.py --data-root /data/TNT
```

TNT 入口跟随官方脚本生成初始化、预测法线、训练和提取网格。官方 TNT 命令没有启用 opacity_decay，且将深度平滑起点放在训练终点；本入口保留这些行为。改变设置应另开实验，避免混用官方配置与改动后的指标。

TNT F1 仍需独立评估环境。官方 scripts/run_tnt_eval.py 写死作者本机 Python 路径，并要求 Python 3.7 与 Open3D 0.9；它与训练环境的 Open3D 0.19 接口不兼容。后续需单独建立 uv 评估环境，或迁移后验证数值一致性。目前未提供未经验证的评估器迁移，因此网格生成完成不等于 F1 复现完成。

论文自采的 5 个场景在官方 README 中尚未找到下载入口。公开基准复现与论文所有实验复现须区分，后者还包括自采数据、补充实验及消融。

## 五 验证边界

已完成源码检查、Python 与 Bash 语法检查、命令预演和学习文档。依赖解析及其结果见 STATUS.md。仍需服务器完成依赖安装、CUDA 编译、权重和数据检查、scan24 CD 与显存记录，再扩展到全场景。真实实验应保存标准输出、环境快照、耗时和指标文件。

参考：[DP-GS 论文](https://arxiv.org/html/2607.03765v1)、[官方仓库](https://github.com/hanl2010/DP-GS)、[uv 文档](https://docs.astral.sh/uv/)、[MMEngine collect_env](https://mmengine.readthedocs.io/en/latest/api/generated/mmengine.utils.dl_utils.collect_env.html)、[NVIDIA 规格](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4080-family/)。

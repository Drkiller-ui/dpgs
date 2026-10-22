# DP GS 服务器交接记录

> 下文是迁移时的历史快照。2026 年 10 月 20 日已在原生远端会话继续，旧 PID 均已失效；CUDA 12.4 的 dpkg 配置已正常完成，uv 环境与扩展已就绪，scan24 训练和 CD 验收已完成。最新实机状态见 STATUS.md，不要根据下文历史 PID 发送信号。

记录时间：2026 年 10 月 20 日。用户最新要求：先在服务器安装 Codex，再通过桌面端的原生 SSH 连接在服务器上工作。当前本地聊天已停止继续配置训练环境，没有启动训练。

## 位置与硬件

- SSH 别名：dpgs，用户 root。
- 项目目录：/root/autodl-tmp/DP-GS。
- Ubuntu 22.04.5，RTX 4080 SUPER，32760 MiB 显存，驱动 595.91.07。
- /root/autodl-tmp 是 50GB 本地数据盘；/root/autodl-fs 是 200GB 持久数据目录。
- 官方源码提交：dc1ce835ea3c6bb7887333c50df7a13497a45c09。
- pyproject.toml、uv.lock 与 repro 目录已从本地同步。

## Codex 连接准备

官方安装器已安装 Codex CLI 0.163.0，路径 /root/.local/bin/codex。另提供 /usr/local/bin/codex 链接，使 SSH 登录 shell 能发现命令。登录正在通过 codex login --device-auth 由用户授权，必须用 codex login status 再确认成功。

桌面端后续选择 设置 > 连接 > SSH > dpgs，然后添加项目目录 /root/autodl-tmp/DP-GS。尚未将当前本地聊天迁移到远端，不要把 CLI 安装完成视为桌面端已经连接。

## 已完成与下载文件

uv 已安装在 /root/.local/bin/uv。尚未运行 uv sync、CUDA 扩展编译、训练或几何指标计算。

作者的处理数据 ZIP 已下载到 /root/autodl-fs/dpgs/downloads/DTU_sparse2.zip，大小约 1.19GB。scan24 的 small_overlap 与 large_overlap 已提取到项目 data/DTU 目录。压缩包中包含三个训练视图、COLMAP 点云、MASt3R 点云和 normals_metric3d 文件，可先检查已有法线再决定是否重新预测。

Metric3D 权重已下载到 Metric3D/weight/metric_depth_vit_large_800k.pth，约 1.65GB。完整性尚未通过模型加载验证。

官方几何真值通过 HTTP Range 从 SampleSet.zip 与 Points.zip 读取。data/DTU_GT/ObsMask 下 scan24 的观测掩码与平面文件已提取。Points/stl/stl024_total.ply 可能仍为 .partial，需确认完整性后使用。

## 暂停的进程

为切换到远端 Codex，已发送 SIGSTOP 暂停下列进程，尚未删除任何下载文件。PID 只能用于当前这次服务器会话；后续操作必须先核实进程命令，防止 PID 被复用。

| PID | 命令 | 状态 |
| --- | --- | --- |
| 1516 | bash /root/bootstrap-dpgs.sh | 暂停 |
| 1839 | apt-get install CUDA 12.4 编译组件 | 暂停 |
| 1389 | uv tool run gdown 运行 fetch_assets.py | 暂停 |
| 1428 | python fetch_assets.py | 暂停 |

apt 可能仍持有包管理锁。不要删除锁文件或并发启动另一个包管理器。核对上述命令仍一致后，可由远端 Codex在继续环境配置时发送 SIGCONT 恢复。恢复 bootstrap 后会依次完成 CUDA 12.4 安装和 repro/setup_uv.sh；恢复 asset 进程会继续剩余真值提取。

日志位于 /root/bootstrap-dpgs.log 与 /root/dpgs-assets.log。资产日志包含下载进度的回车字符，查看时建议只筛选 READY、Extracted、Error 等信息。

## 环境配置要点

镜像已有 CUDA 13.0，DP-GS 指定的 PyTorch 2.4.1 使用 cu124，因此计划并存安装 CUDA 12.4 的编译器与必要头文件。NVIDIA apt 源已写入 /etc/apt/sources.list.d/dpgs-cuda.list。此次只安装工具链与开发库，不安装或升级主机 GPU 驱动。

后续环境变量：CUDA_HOME=/usr/local/cuda-12.4，PATH 加入该目录的 bin 与 /root/.local/bin，TORCH_CUDA_ARCH_LIST=8.9，MAX_JOBS=8，UV_CACHE_DIR=/root/autodl-tmp/dpgs-cache/uv。

repro/setup_uv.sh 负责 uv 环境和原生扩展。repro/patch_metric3d.py 将旧的 collect_env 导入迁移到 MMEngine；该操作尚需在服务器执行。没有修改训练损失或优化算法。

## 后续目标

在远端聊天继续用户原任务时，先核实并完成环境与数据，再以 scan24、small_overlap、COLMAP 初始化运行 10000 次训练、网格提取和 CD 评估。入口为 repro/run_dtu.py，数据路径为项目 data/DTU 与 data/DTU_GT。先完成一场景验收，再扩展 15 个 DTU 场景与 TNT。真实成功应有训练日志、非空网格和 results.json，不能以 dry-run 或环境安装成功代替。

repro/run_logged.sh 可用于 nohup 后台任务，记录命令、输出、退出码和 GPU 用量。当前没有后台训练任务。

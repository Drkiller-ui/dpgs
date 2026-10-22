# TNT 独立旧版评估环境

此目录使用 uv、Python 3.10.20、NumPy 1.23.5 和从官方源码构建的 Open3D 0.9.0.0，与训练环境的 Open3D 0.19 分离。

Open3D 源码版本为 v0.9.0，提交 `e5ba382ff991b4af74c8f7e4a66167fcabf26c5f`。几何核心和原始 pybind11 子模块均未修改。为支持 GCC 11，仅通过 `-Wno-error=maybe-uninitialized` 将一个旧代码警告恢复为警告；优化级别仍为 Release。Notebook 仅是原 wheel 的声明依赖，没有启动服务。

安装已构建的 wheel 并验证环境：

```bash
uv sync --project repro/tnt_eval_env --frozen
uv pip check --python repro/tnt_eval_env/.venv/bin/python
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 repro/tnt_eval_env/.venv/bin/python repro/check_tnt_evaluator.py --expected-version 0.9
```

`wheels/` 保存实际构建的 cp310 Linux wheel，其来源、SHA256 与构建配置见 `../records/tnt_legacy_build.json`。构建源目录位于 `/root/autodl-tmp/dpgs-builds/Open3D-0.9`，日志由 `logs/tnt_legacy_build.latest` 指向。

自测使用解析几何验证同点集、覆盖不完整、离群点、严格距离阈值、相似变换 ICP 及原轨迹 RANSAC。原始 `scripts/tnt_eval` 评估代码保留。`tnt_evaluator_selftest_0.9.json` 不是 DP-GS 实验结果，自测不能代替真实 TNT 场景训练、网格和 F1 验收。

真实场景产物和真值齐备后执行：

```bash
python repro/eval_tnt.py --mesh /path/to/fuse_post.ply --scene Truck --gt-root data/TNT_GT
```

默认轨迹与官方脚本一致，使用真值目录的参考 COLMAP 轨迹；若网格采用其他 COLMAP 坐标系，应通过 `--trajectory` 指定该坐标系下的完整相机轨迹。真实结果存入网格目录的 `result.csv` 和 `repro_tnt_evaluation.json`。

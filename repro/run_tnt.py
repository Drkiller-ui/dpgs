"""Sequential TNT preprocessing/training/meshing, matching upstream script flags."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCENES = ['Truck', 'Caterpillar', 'Barn', 'Ignatius']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, default=ROOT / 'output')
    p.add_argument('--scenes', nargs='+', choices=SCENES, default=SCENES)
    p.add_argument('--views', nargs='+', type=int, choices=[5, 10, 20], default=[5, 10, 20])
    p.add_argument('--stage', choices=['prepare', 'train', 'mesh', 'all'], default='all')
    p.add_argument('--gpu', default='0')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    a.data_root = a.data_root.resolve()
    a.output_root = a.output_root.resolve()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu)
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    stages = ['prepare', 'train', 'mesh'] if a.stage == 'all' else [a.stage]

    def need(path):
        if not a.dry_run and not path.exists():
            raise FileNotFoundError(path)

    def run(cmd, cwd=ROOT):
        cmd = [str(x) for x in cmd]
        print(f'[{cwd}] {shlex.join(cmd)}', flush=True)
        if not a.dry_run:
            subprocess.run(cmd, cwd=cwd, env=env, check=True)

    for views in a.views:
        for name in a.scenes:
            scene = a.data_root / name
            prepared = scene / f'mast3r_{views}views'
            model = a.output_root / 'TNT' / f'{name}_mast3r_sfm_{views}views'
            if 'prepare' in stages:
                need(scene / 'images')
                need(scene / 'sparse/0/cameras.bin')
                need(scene / 'sparse/0/images.bin')
                run([sys.executable, 'run_mast3r.py', '--scene_path', scene, '--output_dir', prepared,
                     '--n_images', views], ROOT / 'mast3r')
                run([sys.executable, 'test.py', '--data_dir', prepared, '--get_normal', '--resolution', 2], ROOT / 'Metric3D')
            if 'train' in stages:
                need(prepared / 'points.ply')
                need(prepared / 'normals_metric3d')
                if not a.dry_run and (model / 'cfg_args').exists() and not a.force:
                    raise FileExistsError(f'Existing run: {model}; choose another output root or --force')
                # Preserve upstream TNT configuration: no opacity decay; depth
                # smoothing begins after the training endpoint.
                run([sys.executable, 'train.py', '-s', prepared, '-m', model, '--dataset_name', 'TNT',
                     '--resolution', 2, '--iterations', 10000, '--sfm_type', 'mast3r_sfm',
                     '--depth_smooth_iter_start', 10000])
                need(model / 'point_cloud/iteration_10000/point_cloud.ply')
            if 'mesh' in stages:
                need(model / 'cfg_args')
                run([sys.executable, 'render_and_mesh_extract.py', '--model_path', model,
                     '--dataset_name', 'TNT', '--sfm_type', 'mast3r_sfm', '--skip_train', '--skip_test',
                     '--iteration', 10000])
                need(model / 'train/ours_10000/fuse_post.ply')
    print('Dry run only; no validation performed.' if a.dry_run else 'TNT stages complete; legacy F1 evaluation is separate.')


if __name__ == '__main__':
    main()

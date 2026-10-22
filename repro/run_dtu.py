"""Sequential DTU prepare/train/mesh/eval driver using upstream algorithms."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCANS = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]


def validate_normals(scene, names):
    import numpy as np
    for name in names:
        prior = scene / 'normals_metric3d' / f'{name}.npy'
        normal = np.load(prior)
        if normal.ndim != 3 or normal.shape[-1] != 3 or not np.isfinite(normal).all():
            raise ValueError(f'Invalid normal map: {prior}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='Parent of small_overlap/large_overlap')
    parser.add_argument('--gt-root', type=Path, help='Directory containing ObsMask/ and Points/')
    parser.add_argument('--scans', type=int, nargs='+', default=[24], choices=SCANS)
    parser.add_argument('--data-type', choices=['small_overlap', 'large_overlap'], default='small_overlap')
    parser.add_argument('--sfm-type', choices=['colmap_sfm', 'mast3r_sfm'], default='colmap_sfm')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'output')
    parser.add_argument('--stage', choices=['prepare', 'train', 'mesh', 'eval', 'all'], default='all')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--dry-run', action='store_true', help='Print commands only; data is not checked')
    parser.add_argument('--force', action='store_true', help='Allow rerunning stages in an existing output directory')
    parser.add_argument('--recompute-normals', action='store_true', help='Recompute even if all supplied normal maps are valid')
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.gt_root:
        args.gt_root = args.gt_root.resolve()
    if args.stage in ['eval', 'all'] and not args.gt_root:
        parser.error('--gt-root is required for eval/all (use --stage mesh to stop before evaluation)')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    # Nested upstream commands call `python`; make it the same uv interpreter.
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    stages = ['prepare', 'train', 'mesh', 'eval'] if args.stage == 'all' else [args.stage]
    journal = []

    def run(command, cwd=ROOT):
        cmd = [str(x) for x in command]
        print(f'[{cwd}] {shlex.join(cmd)}', flush=True)
        if args.dry_run:
            return
        started = time.time()
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
        journal.append({'command': cmd, 'cwd': str(cwd), 'seconds': time.time() - started})

    def need(path):
        if not args.dry_run and (not path.exists() or (path.is_file() and path.stat().st_size == 0)):
            raise FileNotFoundError(path)

    for scan in args.scans:
        scene = args.data_root / args.data_type / f'scan{scan}'
        model = args.output_root / 'DTU' / f'scan{scan}_{args.sfm_type}_{args.data_type}'
        meshdir = model / 'train' / 'ours_10000'
        mesh = meshdir / 'fuse_post.ply'
        need(scene / 'images')
        need(scene / 'sparse/0/cameras.bin')
        need(scene / 'sparse/0/images.bin')
        if not args.dry_run:
            spec = importlib.util.spec_from_file_location('colmap_loader', ROOT / 'scene/colmap_loader.py')
            loader = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loader)
            names = [Path(x.name).stem for x in loader.read_extrinsics_binary(scene / 'sparse/0/images.bin').values()]
            splitfile = scene / 'split.json'
            train = json.loads(splitfile.read_text())['train'] if splitfile.exists() else names
            if len(train) != 3 or not set(train).issubset(names):
                raise ValueError(f'{scene}: expected exactly 3 valid training views; inspect split.json/COLMAP')
            for name in names:
                matches = [p for p in (scene / 'images').iterdir() if p.stem == name]
                if len(matches) != 1:
                    raise ValueError(f'Expected one image for COLMAP view {name}: {matches}')
        if 'prepare' in stages:
            if args.sfm_type == 'mast3r_sfm':
                splitfile = scene / 'split.json'
                # Upstream --use_all_images ignores split.json. Explicitly select only training views.
                indexes = []
                if not args.dry_run:
                    ordered_names = sorted(p.stem for p in (scene / 'images').iterdir()
                                           if p.suffix.lower() in ['.jpg', '.jpeg', '.png'])
                    indexes = [ordered_names.index(name) for name in train]
                need(ROOT / 'mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth')
                run([sys.executable, 'run_mast3r.py', '--scene_path', scene, '--output_dir', scene / 'mast3r',
                     '--image_idx', *(indexes or [0, 1, 2])], ROOT / 'mast3r')
            reuse_normals = False
            if not args.dry_run and not args.recompute_normals:
                try:
                    validate_normals(scene, names)
                    reuse_normals = True
                except (OSError, ValueError):
                    pass
            if reuse_normals:
                print(f'Reusing validated supplied normal maps: {scene}', flush=True)
            else:
                need(ROOT / 'Metric3D/weight/metric_depth_vit_large_800k.pth')
                run([sys.executable, 'test.py', '--data_dir', scene, '--get_normal', '--resolution', '2'], ROOT / 'Metric3D')
        if any(stage in stages for stage in ['train', 'mesh']):
            if not args.dry_run:
                validate_normals(scene, names)
            need(scene / ('dense/fused.ply' if args.sfm_type == 'colmap_sfm' else 'mast3r/points.ply'))
        if 'train' in stages:
            if not args.dry_run and (model / 'cfg_args').exists() and not args.force:
                raise FileExistsError(f'{model} already has a run; use another output root or --force')
            smooth, propagation = ('0.03', '10') if args.data_type == 'small_overlap' else ('0.0', '5')
            run([sys.executable, 'train.py', '-s', scene, '-m', model, '--dataset_name', 'DTU',
                 '--resolution', '2', '--iterations', '10000', '--sfm_type', args.sfm_type,
                 '--opacity_decay', '--depth_smooth_weight', smooth, '--depth_propagation_iters', propagation])
            need(model / 'point_cloud/iteration_10000/point_cloud.ply')
        if 'mesh' in stages:
            clusters = 4 if scan in [37, 55, 63, 97] else 1
            run([sys.executable, 'render_and_mesh_extract.py', '--model_path', model,
                 '--dataset_name', 'DTU', '--sfm_type', args.sfm_type, '--skip_test', '--skip_train',
                 '--iteration', '10000', '--num_cluster', clusters, '--voxel_size', '0.002'])
            need(mesh)
        if 'eval' in stages:
            need(mesh)
            need(scene / 'cameras.npz')
            for rel in [f'ObsMask/ObsMask{scan}_10.mat', f'ObsMask/Plane{scan}.mat',
                        f'Points/stl/stl{scan:03}_total.ply']:
                need(args.gt_root / rel)
            result = meshdir / 'results.json'
            if not args.dry_run and result.exists() and not args.force:
                raise FileExistsError(f'{result} exists; use --force to recompute')
            run([sys.executable, 'repro/eval_dtu.py', '--mesh', mesh, '--scan', scan,
                 '--scene', scene, '--gt-root', args.gt_root, '--data-type', args.data_type])
            need(result)
        if not args.dry_run:
            model.mkdir(parents=True, exist_ok=True)
            (model / f'repro_{args.stage}_record.json').write_text(json.dumps({
                'upstream_commit': (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
                                    if (ROOT / '.git').exists()
                                    else (ROOT / 'repro/upstream_commit.txt').read_text().strip()),
                'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'commands': journal,
            }, indent=2), encoding='utf-8')
            journal.clear()
    print('Dry run complete; no data or GPU validation performed.' if args.dry_run else 'Requested stages complete.')


if __name__ == '__main__':
    main()

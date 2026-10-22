"""Record input/output integrity without changing the upstream experiment."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from plyfile import PlyData
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]


def cloud(path):
    data = PlyData.read(path)['vertex'].data
    xyz = np.column_stack([data[key] for key in ('x', 'y', 'z')])
    assert len(xyz) > 0 and np.isfinite(xyz).all(), path
    for name in data.dtype.names:
        if np.issubdtype(data[name].dtype, np.floating):
            assert np.isfinite(data[name]).all(), f'{path}: non-finite {name}'
    return {'points': len(xyz), 'minimum': xyz.min(0).tolist(), 'maximum': xyz.max(0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', required=True, type=int)
    parser.add_argument('--data-type', default='small_overlap', choices=['small_overlap', 'large_overlap'])
    parser.add_argument('--sfm-type', default='colmap_sfm', choices=['colmap_sfm', 'mast3r_sfm'])
    parser.add_argument('--stage', required=True, choices=['inputs', 'outputs'])
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/DTU')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'output')
    parser.add_argument('--gt-root', type=Path, default=ROOT / 'data/DTU_GT')
    args = parser.parse_args()
    scene = args.data_root.resolve() / args.data_type / f'scan{args.scan}'
    model = args.output_root.resolve() / 'DTU' / f'scan{args.scan}_{args.sfm_type}_{args.data_type}'
    report = {'arguments': {key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()}}
    if args.stage == 'inputs':
        spec = importlib.util.spec_from_file_location('colmap_loader', ROOT / 'scene/colmap_loader.py')
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        views = list(loader.read_extrinsics_binary(scene / 'sparse/0/images.bin').values())
        expected = ['0022', '0025', '0028'] if args.data_type == 'small_overlap' else ['0023', '0024', '0033']
        assert sorted(Path(view.name).stem for view in views) == expected
        report['views'] = []
        for view in views:
            assert np.isfinite(view.qvec).all() and np.isfinite(view.tvec).all()
            with Image.open(scene / 'images' / view.name) as img:
                width, height = img.size
            normal = np.load(scene / 'normals_metric3d' / (Path(view.name).stem + '.npy'))
            assert normal.shape == (height // 2, width // 2, 3) and np.isfinite(normal).all()
            report['views'].append({'name': view.name, 'image_size': [width, height], 'normal_shape': list(normal.shape)})
        init = scene / ('dense/fused.ply' if args.sfm_type == 'colmap_sfm' else 'mast3r/points.ply')
        report['initialization'] = cloud(init)
        gt = args.gt_root.resolve()
        report['ground_truth'] = cloud(gt / f'Points/stl/stl{args.scan:03}_total.ply')
        mask = loadmat(gt / f'ObsMask/ObsMask{args.scan}_10.mat')
        assert {'ObsMask', 'BB', 'Res'}.issubset(mask)
        assert np.isfinite(loadmat(gt / f'ObsMask/Plane{args.scan}.mat')['P']).all()
        report['observation_mask_shape'] = list(mask['ObsMask'].shape)
        with np.load(scene / 'cameras.npz') as cams:
            assert all(np.isfinite(cams[f'{prefix}_{i}']).all() for prefix in ['world_mat', 'scale_mat'] for i in range(49))
    else:
        report['gaussians'] = cloud(model / 'point_cloud/iteration_10000/point_cloud.ply')
        meshdir = model / 'train/ours_10000'
        mesh = o3d.io.read_triangle_mesh(str(meshdir / 'fuse_post.ply'))
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        assert len(vertices) > 0 and len(faces) >= 50 and np.isfinite(vertices).all()
        assert faces.min() >= 0 and faces.max() < len(vertices)
        report['mesh'] = {'vertices': len(vertices), 'triangles': len(faces)}
        values = json.loads((meshdir / 'results.json').read_text())
        assert all(np.isfinite(values[key]) and values[key] >= 0 for key in ['overall', 'mean_d2s', 'mean_s2d'])
        assert np.isclose(values['overall'], (values['mean_d2s'] + values['mean_s2d']) / 2)
        report['metrics'] = values
    target = ROOT / 'repro/records' / f'scan{args.scan}_{args.sfm_type}_{args.data_type}_{args.stage}.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Run the unchanged TNT metric code with the isolated Open3D 0.9 interpreter."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--scene', choices=['Truck', 'Caterpillar', 'Barn', 'Ignatius'], required=True)
    parser.add_argument('--gt-root', type=Path, required=True)
    parser.add_argument('--trajectory', type=Path,
                        help='COLMAP trajectory in the mesh coordinate frame; default: official reference trajectory')
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--force', action='store_true', help='Allow recomputing an existing evaluation')
    args = parser.parse_args()
    legacy = ROOT / 'repro/tnt_eval_env/.venv/bin/python'
    mesh = args.mesh.resolve()
    dataset = args.gt_root.resolve() / args.scene
    trajectory = (args.trajectory or dataset / f'{args.scene}_COLMAP_SfM.log').resolve()
    output = (args.out_dir or mesh.parent).resolve()
    if not args.force and any((output / name).exists() for name in ['result.csv', 'repro_tnt_evaluation.json']):
        raise FileExistsError(f'Existing evaluation in {output}; inspect it or pass --force')
    for path in [legacy, mesh, trajectory, dataset / f'{args.scene}.ply',
                 dataset / f'{args.scene}.json', dataset / f'{args.scene}_trans.txt',
                 dataset / f'{args.scene}_COLMAP_SfM.log']:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MPLBACKEND='Agg')
    check = ('import open3d,numpy as np,sys; '
             'assert open3d.__version__ == "0.9.0.0", open3d.__version__; '
             'mesh=open3d.io.read_triangle_mesh(sys.argv[1]); '
             'assert len(mesh.vertices)>0 and len(mesh.triangles)>0; '
             'assert np.isfinite(np.asarray(mesh.vertices)).all()')
    subprocess.run([str(legacy), '-c', check, str(mesh)],
                   env=env, check=True)
    command = [str(legacy), str(ROOT / 'scripts/tnt_eval/run.py'), '--dataset-dir', str(dataset),
               '--traj-path', str(trajectory), '--ply-path', str(mesh), '--out-dir', str(output)]
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    with (output / 'result.csv').open() as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 1:
        raise ValueError('Expected one metric row')
    metrics = {name: float(rows[0][name]) for name in ['precision', 'recall', 'fscore']}
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in metrics.values()):
        raise ValueError(metrics)
    record = {'scene': args.scene, 'mesh': str(mesh), 'dataset': str(dataset),
              'trajectory': str(trajectory), 'open3d_version': '0.9.0.0',
              'metrics': metrics, 'f1_percent': 100 * metrics['fscore'], 'command': command}
    (output / 'repro_tnt_evaluation.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()

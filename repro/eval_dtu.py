"""Use the upstream culling and metric code, propagate subprocess failures."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh', required=True, type=Path)
    p.add_argument('--scan', required=True, type=int)
    p.add_argument('--scene', required=True, type=Path)
    p.add_argument('--gt-root', required=True, type=Path)
    p.add_argument('--data-type', choices=['small_overlap', 'large_overlap'], required=True)
    a = p.parse_args()
    scriptdir = Path(__file__).resolve().parents[1] / 'scripts/dtu_eval'
    sys.path.insert(0, str(scriptdir))
    from evaluate_single_scene import cull_scan
    culled = a.mesh.parent / 'culled_mesh.ply'
    views = [22, 25, 28] if a.data_type == 'small_overlap' else [23, 24, 33]
    cull_scan(str(a.scan), str(a.mesh), str(culled), str(a.scene), view_list=views)
    subprocess.run([sys.executable, str(scriptdir / 'eval.py'), '--data', str(culled),
                    '--scan', str(a.scan), '--mode', 'mesh', '--dataset_dir', str(a.gt_root),
                    '--vis_out_dir', str(a.mesh.parent)], check=True)


if __name__ == '__main__':
    main()

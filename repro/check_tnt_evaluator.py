"""Analytic evaluator checks; these are NOT DP-GS benchmark results."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/tnt_eval'))
from evaluation import EvaluateHisto, get_f1_score_histo2


def cloud(points):
    result = o3d.geometry.PointCloud()
    result.points = o3d.utility.Vector3dVector(points)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-version', required=True)
    args = parser.parse_args()
    assert o3d.__version__.startswith(args.expected_version), o3d.__version__
    volume = o3d.visualization.SelectionPolygonVolume()
    volume.orthogonal_axis = 'Z'
    volume.axis_min, volume.axis_max = -1, 20
    volume.bounding_polygon = o3d.utility.Vector3dVector(
        [[-1, -1, 0], [20, -1, 0], [20, 20, 0], [-1, 20, 0]])
    grid = np.stack(np.meshgrid(np.arange(3), np.arange(3), np.arange(3)), -1).reshape(-1, 3) * 0.2
    outliers = np.column_stack([np.arange(9) * 0.2 + 10, np.full(9, 10), np.full(9, 10)])
    cases = [('identical', grid, (1, 1, 1)),
             ('partial_coverage', grid[:18], (1, 2 / 3, 0.8)),
             ('outliers', np.concatenate([grid + [0.001, 0.002, 0], outliers]), (0.75, 1, 6 / 7))]
    results = []
    with tempfile.TemporaryDirectory(prefix='dpgs-tnt-eval-') as directory:
        for name, points, expected in cases:
            values = EvaluateHisto(cloud(points), cloud(grid), np.eye(4), volume,
                                   0.005, 0.01, directory, 5, name)[:3]
            assert np.allclose(values, expected, rtol=0, atol=1e-12), (name, values, expected)
            results.append({'case': name, 'actual': list(values), 'expected': list(expected)})
        threshold = 0.01
        values = get_f1_score_histo2(threshold, directory, 5, [0, threshold, 2 * threshold],
                                    [0, threshold, 2 * threshold])[:3]
        assert np.allclose(values, [1 / 3] * 3, rtol=0, atol=1e-12)
        results.append({'case': 'strict_threshold', 'actual': list(values), 'expected': [1 / 3] * 3})
    # Recover a known similarity transform using the evaluator's scaling mode.
    points = np.random.default_rng(17).uniform(0, 1, (200, 3))
    angle = 0.03
    transform = np.eye(4)
    transform[:3, :3] = 1.02 * np.array([[np.cos(angle), -np.sin(angle), 0],
                                       [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    transform[:3, 3] = [0.02, -0.01, 0.01]
    target = points @ transform[:3, :3].T + transform[:3, 3]
    registration = o3d.registration if hasattr(o3d, 'registration') else o3d.pipelines.registration
    fit = registration.registration_icp(cloud(points), cloud(target), 0.2, np.eye(4),
        registration.TransformationEstimationPointToPoint(True),
        registration.ICPConvergenceCriteria(1e-9, 1e-9, 100))
    error = float(np.max(np.abs(fit.transformation - transform)))
    assert error < 1e-6, error
    trajectory_error = None
    if o3d.__version__.startswith('0.9'):
        from registration import trajectory_alignment
        from trajectory_io import CameraPose

        def trajectory(centers):
            poses = []
            for i, center in enumerate(centers):
                pose = np.eye(4)
                pose[:3, 3] = center
                poses.append(CameraPose([i, i, 0], pose))
            return poses

        estimated = trajectory_alignment('unused_for_short_trajectory', trajectory(points[:20]),
                                         trajectory(target[:20]), np.eye(4), 'analytic_fixture')
        trajectory_error = float(np.max(np.abs(estimated - transform)))
        assert trajectory_error < 1e-6, trajectory_error
    report = {'not_a_dp_gs_result': True, 'open3d_version': o3d.__version__,
              'python': sys.version, 'metric_cases': results,
              'similarity_icp_max_error': error,
              'legacy_trajectory_ransac_max_error': trajectory_error,
              'scope': 'Known geometry and original EvaluateHisto API checks; no TNT scene reconstruction or benchmark F1 claimed.'}
    target_path = ROOT / 'repro/records' / f'tnt_evaluator_selftest_{args.expected_version}.json'
    target_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

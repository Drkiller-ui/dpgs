"""Verify the completed paired campaign and decode every comparison video."""
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess

from PIL import Image
import imageio_ffmpeg
import numpy as np

from run_dtu import SCANS

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    gallery = ROOT / 'output/MAtCha_comparison'
    summary = read(gallery / 'manifest.json')
    assert summary['complete']
    assert summary['evaluated_scans'] == summary['visualized_scans'] == len(SCANS) == 15
    job = ROOT / (ROOT / 'logs/matcha_colmap.latest').read_text().strip()
    assert (job / 'status').read_text().strip() == 'completed'
    assert (job / 'exit_code').read_text().strip() == '0'
    progress = read(ROOT / 'repro/records/matcha_progress.json')
    assert progress['status'] == 'completed'
    rows = list(csv.DictReader((ROOT / 'output/dtu_matcha_vs_dpgs.csv').open()))
    assert [int(row['scan']) for row in rows] == SCANS
    baseline = {int(r['scan']): float(r['overall']) for r in csv.DictReader(
        (ROOT / 'output/dtu_colmap_sfm_small_overlap.csv').open())}
    checks = []
    for row in rows:
        scan = int(row['scan'])
        folder = ROOT / f'output/MAtCha_DTU/colmap_small_overlap/scan{scan}'
        result = read(folder / 'tetra_meshes/results.json')
        assert float(row['dpgs_cd']) == baseline[scan]
        assert float(row['matcha_cd']) == result['overall']
        assert all(math.isfinite(float(v)) for v in row.values())
        integrity = read(folder / 'integrity.json')
        assert integrity['all_gaussian_float_properties_finite']
        assert integrity['all_mesh_vertices_finite']
        assert integrity['iteration'] == 7000 and integrity['mesh_faces'] > 0
        for name in ['checkpoint', 'mesh']:
            assert Path(integrity[name]).stat().st_size > 0
        init = read(folder / 'colmap_init/init_manifest.json')
        source = ROOT / f'data/DTU/small_overlap/scan{scan}/dense/fused.ply'
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest == init['sha256']
        assert hashlib.sha256((folder / 'colmap_init/sparse/0/points3D.ply').read_bytes()).hexdigest() == digest
        assert init['all_xyz_and_rgb_identical_after_binary_roundtrip']
        assert init['initial_ply_identical_to_dpgs'] and not init['mast3r_executed']
        calibration = read(folder / 'calibration_check.json')
        assert all(v == 0 for camera in calibration.values() for v in camera.values())
        images = gallery / f'scan{scan}'
        manifests = [read(images / method / 'manifest.json') for method in ['dpgs', 'matcha']]
        assert manifests[0]['cameras'] == manifests[1]['cameras']
        assert manifests[0]['static_parameters'] == manifests[1]['static_parameters']
        for manifest in manifests:
            assert manifest['protected_unchanged'] and manifest['camera_projection_alignment_checked']
            assert not manifest['novel_images_have_ground_truth'] and not manifest['novel_mask_used']
        for name in ['training_rgb.png', 'training_foreground.png', 'novel_rgb_mesh.png']:
            with Image.open(images / name) as picture:
                picture.verify()
        # Check the actual exported pixels, so a swapped or mislabeled column
        # cannot pass just because all three PNG files exist.
        with Image.open(images / 'training_rgb.png') as picture:
            board = np.asarray(picture.convert('RGB'))
        for y, view in enumerate(['0022', '0025', '0028']):
            originals = ROOT / f'output/DTU_visuals/colmap_sfm_small_overlap/scan{scan}/gt_raw/{view}.png'
            sources = [originals, images / f'matcha/train_{view}.png', images / f'dpgs/train_{view}.png']
            for x, source in enumerate(sources):
                with Image.open(source) as picture:
                    expected = np.asarray(picture.convert('RGB'))
                h, w = expected.shape[:2]
                left, top = 12 + x * (w + 12), 70 + y * (h + 36)
                np.testing.assert_array_equal(board[top:top+h, left:left+w], expected)
        video = images / 'comparison_motion.mp4'
        decoded = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-v', 'error',
            '-xerror', '-nostdin', '-threads', '2', '-i', str(video), '-map', '0:v:0',
            '-progress', 'pipe:1', '-f', 'null', '-'], text=True, capture_output=True, check=True)
        frames = [int(line.split('=', 1)[1]) for line in decoded.stdout.splitlines() if line.startswith('frame=')]
        assert frames and frames[-1] == 60, (scan, decoded.stdout, decoded.stderr)
        assert not decoded.stderr.strip(), (scan, decoded.stderr)
        checks.append({'scan': scan, 'dpgs_cd': baseline[scan], 'matcha_cd': result['overall'],
                       'original_colmap_sha256': digest, 'camera_poses_identical': True,
                       'training_columns_pixels_verified': ['Original RGB', 'MAtCha', 'DP-GS'],
                       'comparison_pngs_verified': 3, 'comparison_video_decoded_frames': frames[-1]})
        print(f'VERIFIED scan{scan} CD, initialization, cameras, images, 60 video frames', flush=True)
    means = {method: sum(float(r[f'{method}_cd']) for r in rows) / len(rows) for method in ['dpgs', 'matcha']}
    for method, mean in means.items():
        assert math.isclose(mean, summary[f'paired_{method}_mean'], rel_tol=1e-12)
    record = {'verified_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': 'completed', 'scans': SCANS, 'mean_cd_mm': means, 'checks': checks,
              'scope': summary['scope'], 'reproduction_status': summary['reproduction_status'],
              'recovered_failure_record': 'repro/records/matcha_scan110_missing_normals.json'}
    (ROOT / 'repro/records/matcha_completion.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'status': record['status'], 'scans': len(checks), 'means': means}), flush=True)


if __name__ == '__main__':
    main()

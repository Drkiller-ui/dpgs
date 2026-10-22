"""Prepare exactly the DP-GS three input RGB views and 777x581 calibration."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'baselines/MAtCha/mast3r'))
from colmap.read_write_model import (read_cameras_binary, read_images_binary,
                                    write_cameras_binary, write_images_binary)
from run_dtu import SCANS


def stage(scan):
    source = ROOT / f'data/DTU/small_overlap/scan{scan}'
    target = ROOT / f'data/MAtCha_DTU/small_overlap/scan{scan}'
    (target / 'images').mkdir(parents=True, exist_ok=True)
    (target / 'sparse/0').mkdir(parents=True, exist_ok=True)
    cameras = read_cameras_binary(source / 'sparse/0/cameras.bin')
    images = read_images_binary(source / 'sparse/0/images.bin')
    assert sorted(i.name for i in images.values()) == ['0022.png', '0025.png', '0028.png']
    scaled_cameras = {}
    for key, camera in cameras.items():
        assert camera.model == 'PINHOLE'
        assert (camera.width, camera.height) == (1554, 1162)
        params = camera.params * .5
        assert np.allclose(params[2:], [388.5, 290.5])
        scaled_cameras[key] = camera._replace(width=777, height=581, params=params)
    scaled_images = {key: im._replace(xys=im.xys * .5) for key, im in images.items()}
    write_cameras_binary(scaled_cameras, target / 'sparse/0/cameras.bin')
    write_images_binary(scaled_images, target / 'sparse/0/images.bin')
    point_target = target / 'sparse/0/points3D.bin'
    if not point_target.exists():
        point_target.symlink_to((source / 'sparse/0/points3D.bin').resolve())
    files = []
    for name in ['0022.png', '0025.png', '0028.png']:
        # DP-GS also reads RGB separately from the alpha channel for training.
        original = PILImage.open(source / 'images' / name)
        im = PILImage.merge('RGB', tuple(channel.resize((777,581)) for channel in original.split()[:3]))
        path = target / 'images' / name
        im.save(path)
        reference = ROOT / f'output/DTU_visuals/colmap_sfm_small_overlap/scan{scan}/gt_raw/{Path(name).stem}.png'
        if reference.exists():
            assert np.array_equal(np.asarray(im), np.asarray(PILImage.open(reference))), name
        files.append({'name': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    record = {'scan': scan, 'views': [22,25,28], 'resolution': [777,581],
              'source': str(source.resolve()), 'target': str(target.resolve()),
              'training_rgb_matches_saved_dpgs_gt_raw': True, 'images': files,
              'poses': 'Unmodified author COLMAP poses; intrinsics and observations scaled by 0.5',
              'mask': 'RGB only, same DP-GS training image pixels; original masks retained for identical CD evaluation'}
    (target / 'input_manifest.json').write_text(json.dumps(record, indent=2))
    return record


if __name__ == '__main__':
    records = [stage(s) for s in SCANS]
    (ROOT / 'repro/records/matcha_inputs.json').write_text(json.dumps(records, indent=2))
    print(f'Validated {len(records)} scans with identical DP-GS training RGB pixels', flush=True)

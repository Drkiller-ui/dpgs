"""Sequential official MAtCha posed pipeline and identical DP-GS DTU evaluation."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from plyfile import PlyData
from run_dtu import SCANS

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'baselines/MAtCha'
OUT = ROOT / 'output/MAtCha_DTU/colmap_small_overlap'


def finite_ply(path):
    vertex = PlyData.read(path,known_list_len={'face':{'vertex_indices':3}})['vertex'].data
    assert len(vertex) > 0, path
    for name in vertex.dtype.names:
        if vertex[name].dtype.kind == 'f':
            assert np.isfinite(vertex[name]).all(), (path, name)
    return len(vertex)


def verify_cameras(source, sfm):
    sys.path.insert(0,str(CODE / 'mast3r'))
    from colmap.read_write_model import read_images_binary,qvec2rotmat
    original={im.name:im for im in read_images_binary(source/'sparse/0/images.bin').values()}
    result={im.name:im for im in read_images_binary(sfm/'sparse/0/images.bin').values()}
    assert set(original)==set(result)=={'0022.png','0025.png','0028.png'}
    errors={name:{'rotation_max_error':float(np.abs(qvec2rotmat(original[name].qvec)-qvec2rotmat(result[name].qvec)).max()),
                  'translation_max_error':float(np.abs(original[name].tvec-result[name].tvec).max())} for name in original}
    assert max(v for values in errors.values() for v in values.values())<1e-6,errors
    (sfm.parent/'calibration_check.json').write_text(json.dumps(errors,indent=2))
    from PIL import Image
    sizes={p.name:list(Image.open(p).size) for p in (sfm/'images').glob('*.png')}
    (sfm.parent/'preprocessing.json').write_text(json.dumps({'input_resolution':[777,581],
        'official_rectified_cropped_image_sizes':sizes,
        'note':'Native COLMAP input retained; no MASt3R preprocessing or geometry used'},indent=2))


def summarize():
    baseline = {int(row['scan']): float(row['overall']) for row in csv.DictReader((ROOT / 'output/dtu_colmap_sfm_small_overlap.csv').open())}
    rows = []
    for scan in SCANS:
        metric = OUT / f'scan{scan}/tetra_meshes/results.json'
        if not metric.exists():
            continue
        result = json.loads(metric.read_text())
        cd = float(result['overall'])
        assert np.isfinite(cd)
        rows.append({'scan': scan, 'dpgs_cd': baseline[scan], 'matcha_cd': cd,
                     'matcha_minus_dpgs': cd-baseline[scan], 'matcha_mean_d2s': result['mean_d2s'],
                     'matcha_mean_s2d': result['mean_s2d']})
    target = ROOT / 'output/dtu_matcha_vs_dpgs.csv'
    if rows:
        with target.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scans', nargs='+', type=int, choices=SCANS, default=SCANS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    progress = ROOT / 'repro/records/matcha_progress.json'
    if progress.exists():
        history=progress.with_name(f'matcha_progress_{int(time.time())}.json')
        history.write_bytes(progress.read_bytes())
    report = {'method': 'MAtCha', 'commit': subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip(),
              'scans': args.scans, 'views': [22,25,28], 'input_resolution': [777,581],
              'sfm': 'Identical original dense/fused.ply COLMAP cloud as DP-GS; native COLMAP loading and fixed cameras; no MASt3R',
              'charts_alignment_iterations': 1000, 'refinement_iterations': 7000,
              'mesh': 'Official adaptive tetrahedralization, default config, downsample=0.5, interpolated views',
              'evaluation': 'Same original masks, culling, world coordinates, ObsMask, plane, sampling and CD as DP-GS',
              'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'stages': []}

    def save():
        report['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report['completed'] = summarize()
        progress.write_text(json.dumps(report,indent=2))

    def run(scan, stage, command, cwd=CODE):
        entry = {'scan': scan,'stage': stage,'command': [str(s) for s in command], 'cwd':str(cwd), 'status':'running','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        report['current_scan'], report['current_stage'] = scan,stage
        report['stages'].append(entry)
        save()
        folder = OUT / f'scan{scan}'
        log = folder / f'{stage}.log'
        print(f'MATCHA_STAGE scan{scan} {stage} log={log}',flush=True)
        started = time.time()
        with log.open('w') as output:
            proc = subprocess.run([str(s) for s in command],cwd=cwd,stdout=output,stderr=subprocess.STDOUT)
        entry.update(status='completed' if proc.returncode == 0 else 'failed',exit_code=proc.returncode,seconds=time.time()-started)
        save()
        if proc.returncode:
            print(log.read_text(errors='replace')[-10000:],flush=True)
            raise subprocess.CalledProcessError(proc.returncode, command)

    try:
        for scan in args.scans:
            model = OUT / f'scan{scan}'
            model.mkdir(parents=True,exist_ok=True)
            sfm = model / 'colmap_init'
            source = ROOT / f'data/MAtCha_DTU/small_overlap/scan{scan}'
            free = model / 'free_gaussians'
            meshdir = model / 'tetra_meshes'
            mesh = meshdir / 'tetra_mesh_binary_search_7.ply'
            metric = meshdir / 'results.json'
            if not (sfm / 'init_manifest.json').exists():
                run(scan,'colmap_input',[ROOT/'.venv/bin/python',ROOT/'repro/prepare_matcha_colmap.py','--scan',scan,'--output',sfm],cwd=ROOT)
            verify_cameras(source,sfm)
            if not (sfm / 'charts_data.npz').exists():
                run(scan,'alignment',[sys.executable,ROOT/'repro/align_matcha_colmap.py','--scene',sfm])
            point = free / 'point_cloud/iteration_7000/point_cloud.ply'
            if not point.exists():
                if (free / 'cfg_args').exists():
                    raise RuntimeError(f'Partial training found at {free}; preserve and explicitly restart from initialization')
                run(scan,'refinement',[sys.executable,'scripts/refine_free_gaussians.py','--mast3r_scene',sfm,'--output_path',free,'--config','default'])
            count = finite_ply(point)
            print(f'MATCHA_MODEL scan{scan} gaussians={count} finite=True',flush=True)
            if not mesh.exists():
                run(scan,'mesh',[sys.executable,'scripts/extract_tetra_mesh.py','-s',sfm,'-m',free,'-o',meshdir,'--config','default','--downsample_ratio',.5,'--interpolate_views'])
            mesh_vertices=finite_ply(mesh)
            triangles=PlyData.read(mesh,known_list_len={'face':{'vertex_indices':3}})['face'].count
            assert triangles>0,mesh
            (model/'integrity.json').write_text(json.dumps({'gaussians':count,'all_gaussian_float_properties_finite':True,
                'mesh_vertices':mesh_vertices,'mesh_faces':triangles,'all_mesh_vertices_finite':True,
                'iteration':7000,'checkpoint':str(point),'mesh':str(mesh)},indent=2))
            if not metric.exists():
                run(scan,'evaluation',[ROOT/'.venv/bin/python',ROOT/'repro/eval_dtu.py','--mesh',mesh,'--scan',scan,'--scene',ROOT/f'data/DTU/small_overlap/scan{scan}','--gt-root',ROOT/'data/DTU_GT','--data-type','small_overlap'],cwd=ROOT)
            comparison=ROOT / f'output/MAtCha_comparison/scan{scan}'
            for method,python in [('dpgs',ROOT/'.venv/bin/python'),('matcha',Path(sys.executable))]:
                if not (comparison/method/'manifest.json').exists():
                    run(scan,f'render_{method}',[python,ROOT/'repro/render_matcha_comparison.py','--scan',scan,'--method',method],cwd=ROOT)
            if not (comparison/'comparison_motion.mp4').exists():
                run(scan,'compose',[ROOT/'.venv/bin/python',ROOT/'repro/render_matcha_comparison.py','--scan',scan,'--method','compose'],cwd=ROOT)
            run(scan,'summary',[ROOT/'.venv/bin/python',ROOT/'repro/summarize_matcha.py'],cwd=ROOT)
            rows = summarize()
            print(f'MATCHA_SCAN_COMPLETE scan{scan} completed={len(rows)}/15 cd={json.loads(metric.read_text())["overall"]}',flush=True)
            save()
        report['status']='completed'
    except BaseException as exc:
        report['status']='failed'
        report['error']=repr(exc)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()

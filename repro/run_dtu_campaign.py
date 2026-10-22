"""Resume complete DTU stages without retraining or overwriting partial training."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

from run_dtu import ROOT, SCANS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--gt-root', type=Path, default=ROOT / 'data/DTU_GT')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'output')
    parser.add_argument('--data-type', choices=['small_overlap', 'large_overlap'], default='small_overlap')
    parser.add_argument('--sfm-type', choices=['colmap_sfm', 'mast3r_sfm'], default='colmap_sfm')
    parser.add_argument('--scans', nargs='+', type=int, choices=SCANS, default=SCANS)
    parser.add_argument('--restart-partial', action='store_true',
                        help='Archive interrupted training without a final model and restart it')
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.gt_root = args.gt_root.resolve()
    args.output_root = args.output_root.resolve()
    scope_path = ROOT / 'repro/records/dtu_requested_scope.json'
    if scope_path.exists() and args.output_root == (ROOT / 'output').resolve():
        scope = json.loads(scope_path.read_text())
        if args.sfm_type not in scope.get('initializations', ['colmap_sfm', 'mast3r_sfm']):
            print(f'CAMPAIGN_DISABLED_BY_USER {args.sfm_type}; no preparation, training or evaluation scheduled', flush=True)
            return
    common = ['--data-root', str(args.data_root.resolve()), '--gt-root', str(args.gt_root.resolve()),
              '--output-root', str(args.output_root.resolve()), '--data-type', args.data_type,
              '--sfm-type', args.sfm_type]

    def run(script, options):
        subprocess.run([sys.executable, '-u', str(ROOT / 'repro' / script), *options], cwd=ROOT, check=True)

    for scan in args.scans:
        model = args.output_root / 'DTU' / f'scan{scan}_{args.sfm_type}_{args.data_type}'
        meshdir = model / 'train/ours_10000'
        result = meshdir / 'results.json'
        point = model / 'point_cloud/iteration_10000/point_cloud.ply'
        options = [*common, '--scans', str(scan)]
        validation = [*common, '--scan', str(scan)]
        print(f'CAMPAIGN_SCENE_START scan{scan} {args.sfm_type} {args.data_type}', flush=True)
        if not result.is_file():
            if not point.is_file():
                if (model / 'cfg_args').exists():
                    if not args.restart_partial:
                        raise RuntimeError(f'Partial training preserved: {model}; inspect its checkpoint before resuming')
                    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                    archive = args.output_root / 'DTU_interrupted' / f'{model.name}_{stamp}'
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    model.rename(archive)
                    (archive / 'repro_interruption.json').write_text(json.dumps({
                        'archived_at': stamp, 'original_path': str(model),
                        'reason': 'No final iteration_10000 model; preserve interrupted run and restart from initialization',
                    }, indent=2))
                    print(f'Archived interrupted training at {archive}', flush=True)
                if not (model / 'repro_prepare_record.json').exists():
                    run('run_dtu.py', [*options, '--stage', 'prepare'])
                run('validate_dtu.py', [*validation, '--stage', 'inputs'])
                run('run_dtu.py', [*options, '--stage', 'train'])
            if not (meshdir / 'fuse_post.ply').is_file():
                run('run_dtu.py', [*options, '--stage', 'mesh'])
            run('run_dtu.py', [*options, '--stage', 'eval'])
        run('validate_dtu.py', [*validation, '--stage', 'outputs'])
        rows = []
        for candidate in SCANS:
            report = ROOT / 'repro/records' / f'scan{candidate}_{args.sfm_type}_{args.data_type}_outputs.json'
            actual = args.output_root / 'DTU' / f'scan{candidate}_{args.sfm_type}_{args.data_type}/train/ours_10000/results.json'
            if report.exists() and actual.exists():
                validated = json.loads(report.read_text())
                origin = Path(validated['arguments'].get('output_root', ROOT / 'output')).resolve()
                values = json.loads(actual.read_text())
                if origin == args.output_root and values == validated['metrics']:
                    rows.append({'scan': candidate, **values})
        progress = {'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'completed': len(rows), 'total': 15, 'scenes': rows,
                    'missing': [candidate for candidate in SCANS if candidate not in [r['scan'] for r in rows]]}
        path = ROOT / 'repro/records' / f'dtu_{args.sfm_type}_{args.data_type}_progress.json'
        path.write_text(json.dumps(progress, indent=2))
        print(f'CAMPAIGN_SCENE_COMPLETE scan{scan}; {len(rows)}/15; no partial mean reported', flush=True)
        if args.output_root == (ROOT / 'output').resolve():
            run('report_dtu_campaign.py', [])
    if set(args.scans) == set(SCANS):
        run('summarize_dtu.py', ['--output-root', str(args.output_root),
                                '--data-type', args.data_type, '--sfm-type', args.sfm_type])


if __name__ == '__main__':
    main()

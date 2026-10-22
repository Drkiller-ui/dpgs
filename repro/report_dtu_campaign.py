"""Report saved models and validated metrics for the requested DTU paper table."""
import argparse
import datetime
import json
from pathlib import Path

from run_dtu import ROOT, SCANS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--data-types', nargs='+', choices=['small_overlap', 'large_overlap'],
                        default=['small_overlap'])
    parser.add_argument('--sfm-types', nargs='+', choices=['colmap_sfm', 'mast3r_sfm'])
    args = parser.parse_args()
    if args.sfm_types is None:
        scope_path = ROOT / 'repro/records/dtu_requested_scope.json'
        scope = json.loads(scope_path.read_text()) if scope_path.exists() else {}
        args.sfm_types = scope.get('initializations', ['colmap_sfm', 'mast3r_sfm'])
    args.sfm_types = list(dict.fromkeys(args.sfm_types))
    groups = []
    for overlap in dict.fromkeys(args.data_types):
        for sfm in args.sfm_types:
            models, evaluated, validated, missing = [], [], [], []
            for scan in SCANS:
                model = ROOT / 'output/DTU' / f'scan{scan}_{sfm}_{overlap}'
                point = model / 'point_cloud/iteration_10000/point_cloud.ply'
                result = model / 'train/ours_10000/results.json'
                validation = ROOT / 'repro/records' / f'scan{scan}_{sfm}_{overlap}_outputs.json'
                if point.is_file() and point.stat().st_size > 0:
                    models.append(scan)
                if result.is_file():
                    metrics = json.loads(result.read_text())
                    evaluated.append(scan)
                    if validation.is_file():
                        report = json.loads(validation.read_text())
                        root = Path(report['arguments'].get('output_root', ROOT / 'output')).resolve()
                        if root == (ROOT / 'output').resolve() and report['metrics'] == metrics and scan in models:
                            validated.append(scan)
                if scan not in validated:
                    missing.append(scan)
            group = {'overlap': overlap, 'initialization': sfm, 'models_saved': models,
                     'evaluated': evaluated, 'validated': validated, 'missing': missing}
            if len(validated) == 15:
                group['mean_cd'] = sum(json.loads((ROOT / 'output/DTU' /
                    f'scan{scan}_{sfm}_{overlap}/train/ours_10000/results.json').read_text())['overall']
                    for scan in SCANS) / 15
            groups.append(group)
    report = {'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'models_saved': sum(len(group['models_saved']) for group in groups),
              'validated': sum(len(group['validated']) for group in groups),
              'total': 15 * len(set(args.data_types)) * len(args.sfm_types), 'groups': groups}
    (ROOT / 'repro/records/dtu_all_progress.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if args.require_complete and report['validated'] != report['total']:
        raise SystemExit('DTU campaign is incomplete')


if __name__ == '__main__':
    main()

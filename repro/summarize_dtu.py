"""Require all 15 DTU metrics before reporting the paper-comparable mean."""
import argparse
import csv
import json
from pathlib import Path
import sys

SCANS = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parents[1] / 'output')
    p.add_argument('--sfm-type', choices=['colmap_sfm', 'mast3r_sfm'], default='colmap_sfm')
    p.add_argument('--data-type', choices=['small_overlap', 'large_overlap'], default='small_overlap')
    a = p.parse_args()
    rows, missing = [], []
    for scan in SCANS:
        file = a.output_root / 'DTU' / f'scan{scan}_{a.sfm_type}_{a.data_type}/train/ours_10000/results.json'
        if not file.exists():
            missing.append(scan)
            continue
        values = json.loads(file.read_text())
        value = float(values['overall'])
        if not 0 <= value < float('inf'):
            raise ValueError(f'Non-finite or negative CD: {file}')
        rows.append({'scan': scan, 'mean_d2s': values['mean_d2s'], 'mean_s2d': values['mean_s2d'], 'overall': value})
    print(f'Found {len(rows)}/15 scenes')
    if missing:
        print(f'Missing scans: {missing}; no full-dataset mean reported.', file=sys.stderr)
        return 2
    target = a.output_root / f'dtu_{a.sfm_type}_{a.data_type}.csv'
    with target.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['scan', 'mean_d2s', 'mean_s2d', 'overall'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Mean CD: {sum(row["overall"] for row in rows) / 15:.6f}')
    print(target)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

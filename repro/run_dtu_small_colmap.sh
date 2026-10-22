#!/usr/bin/env bash
# Sequential campaign; reuse completed stages and stop on any failure.
set -euo pipefail
cd "$(dirname "$0")/.."
source repro/env.sh
scans=(24 37 40 55 63 65 69 83 97 105 106 110 114 118 122)
for scan in "${scans[@]}"; do
    model="output/DTU/scan${scan}_colmap_sfm_small_overlap"
    mesh="$model/train/ours_10000/fuse_post.ply"
    result="$model/train/ours_10000/results.json"
    common=(--data-root data/DTU --gt-root data/DTU_GT --scans "$scan")
    echo "CAMPAIGN_SCENE_START scan${scan} $(date -u +%FT%TZ)"
    if [[ -s "$result" ]]; then
        echo "Existing result; verifying before reuse: $result"
    elif [[ -e "$model/cfg_args" ]]; then
        if [[ ! -s "$model/point_cloud/iteration_10000/point_cloud.ply" ]]; then
            echo "Partial training detected at $model; preserve it and inspect checkpoint before resuming." >&2
            exit 4
        fi
        [[ -s "$mesh" ]] || python -u repro/run_dtu.py "${common[@]}" --stage mesh
        python -u repro/run_dtu.py "${common[@]}" --stage eval
    else
        python -u repro/run_dtu.py "${common[@]}"
    fi
    python repro/validate_dtu.py --scan "$scan" --stage outputs
    python - <<'PY'
import datetime, json
from pathlib import Path
scans = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
rows = []
for scan in scans:
    record = Path(f'repro/records/scan{scan}_colmap_sfm_small_overlap_outputs.json')
    if record.exists():
        rows.append({'scan': scan, **json.loads(record.read_text())['metrics']})
report = {'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'completed': len(rows), 'total': 15, 'scenes': rows,
          'missing': [scan for scan in scans if scan not in [r['scan'] for r in rows]]}
Path('repro/records/dtu_small_colmap_progress.json').write_text(json.dumps(report, indent=2))
print('CAMPAIGN_PROGRESS', len(rows), '/ 15; no partial-dataset mean reported', flush=True)
PY
    echo "CAMPAIGN_SCENE_COMPLETE scan${scan} $(date -u +%FT%TZ)"
done
python repro/summarize_dtu.py
echo "CAMPAIGN_COMPLETE $(date -u +%FT%TZ)"

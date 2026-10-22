#!/usr/bin/env bash
# Current user scope: 15 COLMAP scans, small_overlap only.
set -euo pipefail
cd "$(dirname "$0")/.."
source repro/env.sh
export PYTHONUNBUFFERED=1
mkdir -p logs
exec 8>logs/dtu_gpu_campaign.lock
flock -n 8 || { echo 'A DTU GPU campaign is already running' >&2; exit 3; }
for overlap in small_overlap; do
    for sfm in colmap_sfm; do
        root=data/DTU
        if [[ "$sfm" == mast3r_sfm ]]; then
            root=data/DTU_mast3r_recomputed
        fi
        echo "DTU_CONFIGURATION_START $overlap $sfm $(date -u +%FT%TZ)"
        python -u repro/run_dtu_campaign.py --data-root "$root" \
            --data-type "$overlap" --sfm-type "$sfm" --restart-partial
        echo "DTU_CONFIGURATION_COMPLETE $overlap $sfm $(date -u +%FT%TZ)"
    done
done
python -u repro/report_dtu_campaign.py --require-complete
echo "PAPER_DTU_COLMAP_15_COMPLETE $(date -u +%FT%TZ)"

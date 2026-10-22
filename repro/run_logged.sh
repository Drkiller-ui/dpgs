#!/usr/bin/env bash
# Run under nohup/tmux on the server. Captures output, exit status and GPU use.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 2 || ! "$1" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo 'Usage: bash repro/run_logged.sh JOB_NAME COMMAND [ARGUMENTS...]' >&2
    exit 2
fi
job_name="$1"
shift
mkdir -p logs
exec 9>"logs/${job_name}.lock"
flock -n 9 || { echo "Job ${job_name} is already running" >&2; exit 3; }
run_dir="logs/${job_name}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$run_dir"
printf '%s\n' "$$" > "$run_dir/pid"
printf '%s\n' running > "$run_dir/status"
printf '%q ' "$@" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
printf '%s\n' "$run_dir" > "logs/${job_name}.latest"
exec > >(tee -a "$run_dir/output.log") 2>&1
monitor_pid=''
cleanup() {
    result=$?
    trap - EXIT
    if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
    printf '%s\n' "$result" > "$run_dir/exit_code"
    if [[ "$result" -eq 0 ]]; then
        printf '%s\n' completed > "$run_dir/status"
    else
        printf '%s\n' failed > "$run_dir/status"
    fi
    date -u +%FT%TZ > "$run_dir/finished_at"
    exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
date -u +%FT%TZ > "$run_dir/started_at"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
        --format=csv -l 1 > "$run_dir/gpu.csv" 2> "$run_dir/gpu-monitor.log" &
    monitor_pid=$!
fi
echo "Run directory: $run_dir"
"$@"

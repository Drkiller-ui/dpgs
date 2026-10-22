"""Observe the 30-run DTU campaign; stop on completion, failure or a missing runner."""
import datetime
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    job = ROOT / (ROOT / 'logs/dtu_paper_30.latest').read_text().strip()
    text = (job / 'output.log').read_text(errors='replace')
    starts = list(re.finditer(r'CAMPAIGN_SCENE_START scan(\d+) (\w+) (\w+)', text))
    scene = starts[-1] if starts else None
    tail = text[scene.start():] if scene else text
    steps = re.findall(r'Training progress:[^\r\n]+', tail)
    state = (job / 'status').read_text().strip()
    scope_path = ROOT / 'repro/records/dtu_requested_scope.json'
    scope = json.loads(scope_path.read_text()) if scope_path.exists() else {}
    initializations = scope.get('initializations', ['colmap_sfm', 'mast3r_sfm'])
    models = sorted(path.parent.parent.parent.name for sfm in initializations for path in
                    (ROOT / 'output/DTU').glob(f'*_{sfm}_small_overlap/point_cloud/iteration_10000/point_cloud.ply'))
    results = sorted(path.parent.parent.parent.name for sfm in initializations for path in
                     (ROOT / 'output/DTU').glob(f'*_{sfm}_small_overlap/train/ours_10000/results.json'))
    commands = subprocess.check_output(['ps', '-eo', 'args='], text=True).splitlines()
    processes = [line.strip() for line in commands if any(token in line for token in
                 ['python train.py ', 'python render_and_mesh_extract.py ', 'python run_mast3r.py ',
                  'python -u repro/run_dtu_campaign.py ', 'python repro/run_dtu_campaign.py ',
                  'python -u repro/run_all_dtu', 'python /root/autodl-tmp/DP-GS/scripts/dtu_eval/eval.py'])]
    if state == 'running' and not any('repro/run_dtu_campaign.py ' in line for line in processes):
        # The initial preparation and final summary can run without a campaign child.
        if not any('bash repro/run_all_dtu.sh' in line for line in commands):
            state = 'runner_missing'
    report = {'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'job': str(job), 'status': state, 'models_saved': len(models),
              'evaluated': len(results), 'total': 15 * len(initializations),
              'current_scan': int(scene.group(1)) if scene else None,
              'initialization': scene.group(2) if scene else None,
              'overlap': scene.group(3) if scene else 'small_overlap',
              'last_training_progress': steps[-1] if steps else None,
              'active_processes': processes, 'completed_results': results,
              'last_output_age_seconds': round(time.time() - (job / 'output.log').stat().st_mtime, 1)}
    if state != 'running':
        report['log_tail'] = text[-6000:].replace('\r', '\n')
    target = ROOT / 'repro/records/dtu_live_progress.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(target)
    return report


def main():
    previous = None
    while True:
        report = snapshot()
        key = (report['status'], report['evaluated'], report['current_scan'], report['initialization'])
        summary = {key: value for key, value in report.items()
                   if key not in ['completed_results', 'active_processes']}
        summary['event'] = 'change' if key != previous else 'heartbeat'
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        previous = key
        if report['status'] != 'running':
            return
        time.sleep(45)


if __name__ == '__main__':
    main()

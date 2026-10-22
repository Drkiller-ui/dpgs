"""Observe the existing campaign; persist heartbeats without changing experiments."""
import datetime
import json
import shutil
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
record=ROOT/'repro/records/matcha_monitor.jsonl'
last_emit=0.
previous=None
while True:
    progress=json.loads((ROOT/'repro/records/matcha_progress.json').read_text())
    job=ROOT/(ROOT/'logs/matcha_colmap.latest').read_text().strip()
    status=(job/'status').read_text().strip()
    scan=progress.get('current_scan')
    stage=progress.get('current_stage')
    folder=ROOT/f'output/MAtCha_DTU/colmap_small_overlap/scan{scan}'
    log=folder/f'{stage}.log'
    tail=log.read_text(errors='replace').replace('\r','\n').splitlines()[-2:] if log.exists() else []
    trained=sum((ROOT/f'output/MAtCha_DTU/colmap_small_overlap/scan{s}/free_gaussians/point_cloud/iteration_7000/point_cloud.ply').exists() for s in [24,37,40,55,63,65,69,83,97,105,106,110,114,118,122])
    evaluated=len(progress.get('completed',[]))
    signature=(scan,stage,status,trained,evaluated)
    if signature!=previous or time.monotonic()-last_emit>=50:
        report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,
            'scan':scan,'stage':stage,'trained':trained,'evaluated':evaluated,
            'data_disk_free_gib':round(shutil.disk_usage(ROOT).free/2**30,1),
            'persistent_free_gib':round(shutil.disk_usage(ROOT/'output').free/2**30,1),
            'tail':tail}
        if status!='running':
            report['job_log_tail']=(job/'output.log').read_text(errors='replace')[-3500:]
        with record.open('a') as f:
            f.write(json.dumps(report,ensure_ascii=False)+'\n')
        print(json.dumps(report,ensure_ascii=False),flush=True)
        previous=signature
        last_emit=time.monotonic()
    if status!='running':
        break
    time.sleep(5)

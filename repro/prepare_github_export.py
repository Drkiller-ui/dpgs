"""Prepare a clean source/results snapshot, without environments, datasets or models."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {'.git', '.venv', '__pycache__', 'build', 'dist', '.publish'}
EXCLUDED_SUFFIXES = {'.pyc', '.pyo', '.so', '.o', '.a', '.pth', '.pt', '.ckpt', '.safetensors',
                     '.ply', '.npy', '.npz', '.pem', '.key'}
RESULT_SUFFIXES = {'.png', '.jpg', '.jpeg', '.gif', '.mp4', '.tif', '.tiff', '.html', '.md', '.json', '.csv', '.txt'}
# Publish only the validated COLMAP campaigns. Failed/interrupted models and
# exploratory MASt3R results remain on the server, outside the comparison.
RESULT_ROOTS = ['output/DTU', 'output/DTU_visuals', 'output/DTU_geometry',
                'output/MAtCha_DTU/colmap_small_overlap', 'output/MAtCha_comparison']
RESULT_FILES = ['output/dtu_colmap_sfm_small_overlap.csv', 'output/dtu_matcha_vs_dpgs.csv']


def allowed(path):
    relative = path.relative_to(ROOT)
    return (not any(part in EXCLUDED_PARTS or part.endswith('.egg-info') for part in relative.parts)
            and path.suffix.lower() not in EXCLUDED_SUFFIXES
            and path.name != 'publish_inventory.json'
            and not path.name.startswith('.env'))


def collect():
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    candidates = {ROOT / name: 'source' for name in tracked if name}
    for directory in ['repro', 'docs']:
        for path in (ROOT / directory).rglob('*'):
            if path.is_file() and allowed(path):
                candidates[path] = 'source_and_records'
    for name in ['README.md', 'UPSTREAM_README.md', '.gitignore', '.gitattributes', '.python-version', 'pyproject.toml', 'uv.lock']:
        path = ROOT / name
        if path.is_file():
            candidates[path] = 'source'
    for directory in RESULT_ROOTS:
        for path in (ROOT / directory).rglob('*'):
            if path.is_file() and allowed(path):
                if path.suffix.lower() in RESULT_SUFFIXES or path.name == 'cfg_args' or path.name.startswith('events.out.tfevents.'):
                    candidates[path] = 'results_and_training_logs'
    for name in RESULT_FILES:
        path=ROOT/name
        if path.is_file():
            candidates[path]='results_and_training_logs'
    for path in (ROOT / 'logs').rglob('*'):
        if path.is_file() and not path.name.endswith('.lock'):
            candidates[path] = 'logs'
    return {path: kind for path, kind in candidates.items() if path.is_file() and allowed(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(f'Preserve existing export: {destination}')
    files = collect()
    patterns = {
        'github_token': rb'(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{40,255})',
        'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
        'aws_access_key': rb'AKIA[0-9A-Z]{16}',
        'huggingface_token': rb'hf_[A-Za-z0-9]{30,255}',
    }
    findings = []
    inventory = []
    for path, kind in sorted(files.items()):
        relative = path.relative_to(ROOT)
        content = path.read_bytes()
        if kind != 'results_and_training_logs' and path.suffix not in {'.whl'}:
            for label, pattern in patterns.items():
                if re.search(pattern, content):
                    findings.append({'file': str(relative), 'kind': label})
        if len(content) >= 100 * 1024 * 1024:
            raise ValueError(f'File exceeds ordinary GitHub limit: {relative}')
        inventory.append({'path': relative.as_posix(), 'kind': kind, 'bytes': len(content),
                          'sha256': hashlib.sha256(content).hexdigest()})
    if findings:
        raise RuntimeError('Potential credentials require review (values withheld): ' + json.dumps(findings))
    destination.mkdir(parents=True)
    for entry in inventory:
        source = ROOT / entry['path']
        target = destination / entry['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        # Regular copies dereference the original output/log directory links.
        shutil.copy2(source, target)
    report = {'prepared_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'repository': 'https://github.com/Drkiller-ui/dpgs',
              'destination': str(destination), 'files': len(inventory),
              'bytes': sum(entry['bytes'] for entry in inventory),
              'upstream_commit': 'dc1ce835ea3c6bb7887333c50df7a13497a45c09',
              'excluded': ['datasets', 'pretrained weights', 'trained Gaussian and appearance models',
                           'meshes and error point clouds', 'raw NPY/NPZ arrays', 'environments and build caches',
                           'original Git object database', 'failed/interrupted and exploratory result directories'],
              'credential_pattern_scan_passed': True,
              'inventory': inventory}
    target = destination / 'repro/records/publish_inventory.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    (ROOT / 'repro/records/publish_inventory.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'inventory'}, indent=2))


if __name__ == '__main__':
    main()

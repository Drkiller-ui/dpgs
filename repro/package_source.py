"""Package upstream source and reproducibility files, excluding local environments."""
from pathlib import Path
import subprocess
import zipfile

root = Path(__file__).resolve().parents[1]
target = root.parent / 'DP-GS-复现准备包.zip'
tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
files = {root / name for name in tracked if name}
for dirname in ['repro', 'docs']:
    files.update(p for p in (root / dirname).rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and not p.name.startswith('qa-page-'))
files.update(root / name for name in ['.gitignore', '.python-version', 'pyproject.toml', 'uv.lock'])
with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted(files):
        if path.is_file():
            z.write(path, 'DP-GS/' + path.relative_to(root).as_posix())
print(f'{target}: {len(files)} source/artifact files, {target.stat().st_size} bytes')

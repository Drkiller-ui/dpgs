"""Stage input links so new MASt3R runs preserve author initialization."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANS = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]


def main():
    source = ROOT / 'data/DTU'
    target = ROOT / 'data/DTU_mast3r_recomputed'
    records = []
    for overlap in ['small_overlap', 'large_overlap']:
        for scan in SCANS:
            original = source / overlap / f'scan{scan}'
            staged = target / overlap / f'scan{scan}'
            staged.mkdir(parents=True, exist_ok=True)
            for name in ['images', 'sparse', 'dense', 'cameras.npz', 'normals_metric3d', 'split.json']:
                item = original / name
                if not item.exists():
                    if name == 'split.json':
                        continue
                    raise FileNotFoundError(item)
                link = staged / name
                if link.exists() or link.is_symlink():
                    if not link.is_symlink() or link.resolve() != item.resolve():
                        raise FileExistsError(f'Preserve unexpected staging entry: {link}')
                else:
                    link.symlink_to(item.resolve(), target_is_directory=item.is_dir())
            records.append({'source': str(original), 'staged': str(staged),
                            'author_mast3r_preserved': str(original / 'mast3r'),
                            'new_mast3r_output': str(staged / 'mast3r')})
    (ROOT / 'repro/records/mast3r_staged_inputs.json').write_text(json.dumps(records, indent=2))
    print(f'Staged {len(records)} scenes at {target}; author MASt3R outputs preserved.')


if __name__ == '__main__':
    main()

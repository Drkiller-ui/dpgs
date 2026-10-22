"""Fetch only requested TNT scenes using the current official public links."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
import zipfile

from bs4 import BeautifulSoup
import requests

SOURCE = 'https://www.tanksandtemples.org/download/'
SCENES = ['Truck', 'Caterpillar', 'Barn', 'Ignatius']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-dir', type=Path, default=Path('/root/autodl-fs/dpgs/downloads/TNT'))
    parser.add_argument('--scenes', nargs='+', choices=SCENES, default=SCENES)
    args = parser.parse_args()
    args.download_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(SOURCE, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    assets = []
    seen = set()
    for row in soup.select('tr'):
        cells = row.find_all('td')
        if not cells:
            continue
        scene = cells[0].get_text(strip=True)
        if scene not in args.scenes:
            continue
        links = [tag['href'] for tag in row.select('a[href]')]
        if len(links) == 3 and (scene, 'images') not in seen:
            for role, link in [('ground_truth', links[0]), ('images', links[2])]:
                assets.append({'scene': scene, 'role': role, 'url': link})
                seen.add((scene, role))
        elif len(links) == 4:
            for role, link in zip(['trajectory', 'alignment', 'crop'], links[1:]):
                assets.append({'scene': scene, 'role': role, 'url': link})
    assert len(assets) == 5 * len(args.scenes), assets
    confirm = importlib.import_module('gdown.download').get_url_from_gdrive_confirmation

    def fetch(asset):
        asset = dict(asset)
        scene, role = asset['scene'], asset['role']
        suffix = {'images': '.zip', 'ground_truth': '.ply', 'trajectory': '_COLMAP_SfM.log',
                  'alignment': '_trans.txt', 'crop': '.json'}[role]
        target = args.download_dir / scene / (scene + suffix)
        target.parent.mkdir(exist_ok=True)
        asset['file'] = str(target)
        try:
            if not target.exists():
                original = urlparse(asset['url'])
                identifier = original.path.split('/d/')[1].split('/')[0]
                query = parse_qs(original.query)
                params = {'id': identifier, 'export': 'download'}
                if query.get('resourcekey'):
                    params['resourcekey'] = query['resourcekey'][0]
                url = 'https://drive.google.com/uc?' + urlencode(params)
                with requests.Session() as session:
                    for _ in range(4):
                        res = session.get(url, stream=True, timeout=(30, 120))
                        res.raise_for_status()
                        if 'Content-Disposition' in res.headers:
                            break
                        try:
                            url = confirm(res.text)
                        finally:
                            res.close()
                    else:
                        raise RuntimeError('Official public download confirmation did not resolve')
                    partial = target.with_name(target.name + '.partial')
                    expected = int(res.headers.get('Content-Length', '0'))
                    with res, partial.open('wb') as out:
                        for block in res.iter_content(2 ** 20):
                            out.write(block)
                    if expected and partial.stat().st_size != expected:
                        raise RuntimeError('Truncated response')
                    partial.replace(target)
            if role == 'images':
                with zipfile.ZipFile(target) as archive:
                    bad = archive.testzip()
                    if bad:
                        raise RuntimeError('ZIP CRC failed: ' + bad)
                    asset['zip_entries'] = len(archive.infolist())
            elif role == 'ground_truth':
                with target.open('rb') as source:
                    assert source.read(3) == b'ply', target
            elif role == 'crop':
                json.loads(target.read_text())
            digest = hashlib.sha256()
            with target.open('rb') as source:
                for block in iter(lambda: source.read(8 << 20), b''):
                    digest.update(block)
            asset.update(status='downloaded_validated', bytes=target.stat().st_size, sha256=digest.hexdigest())
            print('TNT_READY', scene, role, asset['bytes'], flush=True)
        except Exception as exc:
            asset.update(status='failed', error=str(exc))
            print('TNT_FAILED', scene, role, str(exc), flush=True)
        return asset

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(fetch, assets))
    record = {'source': SOURCE, 'assets': results}
    path = Path(__file__).resolve().parent / 'records/tnt_current_assets.json'
    path.write_text(json.dumps(record, indent=2))
    if any(item['status'] == 'failed' for item in results):
        raise RuntimeError('Some TNT assets are unavailable; see ' + str(path))


if __name__ == '__main__':
    main()

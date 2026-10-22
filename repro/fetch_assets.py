"""Download official DP-GS/DTU assets; extract only requested DTU scans."""
import argparse
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import io
from pathlib import Path, PurePosixPath
import shutil
import zipfile

import gdown
import requests


class HTTPRangeFile(io.RawIOBase):
    """Seek a large official ZIP via byte ranges instead of downloading all scans."""
    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        self.session.headers.update({'Accept-Encoding': 'identity'})
        response = self.session.get(url, headers={'Range': 'bytes=0-0'}, stream=True, timeout=90)
        if response.status_code != 206 or 'Content-Range' not in response.headers:
            response.close()
            raise RuntimeError(f'Server does not support byte ranges: {url}')
        self.length = int(response.headers['Content-Range'].split('/')[-1])
        response.close()
        self.pos = 0
        self.cache = OrderedDict()
        self.blocksize = 2 ** 20

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        target = offset if whence == 0 else self.pos + offset if whence == 1 else self.length + offset
        if target < 0:
            raise ValueError('Negative seek')
        self.pos = target
        return target

    def read(self, n=-1):
        end = self.length if n < 0 else min(self.length, self.pos + n)
        parts = []
        while self.pos < end:
            block = self.pos // self.blocksize
            if block not in self.cache:
                start = block * self.blocksize
                stop = min(self.length, start + self.blocksize) - 1
                response = self.session.get(self.url, headers={'Range': f'bytes={start}-{stop}'}, timeout=120)
                response.raise_for_status()
                if response.status_code != 206 or len(response.content) != stop - start + 1:
                    raise RuntimeError(f'Invalid range response for {self.url}')
                self.cache[block] = response.content
                if len(self.cache) > 16:
                    self.cache.popitem(last=False)
            data = self.cache[block]
            offset = self.pos % self.blocksize
            part = data[offset:offset + min(end - self.pos, len(data) - offset)]
            parts.append(part)
            self.pos += len(part)
        return b''.join(parts)


def extract_member(z, info, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == info.file_size:
        print('Already extracted', target, flush=True)
        return
    temporary = target.with_name(target.name + '.partial')
    with z.open(info) as source, temporary.open('wb') as dest:
        shutil.copyfileobj(source, dest, 2 ** 20)
    if temporary.stat().st_size != info.file_size:
        raise RuntimeError(f'Wrong extracted size: {target}')
    temporary.replace(target)
    print('Extracted', target, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--download-dir', type=Path, default=Path('/root/autodl-fs/dpgs/downloads'))
    p.add_argument('--scans', type=int, nargs='+', default=[24])
    p.add_argument('--only', choices=['all', 'weight', 'data', 'gt'], default='all')
    a = p.parse_args()
    a.download_dir.mkdir(parents=True, exist_ok=True)

    def weight():
        target = a.root / 'Metric3D/weight/metric_depth_vit_large_800k.pth'
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            gdown.download(id='1eT2gG-kwsVzNy5nJrbm4KC-9DbNKyLnr', output=str(target), quiet=False, resume=True)
        if not target.exists() or target.stat().st_size < 100_000_000:
            raise RuntimeError('Metric3D weight download is missing or incomplete')
        print('WEIGHT_READY', target.stat().st_size, flush=True)

    def data():
        archive = a.download_dir / 'DTU_sparse2.zip'
        if not archive.exists() or not zipfile.is_zipfile(archive):
            gdown.download(id='11NBPiVAnjB7HxMIE56rl_2Zk1it0IkWp', output=str(archive), quiet=False, resume=True)
        with zipfile.ZipFile(archive) as z:
            print('Dataset ZIP entries', len(z.infolist()), flush=True)
            matches = 0
            for info in z.infolist():
                if info.is_dir():
                    continue
                parts = PurePosixPath(info.filename.replace('\\', '/')).parts
                for i, part in enumerate(parts):
                    if part in ['small_overlap', 'large_overlap'] and i + 1 < len(parts) and parts[i+1] in {f'scan{x}' for x in a.scans}:
                        suffix = Path(*parts[i:])
                        if '..' in suffix.parts:
                            raise RuntimeError('Unsafe ZIP path')
                        extract_member(z, info, a.root / 'data/DTU' / suffix)
                        matches += 1
                        break
            if matches == 0:
                print('First ZIP entries:', z.namelist()[:40], flush=True)
                raise RuntimeError('No requested DTU scenes matched archive layout')
        print('DTU_DATA_READY', flush=True)

    def gt():
        wanted = {f'stl{x:03}_total.ply' for x in a.scans} | {f'ObsMask{x}_10.mat' for x in a.scans} | {f'Plane{x}.mat' for x in a.scans}
        found = set()
        for filename in ['SampleSet.zip', 'Points.zip']:
            url = f'https://roboimagedata2.compute.dtu.dk/data/MVS/{filename}'
            with HTTPRangeFile(url) as remote, zipfile.ZipFile(remote) as z:
                print(filename, 'remote bytes', remote.length, 'entries', len(z.infolist()), flush=True)
                for info in z.infolist():
                    parts = PurePosixPath(info.filename.replace('\\', '/')).parts
                    if not parts or parts[-1] not in wanted:
                        continue
                    name = parts[-1]
                    suffix = Path('Points/stl', name) if name.endswith('.ply') else Path('ObsMask', name)
                    extract_member(z, info, a.root / 'data/DTU_GT' / suffix)
                    found.add(name)
        if found != wanted:
            raise RuntimeError(f'Missing DTU GT files: {wanted - found}')
        print('DTU_GT_READY', flush=True)

    tasks = {'weight': weight, 'data': data, 'gt': gt}
    selected = list(tasks.values()) if a.only == 'all' else [tasks[a.only]]
    with ThreadPoolExecutor(max_workers=len(selected)) as executor:
        futures = [executor.submit(task) for task in selected]
        failures = []
        for future in futures:
            try:
                future.result()
            except Exception as exc:
                failures.append(exc)
                print(type(exc).__name__, str(exc), flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} asset tasks failed')


if __name__ == '__main__':
    main()

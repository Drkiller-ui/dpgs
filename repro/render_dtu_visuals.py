"""Export actual saved DTU models at their three training cameras, without training."""
import argparse
from argparse import Namespace
import ast
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps
import torch
import torch.nn.functional as F

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readColmapSceneInfo
from scene.app_model import AppModel
from utils.camera_utils import cameraList_from_camInfos
from run_dtu import SCANS

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def font(size):
    return ImageFont.truetype(FONT, size)


def save_rgb(path, array):
    if not np.isfinite(array).all():
        raise ValueError(f'Non-finite image: {path}')
    pixels = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def load_config(path):
    node = ast.parse(path.read_text(), mode='eval').body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'Namespace' or node.args:
        raise ValueError(f'Unexpected config syntax: {path}')
    return Namespace(**{kw.arg: ast.literal_eval(kw.value) for kw in node.keywords})


def fingerprint(paths):
    return {str(path): {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
            for path in paths}


def panel(images, title, subtitle, path):
    width, height, gap = 360, 269, 12
    canvas = Image.new('RGB', (4 * width + 5 * gap, 78 + len(images) * (height + 48) + 34), '#f3f5f8')
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 10), title, font=font(24), fill='#172030')
    draw.text((gap, 43), subtitle, font=font(15), fill='#475569')
    for row_index, (view, files, limits) in enumerate(images):
        top = 78 + row_index * (height + 48)
        for col, (label, image_path) in enumerate(files):
            left = gap + col * (width + gap)
            draw.text((left, top), f'{view} | {label}', font=font(17), fill='#172030')
            with Image.open(image_path) as im:
                tile = im.convert('RGB').resize((width, height), Image.Resampling.LANCZOS)
            canvas.paste(tile, (left, top + 25))
        draw.text((gap + 2 * (width + gap), top + 25 + height + 3),
                  f'depth display: {limits[0]:.3f} to {limits[1]:.3f}', font=font(13), fill='#475569')
    draw.text((gap, canvas.height - 25), 'Training views; supplied alpha mask applied. Depth colors use foreground 2-98 percentiles.',
              font=font(14), fill='#475569')
    canvas.save(path)


@torch.no_grad()
def export_scan(scan, output_root):
    model = ROOT / 'output/DTU' / f'scan{scan}_colmap_sfm_small_overlap'
    target = output_root / f'scan{scan}'
    folders = ['gt', 'gt_raw', 'renders', 'renders_raw', 'depth', 'normal', 'depth_normal', 'alpha']
    for name in folders:
        (target / name).mkdir(parents=True, exist_ok=True)
    point = model / 'point_cloud/iteration_10000/point_cloud.ply'
    protected = [point, model / 'cfg_args', model / 'app_model/iteration_10000/app.pth',
                 model / 'train/ours_10000/fuse_post.ply', model / 'train/ours_10000/results.json']
    before = fingerprint(protected)
    args = load_config(model / 'cfg_args')
    assert args.iterations == 10000 and args.resolution == 2 and args.sfm_type == 'colmap_sfm'
    # Match Scene(shuffle=False) camera loading, without rewriting its multi_view.json.
    scene_info = readColmapSceneInfo(args.source_path, args.images, args.eval, args.n_views,
                                     args.dataset_name, args.sfm_type)
    cameras = cameraList_from_camInfos(scene_info.train_cameras, 1.0, args)
    assert sorted(cam.image_name for cam in cameras) == ['0022', '0025', '0028']
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(str(point))
    background = torch.tensor([1., 1., 1.] if args.white_background else [0., 0., 0.], device='cuda')
    app_model = None
    if args.exposure_compensation:
        app_model = AppModel()
        app_model.load_weights(str(model), 10000)
        gaussians.use_app = True
    metrics = json.loads((model / 'train/ours_10000/results.json').read_text())
    records, panels = [], []
    for cam in cameras:
        name = cam.image_name
        pkg = render(cam, gaussians, args, background, app_model=app_model,
                     return_plane=True, return_depth_normal=True)
        rgb = pkg.get('app_image', pkg['render']).permute(1, 2, 0).cpu().numpy()
        gt = cam.original_image[:3].permute(1, 2, 0).cpu().numpy()
        depth = pkg['plane_depth'][0].cpu().numpy()
        mask = np.ones_like(depth) if cam.mask is None else cam.mask[0].cpu().numpy()
        normal = F.normalize(pkg['rendered_normal'], dim=0).permute(1, 2, 0).cpu().numpy()
        depth_normal = F.normalize(pkg['depth_normal'], dim=0).permute(1, 2, 0).cpu().numpy()
        for array in [rgb, gt, depth, mask, normal, depth_normal]:
            assert np.isfinite(array).all(), f'scan{scan}/{name}: non-finite render'
        valid = (mask > .5) & (depth > 0)
        assert valid.sum() > 100
        limits = np.percentile(depth[valid], [2, 98])
        assert limits[1] > limits[0]
        colored_depth = colormaps['turbo'](np.clip((depth - limits[0]) / (limits[1] - limits[0]), 0, 1))[..., :3]
        colored_depth[~valid] = 0
        save_rgb(target / 'gt_raw' / f'{name}.png', gt)
        save_rgb(target / 'renders_raw' / f'{name}.png', rgb)
        save_rgb(target / 'gt' / f'{name}.png', gt * mask[..., None])
        save_rgb(target / 'renders' / f'{name}.png', rgb * mask[..., None])
        save_rgb(target / 'depth' / f'{name}.png', colored_depth)
        save_rgb(target / 'normal' / f'{name}.png', (normal * .5 + .5) * valid[..., None])
        save_rgb(target / 'depth_normal' / f'{name}.png', (depth_normal * .5 + .5) * valid[..., None])
        Image.fromarray(np.rint(mask * 255).astype(np.uint8)).save(target / 'alpha' / f'{name}.png')
        np.save(target / 'depth' / f'{name}.npy', depth)
        Image.fromarray(depth.astype(np.float32)).save(target / 'depth' / f'{name}.tiff')
        np.save(target / 'normal' / f'{name}.npy', normal)
        np.save(target / 'depth_normal' / f'{name}.npy', depth_normal)
        files = [('Input', target / 'gt' / f'{name}.png'), ('Render', target / 'renders' / f'{name}.png'),
                 ('Depth', target / 'depth' / f'{name}.png'), ('Normal', target / 'normal' / f'{name}.png')]
        panels.append((name, files, limits))
        records.append({'view': name, 'source_image': cam.image_path, 'size': [cam.image_width, cam.image_height],
                        'camera_uid': cam.uid, 'training_view': True, 'alpha_mask_applied_for_display': cam.mask is not None,
                        'depth_display_2_98_percentiles': limits.tolist(),
                        'depth_units': 'model camera coordinates; no conversion to DTU millimeters',
                        'normal_coordinates': 'camera-local xyz; RGB = 0.5 * normalized normal + 0.5'})
    assert before == fingerprint(protected), 'Existing model, mesh, config or metric file changed'
    title = f'scan{scan} | COLMAP | 10,000 iterations | CD {metrics["overall"]:.4f}'
    panel(panels, title, 'Input / actual Gaussian render / depth / surface normal', target / 'comparison.png')
    report = {'scan': scan, 'model': str(model.resolve()), 'iteration': 10000, 'cd': metrics['overall'],
              'model_sha256': hashlib.sha256(point.read_bytes()).hexdigest(),
              'protected_files_unchanged': True, 'views': records, 'output': str(target.resolve())}
    (target / 'manifest.json').write_text(json.dumps(report, indent=2))
    del pkg, cameras, gaussians, app_model
    torch.cuda.empty_cache()
    return report


def make_gallery(reports, output_root):
    tile_w, tile_h, gap = 480, 232, 16
    overview = Image.new('RGB', (3 * tile_w + 4 * gap, 92 + 5 * tile_h + 20), '#f3f5f8')
    draw = ImageDraw.Draw(overview)
    draw.text((gap, 12), 'DTU | 15 COLMAP models | training view 0025', font=font(29), fill='#172030')
    draw.text((gap, 53), 'Each pair: INPUT (left) / RENDER (right); same supplied foreground mask. All 15 scans shown.',
              font=font(17), fill='#475569')
    cards = []
    for index, report in enumerate(reports):
        scan = report['scan']
        left, top = gap + (index % 3) * (tile_w + gap), 92 + (index // 3) * tile_h
        draw.text((left, top), f'scan{scan}   CD {report["cd"]:.4f}    Input | Render', font=font(18), fill='#172030')
        for col, kind in enumerate(['gt', 'renders']):
            with Image.open(output_root / f'scan{scan}' / kind / '0025.png') as im:
                overview.paste(im.convert('RGB').resize((238, 178), Image.Resampling.LANCZOS),
                               (left + col * 242, top + 30))
        links = ' &middot; '.join(f'<a href="scan{scan}/{kind}/0025.png">{label}</a>' for kind, label in
                                [('gt_raw', 'Raw input'), ('renders_raw', 'Raw render'),
                                 ('depth', 'Depth'), ('normal', 'Normal'), ('depth_normal', 'Depth normal')])
        cards.append(f'<article><h2>scan{scan} <small>CD {report["cd"]:.4f}</small></h2>'
                     f'<a href="scan{scan}/comparison.png"><img loading="lazy" src="scan{scan}/comparison.png" alt="scan{scan}: all three training views"></a>'
                     f'<p>{links} &middot; <a href="scan{scan}/manifest.json">View metadata</a></p></article>')
    overview.save(output_root / 'overview.png')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DTU COLMAP 渲染结果</title><style>body{font:16px system-ui,sans-serif;max-width:1550px;margin:32px auto;padding:0 20px;background:#f3f5f8;color:#172030}article{background:white;padding:18px;margin:24px 0;border-radius:10px}img{width:100%;height:auto}small{font-size:16px;color:#475569}a{color:#225bb3}p{line-height:1.7}</style>
<h1>DTU · 15 个 COLMAP 模型</h1><p>每个模型加载 iteration_10000，在训练视角 0022 / 0025 / 0028 生成图像，分辨率 777×581。没有重新训练，也没有修改已有模型、网格或 CD。</p>
<p><strong>这些是训练视角的定性展示，不是新视角测试。</strong>主对比图按上游导出习惯，对输入和渲染使用同一原图 alpha 前景遮罩；Raw input / Raw render 保留完整画面。法线为相机坐标，颜色表示方向；深度色图按各视角前景 2%–98% 分位数显示，不能直接跨场景比较颜色。原始浮点深度和法线以 NPY 保存。</p>
<p>点击每张对比图查看完整尺寸；每个 scan 文件夹均含三个视角的原始图像与数组。CD 是已有完整几何评估结果。</p>'''
    (output_root / 'index.html').write_text(page + ''.join(cards) + '</html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scans', nargs='+', type=int, choices=SCANS, default=SCANS)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'output/DTU_visuals/colmap_sfm_small_overlap')
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    started = time.time()
    reports = []
    for scan in args.scans:
        print(f'VISUALIZATION_START scan{scan}', flush=True)
        reports.append(export_scan(scan, args.output_root))
        print(f'VISUALIZATION_COMPLETE scan{scan} {len(reports)}/{len(args.scans)}', flush=True)
    make_gallery(reports, args.output_root)
    record = {'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'seconds': time.time() - started,
              'scans': len(reports), 'views': sum(len(r['views']) for r in reports),
              'training_views_only': True, 'retrained': False, 'existing_artifacts_unchanged': True,
              'output': str(args.output_root), 'reports': reports}
    (args.output_root / 'manifest.json').write_text(json.dumps(record, indent=2))
    (ROOT / 'repro/records/dtu_visualizations.json').write_text(json.dumps(record, indent=2))
    print(f'VISUALIZATIONS_DONE {len(reports)} scans, {record["views"]} views, {time.time()-started:.1f}s', flush=True)


if __name__ == '__main__':
    main()

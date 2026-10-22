"""Show real DTU Gaussian parallax next to the saved untextured mesh."""
import argparse
import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp
import torch
from pytorch3d.renderer import (BlendParams, Materials, MeshRasterizer, MeshRenderer,
                               PointLights, RasterizationSettings, SoftPhongShader, TexturesVertex)
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection

from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import readColmapSceneInfo
from utils.camera_utils import cameraList_from_camInfos
from render_dtu_visuals import load_config, fingerprint, font
from run_dtu import SCANS

BG = (.035, .045, .06)


def as_rgb(tensor):
    array = tensor.detach().cpu().numpy()
    assert np.isfinite(array).all()
    return np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)


def paired_frame(rgb, clay, scan, parameter):
    height, width = rgb.shape[:2]
    canvas = Image.new('RGB', (2 * width + 36, height + 92), '#f3f5f8')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f'scan{scan} | interpolated camera | path {parameter:.2f}', font=font(22), fill='#172030')
    draw.text((12, 39), 'Gaussian RGB (no input mask)', font=font(18), fill='#172030')
    draw.text((width + 24, 39), 'Saved mesh / untextured clay', font=font(18), fill='#172030')
    canvas.paste(Image.fromarray(rgb), (12, 66))
    canvas.paste(Image.fromarray(clay), (width + 24, 66))
    draw.text((12, height + 70), 'Same camera. Novel views have no ground-truth image; no new CD is computed.',
              font=font(13), fill='#475569')
    return canvas


@torch.no_grad()
def export(scan, output, frames, width):
    start = time.time()
    model = ROOT / 'output/DTU' / f'scan{scan}_colmap_sfm_small_overlap'
    destination = output / f'scan{scan}'
    destination.mkdir(parents=True, exist_ok=True)
    point = model / 'point_cloud/iteration_10000/point_cloud.ply'
    mesh_path = model / 'train/ours_10000/fuse_post.ply'
    protected = [point, mesh_path, model / 'cfg_args', model / 'train/ours_10000/results.json']
    original = fingerprint(protected)
    args = load_config(model / 'cfg_args')
    info = readColmapSceneInfo(args.source_path, args.images, args.eval, args.n_views, args.dataset_name, args.sfm_type)
    source_cams = cameraList_from_camInfos(info.train_cameras, 1.0, args)
    source_cams.sort(key=lambda cam: cam.image_name)
    assert [cam.image_name for cam in source_cams] == ['0022', '0025', '0028']
    c2ws = np.stack([np.linalg.inv(cam.world_view_transform.T.cpu().numpy()) for cam in source_cams])
    times = np.array([0., .5, 1.])
    rotation = Slerp(times, Rotation.from_matrix(c2ws[:, :3, :3]))
    position = CubicSpline(times, c2ws[:, :3, 3], bc_type='natural')
    height = 2 * round(width * source_cams[0].image_height / source_cams[0].image_width / 2)
    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(str(point))
    assert not args.exposure_compensation
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = torch.as_tensor(np.asarray(mesh.vertices).copy(), dtype=torch.float32, device='cuda')
    faces = torch.as_tensor(np.asarray(mesh.triangles).copy(), dtype=torch.int64, device='cuda')
    geometry = Meshes(verts=[vertices], faces=[faces],
                      textures=TexturesVertex(verts_features=[torch.full_like(vertices, .7)]))
    settings = RasterizationSettings(image_size=(height, width), blur_radius=0., faces_per_pixel=1,
                                     bin_size=32, max_faces_per_bin=100000, cull_backfaces=False)
    material = Materials(device='cuda', shininess=24., specular_color=((.12, .12, .12),))
    renderer = MeshRenderer(MeshRasterizer(raster_settings=settings),
                            SoftPhongShader(device='cuda', materials=material,
                                            blend_params=BlendParams(background_color=BG)))
    background = torch.tensor(BG, device='cuda')

    def render_at(t):
        pose = np.eye(4)
        pose[:3, :3] = rotation(t).as_matrix()
        pose[:3, 3] = position(t)
        w2c = np.linalg.inv(pose)
        fovx = np.interp(t, times, [cam.FoVx for cam in source_cams])
        fovy = np.interp(t, times, [cam.FoVy for cam in source_cams])
        camera = Camera(colmap_id=-1, R=torch.tensor(w2c[:3, :3].T, dtype=torch.float32, device='cuda'),
                        T=torch.tensor(w2c[:3, 3], dtype=torch.float32, device='cuda'),
                        FoVx=fovx, FoVy=fovy, image_width=width, image_height=height,
                        image_path='', image_name='interpolated', uid=0, preload_img=False, normal=None)
        pkg = render(camera, gaussians, args, background, return_plane=False, return_depth_normal=False)
        rgb = as_rgb(pkg['render'].permute(1, 2, 0))
        intrinsic, extrinsic = camera.get_calib_matrix_nerf()
        pytorch_camera = cameras_from_opencv_projection(
            R=extrinsic[:3, :3][None], tvec=extrinsic[:3, 3][None],
            camera_matrix=intrinsic[None], image_size=torch.tensor([[height, width]], device='cuda'))
        # Check the conversion numerically rather than relying on a visual axis guess.
        sample = vertices[::max(1, len(vertices) // 1000)]
        camera_points = sample @ extrinsic[:3, :3].T + extrinsic[:3, 3]
        valid = camera_points[:, 2] > .1
        expected = camera_points[valid] @ intrinsic.T
        expected = expected[:, :2] / expected[:, 2:]
        projected = pytorch_camera.transform_points_screen(sample[valid][None])[0, :, :2]
        assert torch.allclose(expected, projected, atol=.02, rtol=1e-4), 'Mesh and Gaussian camera mismatch'
        lights = PointLights(device='cuda', location=[pose[:3, 3].tolist()],
                             ambient_color=((.35, .35, .35),), diffuse_color=((.65, .65, .65),),
                             specular_color=((.1, .1, .1),))
        clay = as_rgb(renderer(geometry, cameras=pytorch_camera, lights=lights)[0, ..., :3])
        return rgb, clay, pose

    stills = []
    for t in [.17, .47, .79]:
        rgb, clay, pose = render_at(t)
        pair = paired_frame(rgb, clay, scan, t)
        pair.save(destination / f'novel_{t:.2f}.png')
        stills.append(pair)
    board = Image.new('RGB', (stills[0].width, sum(im.height for im in stills)), '#f3f5f8')
    for row, im in enumerate(stills):
        board.paste(im, (0, row * im.height))
    board.save(destination / 'geometry_views.png')
    movie = destination / 'rgb_mesh_motion.mp4'
    gif_frames, poses = [], []
    with imageio.get_writer(str(movie), format='FFMPEG', fps=15, codec='libx264', quality=8,
                            macro_block_size=2, ffmpeg_log_level='error') as writer:
        for index in range(frames):
            # Smooth out-and-back movement inside the observed camera span, excluding original poses.
            t = float(.5 - .48 * np.cos(2 * np.pi * (index + .5) / frames))
            rgb, clay, pose = render_at(t)
            pair = paired_frame(rgb, clay, scan, t)
            writer.append_data(np.asarray(pair))
            poses.append({'path_parameter': t, 'camera_to_world': pose.tolist()})
            if scan in [24, 65, 83, 122] and index % 2 == 0:
                gif_frames.append(pair.resize((988, round(pair.height * 988 / pair.width)), Image.Resampling.LANCZOS))
            if (index + 1) % 15 == 0:
                print(f'GEOMETRY_FRAMES scan{scan} {index+1}/{frames}', flush=True)
    if gif_frames:
        gif_frames[0].save(destination / 'preview.gif', save_all=True, append_images=gif_frames[1:],
                           duration=round(2000 / 15), loop=0, optimize=False)
    assert original == fingerprint(protected), 'Existing trained model or evaluation artifact changed'
    report = {'scan': scan, 'frames': frames, 'fps': 15, 'seconds': time.time() - start,
              'resolution': [width, height], 'mesh_vertices': len(vertices), 'mesh_faces': len(faces),
              'mesh': str(mesh_path.resolve()), 'model': str(point.resolve()), 'poses': poses,
              'camera_projection_alignment_checked': True, 'input_alpha_mask_used': False,
              'novel_pose_parameters_exclude_training_poses': all(min(abs(p['path_parameter'] - key) for key in times) > 1e-6 for p in poses),
              'scope': 'camera interpolation within training coverage; qualitative, no novel-view ground truth',
              'mesh_display': 'Saved fuse_post mesh, uniform gray material, camera-mounted light; no geometry edits',
              'protected_artifacts_unchanged': True, 'output': str(destination.resolve())}
    (destination / 'manifest.json').write_text(json.dumps(report, indent=2))
    del geometry, gaussians, renderer, vertices, faces, source_cams, gif_frames, stills
    torch.cuda.empty_cache()
    return report


def gallery(reports, output):
    cards = []
    for report in reports:
        scan = report['scan']
        cards.append(f'<article><h2>scan{scan}</h2><video controls loop muted playsinline preload="none" poster="scan{scan}/novel_0.47.png" src="scan{scan}/rgb_mesh_motion.mp4"></video>'
                     f'<p><a href="scan{scan}/geometry_views.png">三个插值视角静态图</a> · <a href="scan{scan}/rgb_mesh_motion.mp4">MP4 视频</a></p></article>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DTU 三维结构展示</title><style>body{font:17px system-ui,sans-serif;max-width:1400px;padding:25px;margin:auto;background:#f3f5f8;color:#172030}article{padding:15px;background:white;margin:25px 0;border-radius:10px}video{width:100%}p{line-height:1.7}</style><h1>DTU：移动视角 + 网格白模</h1><p>每帧左侧为已训练 Gaussian 模型直接渲染的 RGB，右侧为已保存 fuse_post.ply 网格的纯灰材质。两侧相机投影已数值核对；没有重新训练、修改几何、使用原图前景遮罩或重新计算 CD。</p><p>相机在训练相机 0022 / 0025 / 0028 之间做位置和旋转插值，视频不重复训练相机的原位姿。它用于查看视差、几何凹凸、孔洞和漂浮碎片；没有这些插值视角的真实图像，不能作为新视角定量评估。白模照明来自相机位置，纹理被去除，灰色和高光仅用于显示已有网格形状。</p>'''
    (output / 'index.html').write_text(page + ''.join(cards) + '</html>', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scans', type=int, nargs='+', choices=SCANS, default=SCANS)
    parser.add_argument('--frames', type=int, default=60)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'output/DTU_geometry/colmap_sfm_small_overlap')
    args = parser.parse_args()
    assert args.frames >= 30 and args.width >= 256
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for scan in args.scans:
        print(f'GEOMETRY_START scan{scan}', flush=True)
        reports.append(export(scan, output, args.frames, args.width))
        print(f'GEOMETRY_COMPLETE scan{scan} {len(reports)}/{len(args.scans)} {reports[-1]["seconds"]:.1f}s', flush=True)
    gallery(reports, output)
    record = {'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scans': len(reports), 'output': str(output), 'reports': reports}
    (output / 'manifest.json').write_text(json.dumps(record, indent=2))
    (ROOT / 'repro/records/dtu_geometry_visualizations.json').write_text(json.dumps(record, indent=2))
    print(f'GEOMETRY_DONE {len(reports)} scans', flush=True)


if __name__ == '__main__':
    main()

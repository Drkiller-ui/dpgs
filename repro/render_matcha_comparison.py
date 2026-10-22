"""Render saved DP-GS and MAtCha models with exactly shared cameras/materials."""
import argparse
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scan', type=int, required=True)
parser.add_argument('--method', choices=['dpgs','matcha','compose'], required=True)
args = parser.parse_args()
DEST = ROOT / f'output/MAtCha_comparison/scan{args.scan}'
DEST.mkdir(parents=True,exist_ok=True)

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)


def board(rows, labels, title):
    h,w=rows[0][0].shape[:2]
    out=Image.new('RGB',(len(labels)*(w+12)+12,len(rows)*(h+36)+62),'#f3f5f8')
    d=ImageDraw.Draw(out)
    d.text((12,8),title,font=font(21),fill='#172030')
    for y,row in enumerate(rows):
        for x,im in enumerate(row):
            d.text((12+x*(w+12),46+y*(h+36)),labels[x],font=font(16),fill='#172030')
            out.paste(Image.fromarray(im),(12+x*(w+12),70+y*(h+36)))
    return out


def compose():
    views=['0022','0025','0028']
    visual=ROOT/f'output/DTU_visuals/colmap_sfm_small_overlap/scan{args.scan}'
    rows=[]
    masked=[]
    for view in views:
        row=[np.array(Image.open(visual/f'gt_raw/{view}.png').convert('RGB'))]
        row += [np.array(Image.open(DEST/f'{m}/train_{view}.png')) for m in ['matcha','dpgs']]
        alpha=np.array(Image.open(visual/f'alpha/{view}.png')).astype(float)/255
        if alpha.ndim==3:
            alpha=alpha[:,:,0]
        masked.append([np.rint(im*alpha[:,:,None]+255*(1-alpha[:,:,None])).astype(np.uint8) for im in row])
        rows.append(row)
    board(rows,['Original RGB','MAtCha','DP-GS'],f'scan{args.scan} | training views 0022 / 0025 / 0028 (top to bottom) | raw RGB').save(DEST/'training_rgb.png')
    board(masked,['Original RGB','MAtCha','DP-GS'],f'scan{args.scan} | same supplied foreground mask on all columns').save(DEST/'training_foreground.png')
    rows=[]
    for index in range(3):
        rows.append([np.array(Image.open(DEST/f'{m}/novel_{index}_{kind}.png')) for kind in ['rgb','clay'] for m in ['matcha','dpgs']])
    labels=['MAtCha RGB','DP-GS RGB','MAtCha mesh','DP-GS mesh']
    board(rows,labels,f'scan{args.scan} | shared novel cameras | no image mask').save(DEST/'novel_rgb_mesh.png')
    readers=[imageio.get_reader(str(DEST/f'{m}/motion.mp4'),format='FFMPEG') for m in ['matcha','dpgs']]
    with imageio.get_writer(str(DEST/'comparison_motion.mp4'),format='FFMPEG',fps=15,codec='libx264',quality=8,macro_block_size=2,ffmpeg_log_level='error') as writer:
        for pair in zip(*readers):
            w=pair[0].shape[1]//2
            row=[pair[0][:,:w],pair[1][:,:w],pair[0][:,w:],pair[1][:,w:]]
            writer.append_data(np.asarray(board([row],labels,f'scan{args.scan} | identical camera, lighting, gray material')))
    for reader in readers:
        reader.close()
    return


if args.method=='compose':
    compose()
    sys.exit(0)

import torch
import open3d as o3d
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,Slerp
from pytorch3d.renderer import (BlendParams,Materials,MeshRasterizer,MeshRenderer,PointLights,
                               RasterizationSettings,SoftPhongShader,TexturesVertex)
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection

if args.method=='matcha':
    sys.path.insert(0,str(ROOT/'baselines/MAtCha/2d-gaussian-splatting'))
    sys.path.insert(1,str(ROOT/'baselines/MAtCha'))
else:
    sys.path.insert(0,str(ROOT))
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from scene.cameras import Camera
from scene.colmap_loader import read_intrinsics_binary,read_extrinsics_binary,qvec2rotmat

scan=args.scan
source=ROOT/f'data/DTU/small_overlap/scan{scan}'
intrinsics=read_intrinsics_binary(source/'sparse/0/cameras.bin')
extrinsics=read_extrinsics_binary(source/'sparse/0/images.bin')
names=sorted(extrinsics,key=lambda k:extrinsics[k].name)
source_specs=[]
for key in names:
    ex=extrinsics[key]
    cam=intrinsics[ex.camera_id]
    w2c=np.eye(4,dtype=np.float32)
    w2c[:3,:3]=qvec2rotmat(ex.qvec)
    w2c[:3,3]=ex.tvec
    fov=[2*np.arctan(cam.width/(2*cam.params[0])),2*np.arctan(cam.height/(2*cam.params[1]))]
    source_specs.append({'c2w':np.linalg.inv(w2c),'fov':fov,'name':Path(ex.name).stem})
assert [s['name'] for s in source_specs]==['0022','0025','0028']
poses=np.stack([s['c2w'] for s in source_specs])
rotation=Slerp([0.,.5,1.],Rotation.from_matrix(poses[:,:3,:3]))
position=CubicSpline([0.,.5,1.],poses[:,:3,3],bc_type='natural')
specs=[]
for t in [.17,.47,.79]:
    pose=np.eye(4)
    pose[:3,:3]=rotation(t).as_matrix()
    pose[:3,3]=position(t)
    specs.append({'c2w':pose,'fov':np.array(source_specs[0]['fov']),'parameter':t})
motion=json.loads((ROOT/f'output/DTU_geometry/colmap_sfm_small_overlap/scan{scan}/manifest.json').read_text())
for entry in motion['poses']:
    t=entry['path_parameter']
    fov=[float(np.interp(t,[0.,.5,1.],[s['fov'][i] for s in source_specs])) for i in range(2)]
    specs.append({'c2w':np.array(entry['camera_to_world']),'fov':fov,'parameter':t})

model=(ROOT/f'output/DTU/scan{scan}_colmap_sfm_small_overlap' if args.method=='dpgs'
       else ROOT/f'output/MAtCha_DTU/colmap_small_overlap/scan{scan}/free_gaussians')
cfg=ast.parse((model/'cfg_args').read_text(),mode='eval').body
assert isinstance(cfg,ast.Call) and isinstance(cfg.func,ast.Name) and cfg.func.id=='Namespace'
config=SimpleNamespace(**{k.arg:ast.literal_eval(k.value) for k in cfg.keywords})
config.convert_SHs_python=False
config.compute_cov3D_python=False
config.debug=False
config.depth_ratio=.5
iteration=10000 if args.method=='dpgs' else 7000
point=model/f'point_cloud/iteration_{iteration}/point_cloud.ply'
mesh_path=(model/'train/ours_10000/fuse_post.ply' if args.method=='dpgs'
           else model.parent/'tetra_meshes/tetra_mesh_binary_search_7.ply')
protected={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in [point,mesh_path,model/'cfg_args']}
gaussians=GaussianModel(config.sh_degree)
gaussians.load_ply(str(point))
mesh=o3d.io.read_triangle_mesh(str(mesh_path))
verts=torch.tensor(np.asarray(mesh.vertices).copy(),device='cuda',dtype=torch.float32)
faces=torch.tensor(np.asarray(mesh.triangles).copy(),device='cuda',dtype=torch.long)
geometry=Meshes(verts=[verts],faces=[faces],textures=TexturesVertex(verts_features=[torch.full_like(verts,.7)]))
BG=(.035,.045,.06)
material=Materials(device='cuda',shininess=24.,specular_color=((.12,.12,.12),))
renderer=MeshRenderer(MeshRasterizer(raster_settings=RasterizationSettings(image_size=(478,640),blur_radius=0.,faces_per_pixel=1,bin_size=64,max_faces_per_bin=len(faces),cull_backfaces=False)),
                      SoftPhongShader(device='cuda',materials=material,blend_params=BlendParams(background_color=BG)))
target=DEST/args.method
target.mkdir(exist_ok=True)


def pixels(t):
    a=t.detach().cpu().numpy()
    assert np.isfinite(a).all()
    return np.rint(np.clip(a,0,1)*255).astype(np.uint8)


def camera(spec,w,h):
    w2c=np.linalg.inv(spec['c2w']).astype(np.float32)
    common=dict(colmap_id=-1,R=w2c[:3,:3].T,T=w2c[:3,3],FoVx=float(spec['fov'][0]),FoVy=float(spec['fov'][1]),image_name='shared_camera',uid=0)
    if args.method=='matcha':
        result=Camera(**common,image=torch.zeros((3,h,w),device='cuda'),gt_alpha_mask=None)
    else:
        common.update(R=torch.tensor(common['R'],device='cuda'),T=torch.tensor(common['T'],device='cuda'))
        result=Camera(**common,image_width=w,image_height=h,image_path='',preload_img=False,normal=None)
    K=torch.tensor([[w/(2*np.tan(spec['fov'][0]/2)),0,w/2],[0,h/(2*np.tan(spec['fov'][1]/2)),h/2],[0,0,1]],device='cuda',dtype=torch.float32)
    return result,K,result.world_view_transform.T.contiguous()


@torch.no_grad()
def render_at(spec,w,h,clay=True):
    cam,K,w2c=camera(spec,w,h)
    kw={} if args.method=='matcha' else dict(return_plane=False,return_depth_normal=False)
    bg=torch.tensor(BG if clay else (0.,0.,0.),device='cuda')
    rgb=pixels(render(cam,gaussians,config,bg,**kw)['render'].permute(1,2,0))
    if not clay:
        return rgb,None
    p3dcam=cameras_from_opencv_projection(R=w2c[:3,:3][None],tvec=w2c[:3,3][None],camera_matrix=K[None],image_size=torch.tensor([[h,w]],device='cuda'))
    sample=verts[::max(1,len(verts)//1000)]
    screen=sample@w2c[:3,:3].T+w2c[:3,3]
    valid=screen[:,2]>.1
    expected=screen[valid]@K.T
    expected=expected[:,:2]/expected[:,2:]
    actual=p3dcam.transform_points_screen(sample[valid][None])[0,:,:2]
    assert torch.allclose(expected,actual,atol=.02,rtol=1e-4)
    lights=PointLights(device='cuda',location=[spec['c2w'][:3,3].tolist()],ambient_color=((.35,.35,.35),),diffuse_color=((.65,.65,.65),),specular_color=((.1,.1,.1),))
    clay_rgb=pixels(renderer(geometry,cameras=p3dcam,lights=lights)[0,:,:,:3])
    return rgb,clay_rgb


for spec in source_specs:
    rgb,_=render_at(spec,777,581,False)
    Image.fromarray(rgb).save(target/f'train_{spec["name"]}.png')
for index,spec in enumerate(specs[:3]):
    rgb,clay=render_at(spec,640,478)
    Image.fromarray(rgb).save(target/f'novel_{index}_rgb.png')
    Image.fromarray(clay).save(target/f'novel_{index}_clay.png')
with imageio.get_writer(str(target/'motion.mp4'),format='FFMPEG',fps=15,codec='libx264',quality=8,macro_block_size=2,ffmpeg_log_level='error') as writer:
    for spec in specs[3:]:
        rgb,clay=render_at(spec,640,478)
        writer.append_data(np.concatenate([rgb,clay],axis=1))
assert protected=={p:[Path(p).stat().st_size,Path(p).stat().st_mtime_ns] for p in protected}
(target/'manifest.json').write_text(json.dumps({'method':args.method,'scan':scan,'model':str(point),'mesh':str(mesh_path),'protected_unchanged':True,'camera_projection_alignment_checked':True,'novel_images_have_ground_truth':False,'novel_mask_used':False,'static_parameters':[.17,.47,.79],'cameras':[{'c2w':s['c2w'].tolist(),'fov':list(s['fov'])} for s in specs]},indent=2))
print(f'COMPARISON_RENDER_COMPLETE scan{scan} {args.method}',flush=True)

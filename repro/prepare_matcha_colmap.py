"""Expose the exact DP-GS COLMAP dense cloud as a calibrated MAtCha scene."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from plyfile import PlyData

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'baselines/MAtCha/mast3r'))
from colmap.read_write_model import (Point3D,read_cameras_binary,read_images_binary,
    write_cameras_binary,write_images_binary,write_points3D_binary,read_points3D_binary,qvec2rotmat)
from stage_matcha_dtu import stage


def prepare(scan,destination):
    stage(scan)
    input_scene=ROOT/f'data/MAtCha_DTU/small_overlap/scan{scan}'
    dense=ROOT/f'data/DTU/small_overlap/scan{scan}/dense/fused.ply'
    cloud=PlyData.read(dense)['vertex'].data
    xyz=np.column_stack([cloud[k] for k in ['x','y','z']])
    rgb=np.column_stack([cloud[k] for k in ['red','green','blue']])
    assert len(xyz)>0 and np.isfinite(xyz).all()
    cameras=read_cameras_binary(input_scene/'sparse/0/cameras.bin')
    original=read_images_binary(input_scene/'sparse/0/images.bin')
    images={}
    tracks=[[] for _ in xyz]
    for key,image in original.items():
        cam=cameras[image.camera_id]
        view=xyz@qvec2rotmat(image.qvec).T+image.tvec
        fx,fy,cx,cy=cam.params
        uv=view[:,:2]/view[:,2:]*[fx,fy]+[cx,cy]
        valid=(view[:,2]>0)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<cam.width)&(uv[:,1]>=0)&(uv[:,1]<cam.height)
        indexes=np.flatnonzero(valid)
        for local,point in enumerate(indexes):
            tracks[point].append((key,local))
        images[key]=image._replace(xys=uv[indexes],point3D_ids=indexes.astype(np.int64)+1)
    points={i+1:Point3D(i+1,xyz[i].astype(float),rgb[i],0.,np.array([a for a,b in track],dtype=np.int32),np.array([b for a,b in track],dtype=np.int32)) for i,track in enumerate(tracks)}
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'sparse/0').mkdir(parents=True,exist_ok=True)
    (destination/'images').mkdir(exist_ok=True)
    for image in images.values():
        shutil.copyfile(input_scene/'images'/image.name,destination/'images'/image.name)
    write_cameras_binary(cameras,destination/'sparse/0/cameras.bin')
    write_images_binary(images,destination/'sparse/0/images.bin')
    write_points3D_binary(points,destination/'sparse/0/points3D.bin')
    ply=destination/'sparse/0/points3D.ply'
    if not ply.exists():
        ply.symlink_to(dense.resolve())
    # Read back the actual file consumed by MAtCha and compare every coordinate/color.
    back=read_points3D_binary(destination/'sparse/0/points3D.bin')
    np.testing.assert_array_equal(np.stack([p.xyz for p in back.values()]).astype(xyz.dtype),xyz)
    np.testing.assert_array_equal(np.stack([p.rgb for p in back.values()]),rgb)
    assert hashlib.sha256(ply.read_bytes()).hexdigest()==hashlib.sha256(dense.read_bytes()).hexdigest()
    manifest={'scan':scan,'initialization':'COLMAP dense fused cloud, identical to DP-GS',
        'source':str(dense.resolve()),'sha256':hashlib.sha256(dense.read_bytes()).hexdigest(),'points':len(xyz),
        'all_xyz_and_rgb_identical_after_binary_roundtrip':True,'initial_ply_identical_to_dpgs':True,
        'camera_poses':'unchanged author COLMAP','input_resolution':[777,581],
        'views':{im.name:len(im.point3D_ids) for im in images.values()},
        'reference_selection':'Existing dense points projected by fixed cameras; positive depth and image field of view only; no learned geometry or GT used',
        'mast3r_executed':False}
    (destination/'init_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    prepare(a.scan,a.output)

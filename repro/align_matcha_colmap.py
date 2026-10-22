"""MAtCha chart alignment using the identical COLMAP cloud, with no MASt3R."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'baselines/MAtCha'
sys.path.insert(0,str(CODE))

import torch
import yaml
from pytorch3d.renderer import PointsRasterizationSettings,PointsRasterizer
from pytorch3d.structures import Pointclouds
from matcha.pointmap.depthanythingv2 import get_pointmap_from_colmap_scene_with_depthanything
from matcha.dm_scene.cameras import CamerasWrapper,create_gs_cameras_from_pointmap,rescale_cameras
from matcha.dm_trainers.charts_alignment import align_charts_in_parallel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    a=p.parse_args()
    torch.manual_seed(0)
    config=yaml.safe_load((CODE/'configs/charts_alignment/default.yaml').read_text())
    pm=config['pointmap']
    pointmap,sfm=get_pointmap_from_colmap_scene_with_depthanything(
        str(a.scene),image_indices=[0,1,2],eval_split=False,max_img_size=pm['max_img_size'],
        pointmap_img_size=pm['pointmap_img_size'],depthanything_encoder='vitl',
        depthanything_checkpoint_dir=str(CODE/'Depth-Anything-V2/checkpoints'),device='cuda',return_sfm_data=True)
    assert torch.isfinite(pointmap.points3d).all()
    camera_list=create_gs_cameras_from_pointmap(pointmap,image_resolution=1,load_gt_images=True,
        max_img_size=pm['max_img_size'],use_original_image_size=True,average_focal_distances=False,verbose=False)
    cameras=CamerasWrapper(camera_list,no_p3d_cameras=False)
    factor=config['scene']['target_scale']/cameras.get_spatial_extent()
    cameras=rescale_cameras(cameras,factor)
    h,w=pointmap.points3d.shape[1:3]
    # A one-pixel footprint, nearest depth only. Every reference depth comes
    # from an existing COLMAP point; holes remain explicitly invalid.
    settings=PointsRasterizationSettings(image_size=(h,w),radius=2./min(h,w),points_per_pixel=1,bin_size=0)
    references=[]
    for i,cam in enumerate(cameras.gs_cameras):
        name=Path(cam.image_name).stem
        points=factor*sfm['sfm_xyz'][sfm['image_sfm_points'][name]]
        fragments=PointsRasterizer(cameras=cameras.p3d_cameras[i],raster_settings=settings)(Pointclouds(points=[points]))
        z=fragments.zbuf[0,:,:,0]
        mask=(fragments.idx[0,:,:,0]>=0)&torch.isfinite(z)&(z>0)
        assert mask.sum()>100,(name,int(mask.sum()))
        references.append(torch.where(mask,z,torch.zeros_like(z)))
    references=torch.stack(references)
    valid=references>0
    assert torch.isfinite(references).all()
    report={'initialization':'Same original COLMAP dense points as DP-GS; native COLMAP loader and DepthAnythingV2 charts',
        'reference_grid':[int(h),int(w)],'valid_reference_pixels':[int(x.sum()) for x in valid],
        'reference_point_raster_radius_pixels':1.,'hole_filling':False,'gt_geometry_used':False,'mast3r_executed':False,
        'alignment_config':config['alignment'],'coordinate_scale':float(factor),
        'note':'COLMAP cloud adapted to the native masked reference-depth API; default alignment losses and weights retained'}
    (a.scene.parent/'colmap_alignment_protocol.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    align_charts_in_parallel(pointmap,reference_data=references,masks=valid.float(),
        rendering_size=pm['max_img_size'],target_scale=config['scene']['target_scale'],
        save_charts_data=True,charts_data_path=str(a.scene),verbose=True,return_training_losses=True,
        **config['alignment'])
    print('COLMAP_CHART_ALIGNMENT_COMPLETE',flush=True)


if __name__=='__main__':
    main()

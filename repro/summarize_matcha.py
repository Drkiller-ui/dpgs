"""Publish a local paired comparison gallery and clearly label partial metrics."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
destination=ROOT/'output/MAtCha_comparison'
destination.mkdir(parents=True,exist_ok=True)
csv_path=ROOT/'output/dtu_matcha_vs_dpgs.csv'
rows=list(csv.DictReader(csv_path.open())) if csv_path.exists() else []
cards=[]
for row in rows:
    scan=int(row['scan'])
    folder=destination/f'scan{scan}'
    if not (folder/'comparison_motion.mp4').exists():
        continue
    cards.append(f'<article><h2>scan{scan}</h2><p>CD ↓：DP-GS {float(row["dpgs_cd"]):.4f}，MAtCha {float(row["matcha_cd"]):.4f}</p>'
                 f'<p>每行从左到右：原图、MAtCha、DP-GS；从上到下：0022、0025、0028。原图为实际训练输入，主图不加前景遮罩。</p>'
                 f'<a href="scan{scan}/training_rgb.png">原图 / MAtCha / DP-GS 完整 RGB</a> · '
                 f'<a href="scan{scan}/training_foreground.png">相同前景遮罩对比</a> · '
                 f'<a href="scan{scan}/novel_rgb_mesh.png">新视角 RGB / 网格白模</a>'
                 f'<p><img loading="lazy" src="scan{scan}/training_rgb.png"></p>'
                 f'<details><summary>新视角 RGB / 网格白模（无实拍原图）</summary><img loading="lazy" src="scan{scan}/novel_rgb_mesh.png"></details>'
                 f'<video controls muted loop preload="none" src="scan{scan}/comparison_motion.mp4"></video></article>')
summary={'requested_scans':15,'evaluated_scans':len(rows),'visualized_scans':len(cards),
         'training_comparison_columns':['Original RGB','MAtCha','DP-GS'],
         'training_comparison_views':['0022','0025','0028'],
         'scope':'Same original COLMAP dense/fused.ply initialization, three RGB inputs and original calibration; no MASt3R',
         'reproduction_status':'Local COLMAP adapter for pinned upstream MAtCha; the DP-GS paper authors modified MAtCha COLMAP entry point was not available. Results are for this documented setup, not claimed as an exact reproduction of their table.',
         'novel_views':'Shared camera poses and intrinsics, qualitative only, no held-out image ground truth',
         'metrics':rows}
if rows:
    summary['paired_dpgs_mean']=float(np.mean([float(r['dpgs_cd']) for r in rows]))
    summary['paired_matcha_mean']=float(np.mean([float(r['matcha_cd']) for r in rows]))
    summary['complete']=len(rows)==15 and len(cards)==15
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=np.arange(len(rows))
    fig,ax=plt.subplots(figsize=(max(8,len(rows)*.8),4.5),layout='constrained')
    ax.bar(x-.18,[float(r['dpgs_cd']) for r in rows],.36,label='DP-GS')
    ax.bar(x+.18,[float(r['matcha_cd']) for r in rows],.36,label='MAtCha')
    ax.set_xticks(x,[r['scan'] for r in rows])
    ax.set_xlabel('DTU scan')
    ax.set_ylabel('Chamfer distance (mm), lower is better')
    ax.set_title(f'DTU paired comparison: {len(rows)}/15 scans evaluated')
    ax.legend()
    ax.grid(axis='y',alpha=.2)
    fig.savefig(destination/'cd_comparison.png',dpi=160)
    plt.close(fig)
(destination/'manifest.json').write_text(json.dumps(summary,indent=2))
text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MAtCha 与 DP-GS 对比</title><style>body{{font:17px system-ui;max-width:1500px;margin:auto;padding:25px;background:#f3f5f8;color:#172030}}article{{background:white;padding:15px;margin:20px 0}}img,video{{max-width:100%}}p{{line-height:1.7}}</style><h1>MAtCha / DP-GS：DTU 同一 COLMAP 点云初始化对比</h1><p>CD 完成 {len(rows)}/15，图像视频完成 {len(cards)}/15。只有 15 个场景全部评估完成后才是完整均值，过程中均值只对应两种方法共同完成的场景。</p><p>两种方法使用完全相同的 COLMAP dense/fused.ply、0022 / 0025 / 0028 输入 RGB 与已有 COLMAP 标定。MAtCha 通过原生 COLMAP 载入和 DepthAnythingV2 构建图表；点云投影为带有效性遮罩的参考深度，保留默认对齐损失和权重，未运行 MASt3R。图表对齐 1000 步、高斯细化 7000 步；DP-GS 已有结果为 10000 步。CD 复用同一 DTU 评估流程。新视角使用共同相机；网格用相同纯灰材质和相机照明，展示不修改几何。插值视角无真实图像，只作定性展示。</p>'''
if rows:
    text+=f'<p>当前配对均值：DP-GS {summary["paired_dpgs_mean"]:.4f}，MAtCha {summary["paired_matcha_mean"]:.4f}。</p><img src="cd_comparison.png">'
text+='<p>复现范围：这里使用公开 MAtCha 代码及本地 COLMAP 适配入口，尚未取得 DP-GS 论文作者修改后的 MAtCha COLMAP 入口。分数对应此处记录的配置，不代表已精确复现论文表格。CD 沿用现有 DP-GS 评估中的 mask_cull=False 设置，不额外使用输入图像的前景掩码裁剪网格。</p>'
(destination/'index.html').write_text(text+''.join(cards)+'</html>',encoding='utf-8')
markdown=['# 原图 / MAtCha / DP-GS：全部 15 组对比',
          '每张图从左到右为 **原图、MAtCha、DP-GS**，从上到下为训练视角 **0022、0025、0028**。主图保留完整 RGB，不添加前景遮罩；点击图片可查看原尺寸。',
          '三列使用同一相机和分辨率。原图是实际训练输入，因此这些图展示训练视角的拟合效果；插值新视角没有对应实拍原图，另提供两种方法的 RGB、白模和运动视频。',
          '两种方法使用同一份 COLMAP 点云。本次 MAtCha 使用公开代码的本地 COLMAP 适配入口，配置说明见 [复现记录](../../repro/MATCHA.md)，不声称精确复现论文表格。',
          '[逐场景 CSV](../dtu_matcha_vs_dpgs.csv) · [HTML 图像视频总览](index.html)（克隆后用浏览器打开）']
if rows:
    markdown += [f'平均 CD（mm，越低越好；{len(rows)}/15 组）：MAtCha **{summary["paired_matcha_mean"]:.6f}**，DP-GS **{summary["paired_dpgs_mean"]:.6f}**。',
                 '\n'.join(['| Scan | MAtCha CD ↓ | DP-GS CD ↓ |', '|---|---:|---:|']+
                           [f'| {r["scan"]} | {float(r["matcha_cd"]):.4f} | {float(r["dpgs_cd"]):.4f} |' for r in rows])]
for row in rows:
    scan=int(row['scan'])
    if not (destination/f'scan{scan}/training_rgb.png').exists():
        continue
    markdown += [f'## scan{scan}',
                 f'MAtCha CD：{float(row["matcha_cd"]):.4f}；DP-GS CD：{float(row["dpgs_cd"]):.4f}。',
                 f'![scan{scan}：原图、MAtCha、DP-GS，三行对应 0022、0025、0028](scan{scan}/training_rgb.png)',
                 f'[相同前景遮罩对比](scan{scan}/training_foreground.png) · [新视角 RGB / 白模](scan{scan}/novel_rgb_mesh.png) · [相机运动对比视频](scan{scan}/comparison_motion.mp4)']
(destination/'README.md').write_text('\n\n'.join(markdown)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},ensure_ascii=False),flush=True)

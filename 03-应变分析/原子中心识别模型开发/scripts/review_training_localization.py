"""Read-only localization audit; overlays are predictions, never revised labels."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from atom_center.training import checkpoint_for_run
from atom_center.backends import TorchBackend
from atom_center.pipeline import DetectionPipeline
from atom_center.configuration import pipeline_config
from atom_center.data_workflow import verify_dataset
from atom_center.image_io import load_image, normalize_percentile
from atom_center.storage import read_json, write_json
from atom_center.metrics import match_points

def main():
    root=Path(__file__).resolve().parents[1];out=root/'runs/generalization-20260910/localization_audit';out.mkdir(exist_ok=True)
    m,p,h=checkpoint_for_run(root/'runs/point-selection-20260909',checkpoint='best_points.pt')
    pipe=DetectionPipeline(TorchBackend(p,expected_sha256=h,contract=m['contract'],inference=m['config']['inference'],device='cuda:0'),pipeline_config(m['config']))
    frozen,data=verify_dataset(root/'data/processed/real_workflow_20260909_v2')
    names={r['record_key']:Path(r['source_image']).name for r in read_json(data/'audit_report.json')['records']}
    rows=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    for s in frozen['sources']:
        name=names[s['record_key']]
        if not any(k in name for k in ('0051','0063','0017','0030')):continue
        raw=load_image(data/s['raw'],normalize=False).image;truth=read_json(data/s['truth']);gt=np.array(truth['points_xy'])
        for ri,roi in enumerate(truth['regions_xyxy']):
            x0,y0,x1,y1=map(int,roi);g=gt[(gt[:,0]>=x0)&(gt[:,0]<x1)&(gt[:,1]>=y0)&(gt[:,1]<y1)]
            result=pipe.detect(raw,roi=roi);ps=result.points
            mat=match_points(ps,g,max_distance_px=6.);pairs=mat.matched_indices;delta=ps[pairs[:,0]]-g[pairs[:,1]]
            row={'image':name,'roi':roi,'split':s['split'],'model_sha256':h,'prediction_xy':ps.tolist(),'manual_xy':g.tolist(),
                 'diagnostic_match_radius_px':6.,'diagnostic_matched_pairs':len(pairs),'median_delta_xy_px':np.median(delta,axis=0).tolist(),
                 'median_distance_px':float(np.median(mat.distances_px)), 'truth_nearest_distance_median':float(np.median(cKDTree(g).query(g,k=2)[0][:,1]))}
            rows.append(row)
            base=Image.fromarray(np.rint(normalize_percentile(raw[y0:y1,x0:x1])*255).astype('uint8')).convert('RGB');d=ImageDraw.Draw(base)
            for x,y in g-[x0,y0]:d.ellipse((x-1,y-1,x+1,y+1),fill='#ff5050')
            for x,y in ps-[x0,y0]:d.ellipse((x-3,y-3,x+3,y+3),outline='#00ff88')
            base.save(out/f'{s["record_key"]}_{ri}_overlay.png')
            panels=[]
            # Spatially spread regular samples, independent of measured errors.
            for index in np.linspace(0,len(g)-1,12,dtype=int):
                x,y=g[index];cx,cy=map(int,(x,y));xa,ya=max(0,cx-25),max(0,cy-25);xb,yb=min(raw.shape[1],cx+26),min(raw.shape[0],cy+26)
                pic=Image.fromarray(np.rint(normalize_percentile(raw[ya:yb,xa:xb])*255).astype('uint8')).convert('RGB').resize(((xb-xa)*5,(yb-ya)*5))
                d=ImageDraw.Draw(pic)
                for points,color,radius in ((g,'#ff5050',4),(ps,'#00ff88',8)):
                    for px,py in points:
                        if xa<=px<xb and ya<=py<yb:
                            xx,yy=(px-xa+.5)*5,(py-ya+.5)*5;d.ellipse((xx-radius,yy-radius,xx+radius,yy+radius),outline=color,width=2)
                panel=Image.new('RGB',(275,295),'#111827');panel.paste(pic,(10,32));ImageDraw.Draw(panel).text((8,5),f'x={x:.1f}, y={y:.1f}',font=font,fill='white');panels.append(panel)
            canvas=Image.new('RGB',(1100,945),'#111827');d=ImageDraw.Draw(canvas);d.text((10,5),name,font=font,fill='white');d.text((10,30),'红圈：人工标注；绿圈：固定模型。5倍放大；6px仅用于诊断，不改变验收标准。',font=font,fill='white')
            for i,pic in enumerate(panels):canvas.paste(pic,((i%4)*275,60+(i//4)*295))
            canvas.save(out/f'{s["record_key"]}_{ri}_detail.png');print(name,row['median_delta_xy_px'],row['median_distance_px'],flush=True)
    write_json(out/'audit.json',rows)

if __name__=='__main__':main()

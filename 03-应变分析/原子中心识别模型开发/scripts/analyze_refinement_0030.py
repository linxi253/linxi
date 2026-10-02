"""Fixed-model refinement ablation on validation data, without editing labels."""
from pathlib import Path
from dataclasses import replace
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from atom_center.storage import read_json,write_json
from atom_center.data_workflow import verify_dataset
from atom_center.configuration import pipeline_config
from atom_center.training import checkpoint_for_run
from atom_center.backends import TorchBackend
from atom_center.pipeline import DetectionPipeline
from atom_center.image_io import load_image,normalize_percentile
from atom_center.refinement import refine_with_diagnostics
from atom_center.metrics import match_points
from atom_center.model_manifest import sha256_file
from atom_center.source_identity import verify_source

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



def metric(pred,gt):
    m=match_points(pred,gt,max_distance_px=2.)
    tp,fp,fn=m.true_positives,m.false_positives,m.false_negatives
    return {'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/(2*tp+fp+fn) if tp+fp+fn else 1.,
            'rmse_px':float(np.sqrt(np.mean(m.distances_px**2))) if tp else None}


def main():
    root=Path(__file__).resolve().parents[1];out=root/'runs/refinement-0030-20260909';out.mkdir(exist_ok=False)
    frozen,data_root=verify_dataset(root/'data/processed/real_workflow_20260909_v2')
    audit=read_json(data_root/'audit_report.json');original={s['record_key']:s for s in audit['records']}
    manifest,path,digest=checkpoint_for_run(root/'runs/box-comparison-20260909/box8')
    backend=TorchBackend(path,expected_sha256=digest,contract=manifest['contract'],inference=manifest['config']['inference'],device='cuda:0')
    pipe=DetectionPipeline(backend,replace(pipeline_config(manifest['config']),refine=False))
    cases=[]
    for source in frozen['sources']:
        if source['split']!='val':continue
        raw=load_image(data_root/source['raw'],normalize=False).image
        truth=read_json(data_root/source['truth']);points=np.array(truth['points_xy'])
        for i,roi in enumerate(truth['regions_xyxy']):
            x0,y0,x1,y1=roi;gt=points[(points[:,0]>=x0)&(points[:,0]<x1)&(points[:,1]>=y0)&(points[:,1]<y1)]
            result=pipe.detect(raw,roi=roi)
            cases.append({'name':Path(original[source['record_key']]['source_image']).name,'record_key':source['record_key'],
                'roi_index':i,'roi':roi,'raw':raw,'gt':gt,'candidates':result.points.copy()})
    grid=[('none',0)]+[('com',w) for w in (3,5,7,9,11)]+[('gaussian',w) for w in (5,7,9)]
    rows=[];focus_results={}
    for method,window in grid:
        regions=[]
        for c in cases:
            if method=='none':pred=c['candidates'].copy();quality=[]
            else:
                refined=refine_with_diagnostics(c['raw'],c['candidates'],method=method,window_size=window,polarity='bright',max_shift_px=2.4)
                pred=refined.points;quality=refined.diagnostics
            x0,y0,x1,y1=c['roi'];pred=pred[(pred[:,0]>=x0)&(pred[:,0]<x1)&(pred[:,1]>=y0)&(pred[:,1]<y1)]
            regions.append({'name':c['name'],'record_key':c['record_key'],'roi_index':c['roi_index'],**metric(pred,c['gt']),
                            'quality_counts':dict(Counter(q['status'] for q in quality))})
            if c['name']=='0030 HAADF.tif':focus_results[f'{method}{window}']=pred
        tp,fp,fn=(sum(r[k] for r in regions) for k in ('tp','fp','fn'))
        row={'method':method,'window':window,'polarity':'bright','max_shift_px':2.4,'tp':tp,'fp':fp,'fn':fn,
             'f1':2*tp/(2*tp+fp+fn),'precision':tp/(tp+fp) if tp+fp else 1.,'recall':tp/(tp+fn),
             'rmse_px':float(np.sqrt(sum(r['tp']*(r['rmse_px'] or 0)**2 for r in regions)/tp)) if tp else None,'regions':regions}
        rows.append(row);write_json(out/'grid.json',rows);print('GRID',method,window,tp,fp,fn,row['f1'],flush=True)
    focus=next(c for c in cases if c['name']=='0030 HAADF.tif')
    x0,y0,x1,y1=map(int,focus['roi']);raw=focus['raw'];gt=focus['gt'];cand=focus['candidates']
    correspondence=match_points(cand,gt,max_distance_px=6.)
    pairs=correspondence.matched_indices;deltas=cand[pairs[:,0]]-gt[pairs[:,1]]
    label_probe=refine_with_diagnostics(raw,gt,method='gaussian',window_size=9,polarity='bright',max_shift_px=2.4)
    successful=np.array([d['status']=='refined' for d in label_probe.diagnostics])
    shifts=label_probe.points[successful]-gt[successful]
    dark=refine_with_diagnostics(raw,cand,method='com',window_size=7,polarity='dark',max_shift_px=2.4)
    report={'purpose':'validation_refinement_ablation','model_sha256':digest,'test_images_used':False,
        'label_changes':False,'grid':rows,'0030_diagnostics':{
            'roi':focus['roi'],'raw_candidates':len(cand),'truth_points':len(gt),
            'six_px_correspondence_only_not_acceptance':{'matches':len(pairs),'median_delta_xy':np.median(deltas,axis=0).tolist(),
                'p95_distance_px':float(np.percentile(np.linalg.norm(deltas,axis=1),95))},
            'label_to_local_bright_gaussian_probe':{'successful_fits':int(successful.sum()),'median_shift_xy':np.median(shifts,axis=0).tolist(),
                'median_shift_distance_px':float(np.median(np.linalg.norm(shifts,axis=1))),'note':'Image-based diagnostic only; does not automatically certify or modify manual truth.'},
            'dark_com7_diagnostic_only':metric(dark.points,gt)}}
    for source in original.values():
        verify_source(source['source_image'],source['source_identity'])
        assert sha256_file(source['label_file'])==source['label_sha256']
    write_json(out/'report.json',report)
    best=max(rows,key=lambda r:r['f1']);selected=f"{best['method']}{best['window']}"
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    panels=[]
    for key,title in [('none0','未精修'),('com7','原 COM / 7像素窗口'),(selected,f"全验证集最优：{best['method']} / {best['window']}")]:
        pic=Image.fromarray(np.rint(normalize_percentile(raw[y0:y1,x0:x1])*255).astype('uint8')).convert('RGB')
        draw=ImageDraw.Draw(pic)
        for x,y in gt-[x0,y0]:draw.ellipse((x-1,y-1,x+1,y+1),fill='#ff5050')
        for x,y in focus_results[key]-[x0,y0]:draw.ellipse((x-3,y-3,x+3,y+3),outline='#00ff88',width=1)
        scale=min(700/pic.width,700/pic.height);pic=pic.resize((round(pic.width*scale),round(pic.height*scale)),Image.Resampling.NEAREST)
        panel=Image.new('RGB',(740,830),'#111827');panel.paste(pic,((740-pic.width)//2,120));pd=ImageDraw.Draw(panel)
        m=metric(focus_results[key],gt);pd.text((16,12),title,font=font,fill='white');pd.text((16,47),f"TP={m['tp']}  FP={m['fp']}  FN={m['fn']}",font=font,fill='#fbbf24')
        pd.text((16,79),'绿圈：预测；红点：原始人工标注',font=font,fill='#cbd5e1');panels.append(panel)
    canvas=Image.new('RGB',(2220,830))
    for i,panel in enumerate(panels):canvas.paste(panel,(740*i,0))
    canvas.save(out/'0030_comparison.png')
    # Unmarked raw view and original manual centers for independent visual inspection.
    pic=Image.fromarray(np.rint(normalize_percentile(raw[y0:y1,x0:x1])*255).astype('uint8')).convert('RGB')
    pic.resize((pic.width*2,pic.height*2)).save(out/'0030_raw.png')
    write_json(out/'0030_points.json',{'roi':focus['roi'],'truth_xy':gt.tolist(),'unrefined_xy':cand.tolist(),
                'predictions':{k:v.tolist() for k,v in focus_results.items()}})
    print('BEST',best['method'],best['window'],best['f1'],flush=True)


if __name__=='__main__':main()

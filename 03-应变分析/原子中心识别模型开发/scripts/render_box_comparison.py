"""Render validation-only results from the completed controlled training comparison."""
from pathlib import Path
import argparse
from dataclasses import replace
import csv
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from atom_center.storage import read_json,write_json
from atom_center.training import checkpoint_for_run
from atom_center.backends import TorchBackend
from atom_center.configuration import pipeline_config
from atom_center.pipeline import DetectionPipeline
from atom_center.data_workflow import verify_dataset
from atom_center.image_io import load_image,normalize_percentile
from atom_center.metrics import match_points
from atom_center.source_identity import verify_source
from atom_center.model_manifest import sha256_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve()
    completed=read_json(run/'completed.json')
    rows=completed['validation_comparison']
    selected=max(rows,key=lambda r:(r['f1'],r['precision'],-r['box_size_px'],-r['iou']))
    box=selected['box_size_px'];trial=run/f'box{box}'
    manifest,path,digest=checkpoint_for_run(trial)
    root=Path(__file__).resolve().parents[1]
    frozen,data_root=verify_dataset(root/'data/processed/real_workflow_20260909_v2')
    audit=read_json(data_root/'audit_report.json');lookup={s['record_key']:s for s in audit['records']}
    for source in lookup.values():
        verify_source(source['source_image'],source['source_identity'])
        assert sha256_file(source['label_file'])==source['label_sha256']
    backend=TorchBackend(path,expected_sha256=digest,contract=manifest['contract'],inference={**manifest['config']['inference'],'iou':selected['iou']},device='cuda:0')
    config=replace(pipeline_config(manifest['config']),refine=selected['refinement']=='com')
    pipe=DetectionPipeline(backend,config)
    output=run/'validation_preview';output.mkdir(exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    big=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',30)
    panels=[];predictions=[]
    for source in frozen['sources']:
        if source['split']!='val':continue
        original=lookup[source['record_key']]
        raw=load_image(data_root/source['raw'],normalize=False).image
        truth=read_json(data_root/source['truth']);expected=np.array(truth['points_xy'])
        for index,roi in enumerate(truth['regions_xyxy']):
            x0,y0,x1,y1=map(int,roi);gt=expected[(expected[:,0]>=x0)&(expected[:,0]<x1)&(expected[:,1]>=y0)&(expected[:,1]<y1)]
            result=pipe.detect(raw,roi=roi)
            match=match_points(result.points,gt,max_distance_px=2.)
            pixels=np.rint(normalize_percentile(raw[y0:y1,x0:x1])*255).astype('uint8')
            pic=Image.fromarray(pixels).convert('RGB');draw=ImageDraw.Draw(pic)
            for x,y in gt-[x0,y0]: draw.ellipse((x-2,y-2,x+2,y+2),fill='#ff5a64')
            for x,y in result.points-[x0,y0]:draw.ellipse((x-4,y-4,x+4,y+4),outline='#00ff8c',width=1)
            name=f"{source['record_key']}_r{index}"
            pic.save(output/f'{name}.png')
            payload={'image':Path(original['source_image']).name,'record_key':source['record_key'],'roi':roi,
                'model_sha256':digest,'points_xy':result.points.tolist(),'confidences':result.confidences.tolist(),
                'tp':match.true_positives,'fp':match.false_positives,'fn':match.false_negatives,
                'truth_points':len(gt),'metadata':dict(result.metadata),'overlay':f'{name}.png'}
            predictions.append(payload);write_json(output/f'{name}.json',payload)
            panel=Image.new('RGB',(620,620),'#111827');pd=ImageDraw.Draw(panel)
            pd.text((12,10),f"{payload['image']} / ROI {index+1}",font=font,fill='white')
            pd.text((12,43),f"标注 {len(gt)} / 检出 {len(result.points)}",font=font,fill='#cbd5e1')
            pd.text((12,76),f"匹配 {match.true_positives} / 误检 {match.false_positives} / 漏检 {match.false_negatives}",font=font,fill='#fbbf24')
            pic.thumbnail((592,480),Image.Resampling.LANCZOS);panel.paste(pic,((620-pic.width)//2,125+(480-pic.height)//2));panels.append(panel)
    overview=Image.new('RGB',(1860,115+620*((len(panels)+2)//3)),'#090e18');d=ImageDraw.Draw(overview)
    d.text((18,10),'原有验证集 · 对照实验中点位 F1 最佳方案',font=big,fill='white')
    d.text((18,54),f"框 {box}px | IoU {selected['iou']} | {selected['refinement']} | 置信度 0.25 | 原图匹配距离 2px",font=font,fill='#cbd5e1')
    d.text((18,83),'绿色圆圈：预测中心；红色点：人工标注。验证集用于模型选择，非独立测试结果。',font=font,fill='#cbd5e1')
    for i,panel in enumerate(panels):overview.paste(panel,((i%3)*620,115+(i//3)*620))
    overview.save(output/'overview.png')
    write_json(output/'summary.json',{'selected_on':'validation_point_f1','selected_configuration':selected,'original_sources_unchanged':True,'test_images_used':False,'regions':predictions})
    plt.rcParams['font.family']='Microsoft YaHei'
    fig,axes=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
    for b in (8,16):
        with (run/f'box{b}/training/results.csv').open() as stream:epochs=list(csv.DictReader(stream))
        for ax,key,title in zip(axes,['train/cls_loss','val/cls_loss','metrics/mAP50(B)'],['训练分类损失','验证分类损失','验证框 mAP50']):
            ax.plot([int(e['epoch']) for e in epochs],[float(e[key]) for e in epochs],label=f'{b}px 框');ax.set_title(title);ax.set_xlabel('轮次');ax.grid(alpha=.2);ax.legend()
    fig.savefig(run/'training_curves.png',dpi=160)
    print(selected,flush=True)


if __name__=='__main__':main()

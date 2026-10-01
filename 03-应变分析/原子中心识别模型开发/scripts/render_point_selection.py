"""Render the selected checkpoint on validation images and its selection history."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from atom_center.storage import read_json,write_json
from atom_center.training import checkpoint_for_run
from atom_center.backends import TorchBackend
from atom_center.pipeline import DetectionPipeline
from atom_center.configuration import pipeline_config
from atom_center.data_workflow import verify_dataset
from atom_center.image_io import load_image,normalize_percentile
from atom_center.metrics import match_points
from atom_center.model_manifest import sha256_file
from atom_center.source_identity import verify_source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/point-selection-20260909'))
    parser.add_argument('--split',choices=['train','val'],default='val')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--manifest',type=Path,help='Render an ONNX refinement variant of the selected checkpoint')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];run=args.run.resolve()
    out=args.output.resolve() if args.output else run/f'{args.split}_preview';out.mkdir(exist_ok=False)
    m,path,digest=checkpoint_for_run(run,checkpoint='best_points.pt')
    backend=TorchBackend(path,expected_sha256=digest,contract=m['contract'],inference=m['config']['inference'],device='cuda:0')
    pipe=DetectionPipeline(backend,pipeline_config(m['config']))
    if args.manifest:
        from atom_center.backends import onnx_pipeline
        pipe=onnx_pipeline(args.manifest)
        assert pipe.backend.manifest.metrics['source_checkpoint_sha256']==digest
        digest=pipe.backend.model_sha256
        m['config']['refinement']=dict(pipe.backend.manifest.refinement)
        m['config']['inference']=dict(pipe.backend.manifest.inference)
    frozen,data_root=verify_dataset(root/'data/processed/real_workflow_20260909_v2')
    originals={s['record_key']:s for s in read_json(data_root/'audit_report.json')['records']}
    for s in originals.values():
        verify_source(s['source_image'],s['source_identity']);assert sha256_file(s['label_file'])==s['label_sha256']
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22);big=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',30)
    panels=[];rows=[]
    for s in frozen['sources']:
        if s['split']!=args.split:continue
        raw=load_image(data_root/s['raw'],normalize=False).image;truth=read_json(data_root/s['truth']);all_gt=np.array(truth['points_xy'])
        for i,roi in enumerate(truth['regions_xyxy']):
            x0,y0,x1,y1=map(int,roi);gt=all_gt[(all_gt[:,0]>=x0)&(all_gt[:,0]<x1)&(all_gt[:,1]>=y0)&(all_gt[:,1]<y1)]
            result=pipe.detect(raw,roi=roi)
            match=match_points(result.points,gt,max_distance_px=2.)
            image_name=Path(originals[s['record_key']]['source_image']).name
            pic=Image.fromarray(np.rint(normalize_percentile(raw[y0:y1,x0:x1])*255).astype('uint8')).convert('RGB');d=ImageDraw.Draw(pic)
            for x,y in gt-[x0,y0]:d.ellipse((x-1,y-1,x+1,y+1),fill='#ff5050')
            for x,y in result.points-[x0,y0]:d.ellipse((x-3,y-3,x+3,y+3),outline='#00ff88',width=1)
            name=f"{s['record_key']}_r{i}"
            pic.save(out/f'{name}.png')
            row={'image':image_name,'roi':roi,'record_key':s['record_key'],'roi_index':i,'points_xy':result.points.tolist(),
                 'confidences':result.confidences.tolist(),'metadata':dict(result.metadata),'model_sha256':digest,
                 'truth_points':len(gt),'tp':match.true_positives,'fp':match.false_positives,'fn':match.false_negatives,'overlay':f'{name}.png'}
            rows.append(row);write_json(out/f'{name}.json',row)
            panel=Image.new('RGB',(620,620),'#111827');d=ImageDraw.Draw(panel)
            d.text((12,10),f'{image_name} / ROI {i+1}',font=font,fill='white')
            d.text((12,42),f'真值 {len(gt)} / 检出 {len(result.points)}',font=font,fill='#cbd5e1')
            d.text((12,75),f'TP {match.true_positives} / FP {match.false_positives} / FN {match.false_negatives}',font=font,fill='#fbbf24')
            scale=min(580/pic.width,480/pic.height);pic=pic.resize((round(pic.width*scale),round(pic.height*scale)),Image.Resampling.LANCZOS)
            panel.paste(pic,((620-pic.width)//2,125+(480-pic.height)//2));panels.append(panel)
    canvas=Image.new('RGB',(1860,115+620*((len(panels)+2)//3)),'#090e18');d=ImageDraw.Draw(canvas)
    selection=read_json(run/'point_validation/selection.json');epoch=selection['best']['epoch']
    ref=m['config']['refinement'];inf=m['config']['inference']
    d.text((18,10),f'点位指标选择模型 · {"训练集" if args.split=="train" else "验证集"}预测',font=big,fill='white')
    d.text((18,52),f'第{epoch}轮 | {ref["method"]} {ref["window_size"]} / 最大位移{ref["max_shift_px"]}px | 置信度{inf["conf"]} | 原图匹配距离2px',font=font,fill='#cbd5e1')
    d.text((18,83),'绿圈：预测中心；红点：人工标注。仅统计人工审核 ROI；不是独立测试成绩。',font=font,fill='#cbd5e1')
    for i,p in enumerate(panels):canvas.paste(p,((i%3)*620,115+(i//3)*620))
    canvas.save(out/'overview.png')
    write_json(out/'summary.json',{'split':args.split,'model_sha256':digest,'original_files_unchanged':True,'test_used':False,'regions':rows})
    history=read_json(run/'point_validation/selection.json');epochs=[r['epoch'] for r in history['history']]
    plt.rcParams['font.family']='Microsoft YaHei'
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for k,label in [('f1','F1'),('recall','召回率'),('precision','精确率')]:
        values=[np.nan if k=='precision' and r['metrics']['tp']+r['metrics']['fp']==0 else r['metrics'][k] for r in history['history']]
        axes[0].plot(epochs,values,label=label)
    axes[0].axvline(history['best']['epoch'],linestyle='--',color='black',label='点位最优轮次');axes[0].legend();axes[0].set_title('原图验证点位指标');axes[0].set_ylim(0,1.05)
    axes[1].plot(epochs,[r['metrics']['rmse_px'] if r['metrics']['rmse_px'] is not None else np.nan for r in history['history']]);axes[1].set_title('成功匹配点 RMSE（原图像素）')
    for ax in axes:ax.set_xlabel('轮次');ax.grid(alpha=.2)
    fig.savefig(run/'point_selection_curves.png',dpi=160)


if __name__=='__main__':main()

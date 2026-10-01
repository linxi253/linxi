"""Audit user-reviewed CSV answers against frozen full-image model predictions.

Primary metrics never rerun/tune the model. ROI inference is a separate diagnostic.
Only the exported rectangle is scored; CSV points are never shifted or refined.
"""
from pathlib import Path
import argparse
import csv
from collections import Counter
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from atom_center.backends import onnx_pipeline
from atom_center.image_io import load_image, normalize_percentile
from atom_center.metrics import match_points
from atom_center.source_identity import verify_source
from atom_center.storage import read_json, write_json, new_artifact_directory


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(points, bounds):
    x0, y0, x1, y1 = bounds
    return ((points[:, 0] >= x0) & (points[:, 0] <= x1)
            & (points[:, 1] >= y0) & (points[:, 1] <= y1))


def score(prediction, truth, radius):
    m = match_points(prediction, truth, max_distance_px=radius)
    tp, fp, fn = m.true_positives, m.false_positives, m.false_negatives
    return {'match_distance_px':radius, 'tp':tp, 'fp':fp, 'fn':fn,
            'precision':tp/(tp+fp) if tp+fp else 1.,
            'recall':tp/(tp+fn) if tp+fn else 1.,
            'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
            'rmse_px':float(np.sqrt(np.mean(m.distances_px**2))) if tp else None,
            'p95_px':float(np.percentile(m.distances_px,95)) if tp else None,
            'matched_indices':m.matched_indices.tolist(),
            'unmatched_prediction_indices':m.unmatched_prediction_indices.tolist(),
            'unmatched_truth_indices':m.unmatched_ground_truth_indices.tolist()}


def draw_overlay(raw, truth, prediction, bounds, metrics, title, target):
    base = Image.fromarray(np.rint(normalize_percentile(raw)*255).astype('uint8')).convert('RGB')
    d = ImageDraw.Draw(base)
    d.rectangle(tuple(bounds), outline='#62b9ff', width=2)
    for index in metrics['unmatched_truth_indices']:
        x,y=truth[index];d.line((x-3,y,x+3,y),fill='#ff5e64',width=1);d.line((x,y-3,x,y+3),fill='#ff5e64',width=1)
    for index in metrics['unmatched_prediction_indices']:
        x,y=prediction[index];d.ellipse((x-3,y-3,x+3,y+3),outline='#ffce54',width=1)
    for pidx,tidx in metrics['matched_indices']:
        x,y=prediction[pidx];d.ellipse((x-3,y-3,x+3,y+3),outline='#26e7a3',width=1)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    panel=Image.new('RGB',(max(raw.shape[1],850),raw.shape[0]+130),'#101827');panel.paste(base,(0,130));d=ImageDraw.Draw(panel)
    for i,line in enumerate((title,'2px 一对一匹配 | 蓝框：评测选区；红十字：未匹配答案；黄圈：未匹配预测；绿圈：匹配',
                             f'答案 {len(truth)} / 预测 {len(prediction)} | TP {metrics["tp"]} FP {metrics["fp"]} FN {metrics["fn"]}',
                             f'精确率 {metrics["precision"]:.2%} | 召回率 {metrics["recall"]:.2%} | 未匹配同时包含定位偏差')):
        d.text((12,8+i*29),line,font=font,fill='white')
    panel.save(target)
    return panel


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--answers',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    preview=root/'runs/generalization-20260910/final_testset_preview'
    model=root/'runs/generalization-20260910/adaptive_blob-onnx/model_manifest.json'
    pipe=onnx_pipeline(model)
    index={r['name']:(i+1,r) for i,r in enumerate(read_json(preview/'summary.json')['images'])}
    answer_files=sorted(args.answers.iterdir());hashes={str(p.resolve()):digest(p) for p in answer_files if p.is_file()}
    rows=[];panels=[]
    with new_artifact_directory(args.output.resolve()) as out:
        for metadata_file in sorted(args.answers.glob('*_atoms.metadata.json')):
            meta=read_json(metadata_file);source=Path(meta['source_tiff']);number,entry=index[source.name]
            predfile=preview/entry['prediction'];stored=read_json(predfile)
            assert stored['model_sha256']==pipe.backend.model_sha256
            verify_source(source,stored['source_identity'])
            raw=load_image(source,normalize=False,preserve_dtype=True).image
            csv_file=metadata_file.with_name(metadata_file.name.replace('.metadata.json','.csv'))
            with csv_file.open(encoding='utf-8-sig',newline='') as stream:records=list(csv.DictReader(stream))
            assert meta['frame_count']==1 and all(r['帧号(从1开始)']=='1' for r in records)
            keys=[(r['帧号(从1开始)'],r['原子ID']) for r in records];assert len(keys)==len(set(keys))
            truth=np.asarray([(float(r['x_px']),float(r['y_px'])) for r in records],dtype=float).reshape(-1,2)
            assert len(truth) and np.isfinite(truth).all() and len(np.unique(truth,axis=0))==len(truth)
            assert inside(truth,[0,0,raw.shape[1]-1,raw.shape[0]-1]).all()
            region=meta['frame_regions'][0];assert region['kind']=='rectangle'
            vertices=np.asarray(region['vertices']);bounds=[float(vertices[:,0].min()),float(vertices[:,1].min()),float(vertices[:,0].max()),float(vertices[:,1].max())]
            truth_all=truth.copy()
            truth_mask=inside(truth,bounds)
            outside_truth=truth[~truth_mask]
            truth=truth[truth_mask]
            original=np.asarray(stored['points_xy'],dtype=float).reshape(-1,2);pred=original[inside(original,bounds)]
            primary=score(pred,truth,2.)
            roi_result=pipe.detect(raw,roi=bounds);roi_pred=roi_result.points[inside(roi_result.points,bounds)]
            roi_metrics=score(roi_pred,truth,2.)
            row={'preview_number':number,'source_image':str(source),'source_identity':stored['source_identity'],
                 'csv_file':str(csv_file.resolve()),'csv_sha256':digest(csv_file), 'metadata_sha256':digest(metadata_file),
                 'prediction_file':str(predfile),'prediction_sha256':digest(predfile),'model_sha256':stored['model_sha256'],
                 'coordinate_convention':'zero-based x right, y down; frame 1; no shifts or refinement applied to answers',
                 'review_confirmation':'user explicitly confirmed fully human checked' if number==5 else 'user supplied as manual answer; per-frame manual edit recorded',
                 'generator_sources':dict(Counter(r['来源'] for r in records)), 'generator_manually_edited_frames':meta['manually_edited_frames'],
                 'csv_reference_points':len(truth_all),'reference_points':len(truth),
                 'excluded_reference_points_outside_roi':len(outside_truth),
                 'original_prediction_count':len(original),'prediction_count_in_roi':len(pred),
                 'excluded_predictions_outside_roi':len(original)-len(pred),'roi_xyxy_inclusive':bounds,
                 'quality':{'unique_ids':True,'unique_coordinates':True,'finite_and_in_image':True,'all_points_inside_declared_roi':bool(truth_mask.all()),
                            'nearest_neighbor_median_px':float(np.median(cKDTree(truth).query(truth,k=2)[0][:,1])),
                            'pairs_within_2px':len(cKDTree(truth).query_pairs(2.))},
                 'primary_frozen_full_image':primary,'diagnostic_roi_inference':roi_metrics,
                 'diagnostic_all_csv_vs_full_prediction':score(original,truth_all,2.),
                 'diagnostic_larger_radii_only':[score(pred,truth,r) for r in (4.,6.)]}
            name=f'{number:02d}_comparison'
            write_json(out/f'{name}.json',{**row,'truth_xy':truth.tolist(),'all_csv_truth_xy':truth_all.tolist(),
                       'excluded_reference_xy':outside_truth.tolist(),'frozen_prediction_xy':pred.tolist(),'roi_prediction_xy':roi_pred.tolist()})
            panels.append(draw_overlay(raw,truth,pred,bounds,primary,f'测试集 {number:02d} · 上一轮固定预测对照',out/f'{name}.png'))
            rows.append(row)
            print({k:row[k] for k in ('preview_number','reference_points','original_prediction_count','prediction_count_in_roi')},
                  {k:primary[k] for k in ('tp','fp','fn','precision','recall','rmse_px')},flush=True)
            print('ROI diagnostic',{k:roi_metrics[k] for k in ('tp','fp','fn','precision','recall')},flush=True)
        assert rows
        scaled=[]
        for p in panels:
            p=p.resize((1000,round(p.height*1000/p.width)));scaled.append(p)
        overview=Image.new('RGB',(1000,sum(p.height for p in scaled)+20*(len(scaled)-1)),'#101827');y=0
        for p in scaled:overview.paste(p,(0,y));y+=p.height+20
        overview.save(out/'overview.png')
        for filename,sha in hashes.items():assert digest(Path(filename))==sha
        totals={k:sum(r['primary_frozen_full_image'][k] for r in rows) for k in ('tp','fp','fn')}
        totals.update(precision=totals['tp']/(totals['tp']+totals['fp']),recall=totals['tp']/(totals['tp']+totals['fn']))
        write_json(out/'summary.json',{'purpose':'user_reviewed_reference_comparison','model_changed':False,'test_used_for_training':False,
                  'test_used_for_threshold_tuning':False,'primary_predictions_preexist_answers':True,'primary_radius_px':2.,
                  'scope':'only exported inclusive rectangle; areas outside are not evaluated',
                  'test_is_blind':False,'assumption':'exported detection rectangle represents fully checked coverage',
                  'input_files_unchanged':True,'input_sha256':hashes,'rows':rows,'totals':totals,
                  'missing_coordinate_answers':['03: marked image/stack only, no coordinate CSV','06: Untitled1.tif answer not found'],
                  'limitations':['Two files may share a field and are not independent acquisitions.','ROI-only rerun is a diagnostic, not the original preview performance.','User confirmed full manual review for 05 despite automatic generator provenance.']})
        # Inspectable notebook executes this exact reusable script with a fresh output directory.
        write_json(out/'reproduce.ipynb',{'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
                   'cells':[{'cell_type':'markdown','metadata':{},'source':['# 人工参考答案核对\n','主指标读取答案提供前保存的固定预测；选区推理只作诊断。05 已获用户确认完整人工检查。\n']},
                            {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[
                                'import subprocess\n',f'root = {str(root)!r}\n',
                                f'subprocess.run([{str(root/".runtime/python310/python.exe")!r}, "-X", "utf8", "scripts/evaluate_reviewed_test_answers.py", "--answers", {str(args.answers.resolve())!r}, "--output", {str(args.output.resolve())+"_reproduced"!r}], cwd=root, check=True)\n']} ]})


if __name__=='__main__':
    main()

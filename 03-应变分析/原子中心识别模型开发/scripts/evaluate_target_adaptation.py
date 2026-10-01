"""Select checkpoints for fit on authorized adaptation images, not blind testing."""
from pathlib import Path
import argparse
import gc
import sys
import shutil
import numpy as np
from atom_center.backends import TorchBackend, onnx_pipeline
from atom_center.pipeline import DetectionPipeline
from atom_center.configuration import pipeline_config
from atom_center.training import read_run
from atom_center.storage import read_json, write_json
from atom_center.model_manifest import sha256_file
from atom_center.image_io import load_image
from atom_center.source_identity import verify_source
from atom_center.evaluation import evaluate_dataset
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluate_reviewed_test_answers import score, inside, draw_overlay

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/target-adaptation-20260910'


def target_metrics(pipe, output=None):
    rows=[]
    for number in (4,5):
        truth=read_json(ROOT/f'runs/reviewed-test-20260910/{number:02d}_comparison.json')
        verify_source(Path(truth['source_image']),truth['source_identity'])
        assert sha256_file(Path(truth['csv_file']))==truth['csv_sha256']
        raw=load_image(truth['source_image'],normalize=False,preserve_dtype=True).image
        g=np.asarray(truth['all_csv_truth_xy']);roi=truth['roi_xyxy_inclusive']
        result=pipe.detect(raw);p=result.points
        full=score(p,g,2.);limited=score(p[inside(p,roi)],g[inside(g,roi)],2.)
        row={'number':number,'source_image':truth['source_image'],'reference_count':len(g),'prediction_count':len(p),
             'model_sha256':result.model_sha256,'full_image':full,'same_original_roi':limited,
             'role':'supervised_adaptation_fit','independent_test':False}
        if output:
            output.mkdir(exist_ok=True,parents=True)
            write_json(output/f'{number:02d}_prediction.json',{**row,'points_xy':p.tolist(),'truth_xy':g.tolist(),'confidences':result.confidences.tolist(),
                       'source_identity':truth['source_identity'],'metadata':dict(result.metadata)})
            draw_overlay(raw,g,p,[0,0,raw.shape[1]-1,raw.shape[0]-1],full,
                         f'开发图像 {number:02d} · 人工答案参与训练后的拟合结果',output/f'{number:02d}_comparison.png')
        rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=BASE/'finetune100')
    parser.add_argument('--manifest',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.manifest:
        p=onnx_pipeline(args.manifest);out=args.output or args.manifest.parent/'target_preview'
        rows=target_metrics(p,out);write_json(out/'summary.json',{'rows':rows,'independent_test':False})
        print([{k:r[k] for k in ('number','reference_count','prediction_count')}|{k:r['full_image'][k] for k in ('precision','recall','f1')} for r in rows],flush=True)
        return
    run,m,state=read_run(args.run);assert state['status']=='completed'
    out=run/'target_selection';out.mkdir(exist_ok=False)
    paths={run/v['path'] for v in state['checkpoints'].values()}
    paths.update(p for p in (run/'training/weights').glob('epoch*.pt') if int(p.stem[5:])>=20)
    records=[]
    for path in sorted(paths):
        h=sha256_file(path)
        if any(r['sha256']==h for r in records):continue
        backend=TorchBackend(path,expected_sha256=h,contract=m['contract'],inference=m['config']['inference'],device='cuda:0')
        p=DetectionPipeline(backend,pipeline_config(m['config']))
        targets=target_metrics(p)
        val=evaluate_dataset(p,ROOT/'data/processed/real_workflow_20260909_v2',split='val',max_distance_px=2.)
        compact=[{k:r[k] for k in ('number','reference_count','prediction_count')}|{k:r['full_image'][k] for k in ('tp','fp','fn','precision','recall','f1','rmse_px')} for r in targets]
        rec={'checkpoint':str(path),'sha256':h,'targets':compact,'old_val':{k:val[k] for k in ('tp','fp','fn','precision','recall','f1','rmse_px')}}
        records.append(rec);write_json(out/f'{path.stem}_{h[:12]}.json',{'record':rec,'targets':targets,'old_val':val})
        write_json(out/'comparison.json',records);print(rec,flush=True)
        del p,backend;gc.collect()
    # Prefer both target images meeting P/R >= 95%, then worst-image F1;
    # old validation remains an explicit regression check, not a hidden test.
    def rank(r):
        return (all(x['precision']>=.95 and x['recall']>=.95 for x in r['targets']),
                min(x['f1'] for x in r['targets']),r['old_val']['f1'])
    best=max(records,key=rank);target=out/'target_fit.pt';shutil.copy2(best['checkpoint'],target)
    assert sha256_file(target)==best['sha256']
    state['checkpoints']['target_fit.pt']={'path':target.relative_to(run).as_posix(),'sha256':best['sha256']}
    state['target_adaptation_selection']={'record':'target_selection/selected.json','not_independent_test':True}
    write_json(run/'state.json',state)
    write_json(out/'selected.json',{'best':best,'ranking':'both target P/R >= .95, then max worst target F1, then old val F1',
                                'independent_test':False,'labels_used_in_training':True})
    print('SELECTED',best,flush=True)


if __name__=='__main__':main()

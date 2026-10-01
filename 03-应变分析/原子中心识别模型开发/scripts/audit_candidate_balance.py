"""Inspect training-region regressions among validation-qualified checkpoints."""
from pathlib import Path
import argparse,gc
import torch
from atom_center.storage import read_json,write_json
from atom_center.training import read_run
from atom_center.backends import TorchBackend
from atom_center.pipeline import DetectionPipeline
from atom_center.configuration import pipeline_config
from atom_center.evaluation import evaluate_dataset

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    run,m,state=read_run(a.run);assert state['status']=='completed'
    history=read_json(run/'point_validation/selection.json')['history']
    # Only persisted improvements have immutable candidate files.
    candidates=[]
    for r in history:
        if r['metrics']['precision']>=.95 and r['metrics']['recall']>=.95:
            found=list((run/'point_validation').glob(f'candidate_{r["epoch"]:04d}_*.pt'))
            if found:candidates.append((r,found[0]))
    candidates=sorted(candidates,key=lambda x:x[0]['metrics']['f1'],reverse=True)[:6]
    rows=[];root=Path(__file__).resolve().parents[1]
    for r,path in candidates:
        from atom_center.model_manifest import sha256_file
        h=sha256_file(path)
        b=TorchBackend(path,expected_sha256=h,contract=m['contract'],inference=m['config']['inference'],device='cuda:0')
        result=evaluate_dataset(DetectionPipeline(b,pipeline_config(m['config'])),root/'data/processed/real_workflow_20260909_v2',split='train',max_distance_px=2.)
        dest=run/f'balance_train_epoch{r["epoch"]}.json';write_json(dest,result)
        row={'epoch':r['epoch'],'checkpoint':str(path),'sha256':h,'val':r['metrics'],'train':{k:result[k] for k in ('tp','fp','fn','precision','recall','f1','rmse_px')},
             'min_train_region_recall':min(x['tp']/(x['tp']+x['fn']) for x in result['regions'])}
        rows.append(row);print(row,flush=True);del b;gc.collect();torch.cuda.empty_cache()
    write_json(run/'candidate_balance.json',{'test_used':False,'validation_gate':{'precision':.95,'recall':.95},'candidates':rows})

if __name__=='__main__':main()

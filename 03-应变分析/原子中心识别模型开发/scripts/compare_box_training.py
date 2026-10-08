"""Controlled full training-set comparison, evaluated on existing validation only."""
from pathlib import Path
from dataclasses import replace
import argparse
import gc
import torch
from atom_center.storage import read_json,write_json
from atom_center.data_workflow import verify_dataset
from atom_center.configuration import load_config,pipeline_config
from atom_center.training import train_run,checkpoint_for_run,read_run
from atom_center.backends import TorchBackend
from atom_center.pipeline import DetectionPipeline
from atom_center.evaluation import evaluate_dataset

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=150)
    parser.add_argument('--continue-comparison',action='store_true',help='Reuse completed matching runs after a comparison-script failure')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=args.continue_comparison)
    root=Path(__file__).resolve().parents[1]
    paths={8:root/'data/processed/real_workflow_20260909_v2',16:root/'data/processed/real_workflow_20260909_box16'}
    m8,_=verify_dataset(paths[8]);m16,_=verify_dataset(paths[16])
    assert [(s['record_key'],s['split'],s['annotation_digest']) for s in m8['sources']]==[(s['record_key'],s['split'],s['annotation_digest']) for s in m16['sources']]
    assert all(m8['artifacts'][p]==m16['artifacts'][p] for p in m8['artifacts'] if p.startswith(('images/','truth/','raw/')))
    config=load_config(root/'configs/common.yaml',root/'configs/haadf_stem.yaml',{
        'training.epochs':args.epochs,'training.patience':args.epochs+1,
        'training.batch':4,'training.nbs':4,'training.workers':2,'training.warmup_epochs':0.,
        'training.save_period':-1,'training.degrees':0.,'training.translate':0.,'training.scale':0.,
        'training.fliplr':0.,'training.flipud':0.,'augmentation.probability':0.,
        'inference.max_det':4000,'seed':20260909})
    plan={'purpose':'workflow_validation_box_ablation',
        'test_images_used':False,'epochs':args.epochs,'config':config,
        'identical_source_splits_truth_and_images':True,'varied_training_factor':'label box width and height: 8 vs 16 original pixels',
        'validation_grid':{'iou':[.45,.70],'refinement':['none','com']},
        'match_distance_px':2.,'confidence':.25,'bn_recalibration':False,
        'limitations':['ROI completeness and acquisition independence still require review.',
                       'No augmentation in either arm to isolate target-box effects.']}
    if (out/'comparison_plan.json').exists():
        assert read_json(out/'comparison_plan.json')==plan,'comparison plan changed'
    else:write_json(out/'comparison_plan.json',plan)
    comparison=[]
    for box,data in paths.items():
        run=out/f'box{box}'
        print(f'START_TRAIN_BOX_{box}',flush=True)
        if run.exists():
            assert args.continue_comparison
            _,prior,state=read_run(run)
            expected,_=verify_dataset(data)
            assert state['status']=='completed' and prior['config']==config and prior['dataset_content_sha256']==expected['content_sha256']
        else:train_run(data,run,config)
        gc.collect();torch.cuda.empty_cache()
        manifest,checkpoint,digest=checkpoint_for_run(run)
        backend=TorchBackend(checkpoint,expected_sha256=digest,contract=manifest['contract'],inference=manifest['config']['inference'],device='cuda:0')
        for iou in (.45,.70):
            backend.inference={**backend.inference,'iou':iou}
            for refine in (False,True):
                pipe=DetectionPipeline(backend,replace(pipeline_config(config),refine=refine))
                result=evaluate_dataset(pipe,data,split='val',max_distance_px=2.)
                name=f'val_iou{iou:.2f}_{"com" if refine else "none"}.json'
                write_json(run/name,result)
                row={'box_size_px':box,'checkpoint_sha256':digest,'iou':iou,'refinement':'com' if refine else 'none',
                     'evaluation_file':str(run/name),**{k:result[k] for k in ('tp','fp','fn','precision','recall','f1','rmse_px','p95_px')}}
                comparison.append(row);write_json(out/'validation_comparison.json',comparison)
                print('VALIDATION',row,flush=True)
        del backend,pipe;gc.collect();torch.cuda.empty_cache()
    assert read_json(out/'box8/initialization.json')['model_state_sha256']==read_json(out/'box16/initialization.json')['model_state_sha256']
    write_json(out/'completed.json',{'status':'completed','same_initial_weights':True,'test_images_used':False,
               'validation_comparison':comparison,'scientific_acceptance':False})


if __name__=='__main__':main()

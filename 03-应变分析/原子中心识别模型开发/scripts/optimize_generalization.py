"""Bounded augmentation experiments using frozen training/validation sources only."""
from pathlib import Path
import gc
import torch
from atom_center.configuration import load_config, pipeline_config
from atom_center.training import train_run, checkpoint_for_run, read_run
from atom_center.backends import TorchBackend
from atom_center.pipeline import DetectionPipeline
from atom_center.evaluation import evaluate_dataset
from atom_center.storage import write_json

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/generalization-20260910'
DATA = ROOT/'data/processed/real_workflow_20260909_v2'

def evaluate(run, name):
    manifest, checkpoint, digest = checkpoint_for_run(run, checkpoint='best_points.pt')
    backend = TorchBackend(checkpoint, expected_sha256=digest, contract=manifest['contract'],
                           inference=manifest['config']['inference'], device='cuda:0')
    pipe = DetectionPipeline(backend, pipeline_config(manifest['config']))
    results = {}
    for split in ('train','val'):
        result = evaluate_dataset(pipe, DATA, split=split, max_distance_px=2.)
        write_json(OUT/f'{name}_{split}.json', result)
        results[split] = {k: result[k] for k in ('tp','fp','fn','precision','recall','f1','rmse_px')}
        results[split]['min_region_recall'] = min(r['tp']/(r['tp']+r['fn']) for r in result['regions'])
        print(name, split, results[split], flush=True)
    del backend, pipe
    gc.collect(); torch.cuda.empty_cache()
    return {'name':name, 'run':str(run), 'checkpoint':str(checkpoint), 'sha256':digest, **results}

def main():
    OUT.mkdir(exist_ok=True)
    baseline = ROOT/'runs/point-selection-20260909'
    _, checkpoint, digest = checkpoint_for_run(baseline, checkpoint='best_points.pt')
    experiments = [
        ('augmented250', {'training.epochs':250, 'training.patience':251,
         'training.fliplr':.5,'training.flipud':.5,'training.degrees':5.,
         'training.translate':.05,'training.scale':.25,'augmentation.probability':.5}, None),
        ('finetune150', {'training.epochs':150, 'training.patience':151,
         'training.model':str(checkpoint), 'training.lr0':.00015,
         'training.fliplr':.5,'training.flipud':.5,'training.degrees':0.,
         'training.translate':.025,'training.scale':.15,'augmentation.probability':.5}, digest),
    ]
    plan = {'data':str(DATA), 'test_used_for_training_or_selection':False,
            'match_distance_px':2., 'target':{'precision':.95,'recall':.95,'min_region_recall':.90},
            'experiments':[{'name':n,'overrides':c,'initialization_sha256':s} for n,c,s in experiments]}
    write_json(OUT/'plan.json',plan)
    rows = [evaluate(baseline,'baseline')]
    write_json(OUT/'comparison.json',rows)
    for name, overrides, initial in experiments:
        config = load_config(ROOT/'configs/experiments/haadf_point_selected_150.yaml', overrides=overrides)
        run = OUT/name
        if run.exists():
            _, existing, state = read_run(run)
            assert state['status']=='completed' and existing['config']==config
        else:
            train_run(DATA,run,config,initialization_sha256=initial)
        gc.collect();torch.cuda.empty_cache()
        rows.append(evaluate(run,name)); write_json(OUT/'comparison.json',rows)
    write_json(OUT/'completed.json',{'completed':True,'comparison':rows,'test_used':False})

if __name__=='__main__': main()

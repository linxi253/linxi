"""Compare physical tile sizes on train/val only, with a fixed network input."""
from pathlib import Path
from dataclasses import replace
from atom_center.training import checkpoint_for_run
from atom_center.backends import TorchBackend
from atom_center.pipeline import DetectionPipeline
from atom_center.configuration import pipeline_config
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


def main():
    root=Path(__file__).resolve().parents[1];out=root/'runs/generalization-20260910/tile_scale';out.mkdir(exist_ok=False)
    m,p,h=checkpoint_for_run(root/'runs/point-selection-20260909',checkpoint='best_points.pt')
    b=TorchBackend(p,expected_sha256=h,contract=m['contract'],inference=m['config']['inference'],device='cuda:0');rows=[]
    for tile in (960,1280,1920):
        pipe=DetectionPipeline(b,replace(pipeline_config(m['config']),tile_size=tile,refinement_window=11,refinement_method='com_continuous'))
        row={'tile_size':tile,'input_size':640,'model_sha256':h}
        for split in ('train','val'):
            r=evaluate_dataset(pipe,root/'data/processed/real_workflow_20260909_v2',split=split,max_distance_px=2.)
            write_json(out/f'tile{tile}_{split}.json',r)
            row[split]={k:r[k] for k in ('tp','fp','fn','precision','recall','f1','rmse_px')}
            row[split]['min_region_recall']=min(s['tp']/(s['tp']+s['fn']) for s in r['regions'])
        rows.append(row);print(row,flush=True)
    write_json(out/'summary.json',{'test_used':False,'rows':rows})

if __name__=='__main__':main()

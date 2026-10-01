"""Validation-only tuning of refinement strength, with training regression checks."""
from pathlib import Path
from dataclasses import replace
import hashlib
import numpy as np
from atom_center.backends import onnx_pipeline
from atom_center.pipeline import DetectionPipeline
from atom_center.refinement import refine_with_diagnostics
from atom_center.geometry import merge_close_points
from atom_center.interfaces import DetectionResult
from atom_center.evaluation import evaluate_dataset
from atom_center.storage import write_json

class BlendPipeline:
    def __init__(self,base):
        self.base=DetectionPipeline(base.backend,replace(base.config,refine=False));self.cache={};self.mode=.5
    def detect(self,raw,*,roi=None):
        key=(raw.shape,hashlib.sha256(raw.tobytes()).hexdigest(),tuple(roi) if roi else None)
        if key not in self.cache:
            original=self.base.detect(raw,roi=roi)
            refined=refine_with_diagnostics(raw,original.points,method='adaptive_blob',window_size=33,polarity='bright',max_shift_px=12.)
            self.cache[key]=(original,refined.points)
        original,refined=self.cache[key]
        alpha=np.clip((.95-original.confidences)/.45,0.,1.)[:,None] if self.mode=='confidence' else self.mode
        points=original.points+alpha*(refined-original.points)
        x0,y0,x1,y1=original.metadata['roi_xyxy'];inside=(points[:,0]>=x0)&(points[:,0]<x1)&(points[:,1]>=y0)&(points[:,1]<y1)
        merged=merge_close_points(points[inside],original.confidences[inside],min_distance=3.)
        return DetectionResult(merged.points,merged.confidences,provider=original.provider,model_sha256=original.model_sha256,metadata={'roi_xyxy':original.metadata['roi_xyxy'],'blend':self.mode})

def main():
    root=Path(__file__).resolve().parents[1];out=root/'runs/generalization-20260910/refinement_blend';out.mkdir(exist_ok=False)
    pipe=BlendPipeline(onnx_pipeline(root/'runs/generalization-20260910/adaptive_blob-onnx/model_manifest.json'));rows=[]
    for mode in (.25,.5,.75,'confidence'):
        pipe.mode=mode;row={'mode':mode}
        for split in ('train','val'):
            r=evaluate_dataset(pipe,root/'data/processed/real_workflow_20260909_v2',split=split,max_distance_px=2.);write_json(out/f'{mode}_{split}.json',r)
            row[split]={k:r[k] for k in ('tp','fp','fn','precision','recall','f1','rmse_px')};row[split]['min_region_recall']=min(s['tp']/(s['tp']+s['fn']) for s in r['regions'])
        rows.append(row);print(row,flush=True)
    write_json(out/'comparison.json',{'test_used':False,'rows':rows})

if __name__=='__main__':main()

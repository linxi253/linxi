"""Train/val experiment: estimate blob scale from raw intensities, then refine."""
from pathlib import Path
from dataclasses import replace
import numpy as np
from scipy.ndimage import gaussian_laplace
from atom_center.training import checkpoint_for_run
from atom_center.backends import TorchBackend
from atom_center.pipeline import DetectionPipeline
from atom_center.configuration import pipeline_config
from atom_center.refinement import refine_with_diagnostics
from atom_center.geometry import merge_close_points
from atom_center.interfaces import DetectionResult
from atom_center.evaluation import evaluate_dataset
from atom_center.storage import write_json

class AdaptiveRefinement:
    def __init__(self, pipeline):
        self.pipeline=pipeline;self.scales=(1.,1.5,2.,3.,4.,6.,8.);self.cache={}
    def detect(self,raw,*,roi=None):
        import hashlib
        result=self.pipeline.detect(raw,roi=roi);points=result.points.copy()
        if not len(points):return result
        key=(raw.shape,hashlib.sha256(raw.tobytes()).hexdigest())
        if key not in self.cache:
            self.cache={key:np.stack([-gaussian_laplace(raw,s)*s*s for s in self.scales])}
        responses=self.cache[key];coords=np.rint(points).astype(int);x=np.clip(coords[:,0],0,raw.shape[1]-1);y=np.clip(coords[:,1],0,raw.shape[0]-1)
        selected=np.argmax(responses[:,y,x],axis=0);quality=[]
        for index,scale in enumerate(self.scales):
            mask=selected==index
            if not mask.any():continue
            wide=scale>=3.;window=int(4*scale)+1 if wide else 7
            window=min(33,window);limit=min(12.,.8*(window//2)) if wide else 2.4
            refined=refine_with_diagnostics(raw,points[mask],method='gaussian' if wide else 'com_continuous',
                window_size=window,polarity='bright',max_shift_px=limit)
            points[mask]=refined.points
            quality.append({'sigma_px':scale,'points':int(mask.sum()),'window':window,'max_shift_px':limit,
                            'refined':sum(d['status']=='refined' for d in refined.diagnostics)})
        bounds=result.metadata['roi_xyxy'];x0,y0,x1,y1=bounds
        inside=(points[:,0]>=x0)&(points[:,0]<x1)&(points[:,1]>=y0)&(points[:,1]<y1)
        merged=merge_close_points(points[inside],result.confidences[inside],min_distance=3.)
        return DetectionResult(merged.points,merged.confidences,provider=result.provider,model_sha256=result.model_sha256,
             metadata={**result.metadata,'experimental_adaptive_refinement':quality,'count_before_refinement':len(points),
                       'count_after_refinement_merge':len(merged.points),'test_tuned':False})

def main():
    root=Path(__file__).resolve().parents[1];run=root/'runs/generalization-20260910/spacing_noaug250'
    m,p,h=checkpoint_for_run(run,checkpoint='best_points.pt');b=TorchBackend(p,expected_sha256=h,contract=m['contract'],inference=m['config']['inference'],device='cuda:0')
    pipe=AdaptiveRefinement(DetectionPipeline(b,replace(pipeline_config(m['config']),refine=False)))
    rows={}
    for split in ('train','val'):
        r=evaluate_dataset(pipe,root/'data/processed/real_workflow_20260909_v2',split=split,max_distance_px=2.)
        write_json(run/f'adaptive_{split}.json',r);rows[split]={k:r[k] for k in ('tp','fp','fn','precision','recall','f1','rmse_px')};rows[split]['min_region_recall']=min(s['tp']/(s['tp']+s['fn']) for s in r['regions']);print(split,rows[split],flush=True)
    write_json(run/'adaptive_comparison.json',{'test_used':False,'model_sha256':h,'results':rows})

if __name__=='__main__':main()

"""Experimental flip consensus on validation only; not a deployment backend."""
from pathlib import Path
from dataclasses import replace
import argparse,hashlib
import numpy as np
from scipy.spatial import cKDTree
from atom_center.interfaces import CandidateSet
from atom_center.training import checkpoint_for_run
from atom_center.backends import TorchBackend
from atom_center.pipeline import DetectionPipeline
from atom_center.configuration import pipeline_config
from atom_center.evaluation import evaluate_dataset
from atom_center.storage import write_json

class FlipConsensus:
    def __init__(self,backend,radius=3.,votes=2):
        self.backend=backend;self.name='experimental_flip_consensus';self.model_sha256=backend.model_sha256
        self.radius=radius;self.votes=votes;self.cache={}
    def predict(self,image):
        key=(image.shape,hashlib.sha256(image.tobytes()).hexdigest())
        if key not in self.cache:
            points=[];scores=[];views=[]
            for view,(flip_y,flip_x) in enumerate(((False,False),(False,True),(True,False),(True,True))):
                tile=image[::-1 if flip_y else 1,::-1 if flip_x else 1]
                result=self.backend.predict(tile);p=result.points.copy()
                if flip_x:p[:,0]=image.shape[1]-1-p[:,0]
                if flip_y:p[:,1]=image.shape[0]-1-p[:,1]
                valid=(p[:,0]>=0)&(p[:,0]<image.shape[1])&(p[:,1]>=0)&(p[:,1]<image.shape[0])
                points.extend(p[valid]);scores.extend(result.confidences[valid]);views.extend([view]*int(valid.sum()))
            self.cache[key]=(np.array(points).reshape(-1,2),np.array(scores),np.array(views))
        points,scores,views=self.cache[key]
        if not len(points):return CandidateSet.empty()
        tree=cKDTree(points);used=np.zeros(len(points),bool);out=[];confidence=[]
        for index in np.argsort(-scores,kind='stable'):
            if used[index]:continue
            near=[i for i in tree.query_ball_point(points[index],self.radius) if not used[i]]
            selected=[]
            for view in range(4):
                candidates=[i for i in near if views[i]==view]
                if candidates:selected.append(min(candidates,key=lambda i:(float(np.sum((points[i]-points[index])**2)),-scores[i],i)))
            if len(selected)<self.votes:continue
            used[selected]=True
            out.append(np.average(points[selected],axis=0,weights=scores[selected]));confidence.append(float(np.mean(scores[selected])))
        return CandidateSet(np.array(out).reshape(-1,2),confidence)

def main():
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    root=Path(__file__).resolve().parents[1];args.output.mkdir(exist_ok=False,parents=True)
    m,p,h=checkpoint_for_run(args.run,checkpoint='best_points.pt')
    b=FlipConsensus(TorchBackend(p,expected_sha256=h,contract=m['contract'],inference=m['config']['inference'],device='cuda:0'))
    rows=[]
    for votes,refine in ((1,True),(2,True),(2,False),(3,True)):
        b.votes=votes;pipe=DetectionPipeline(b,replace(pipeline_config(m['config']),refine=refine))
        result=evaluate_dataset(pipe,root/'data/processed/real_workflow_20260909_v2',split='val',max_distance_px=2.)
        write_json(args.output/f'votes{votes}_refine{refine}.json',result)
        row={'votes':votes,'refine':refine,'sha256':h,**{k:result[k] for k in ('tp','fp','fn','precision','recall','f1','rmse_px')}}
        row['min_region_recall']=min(r['tp']/(r['tp']+r['fn']) for r in result['regions']);rows.append(row);print(row,flush=True)
    write_json(args.output/'summary.json',{'test_used':False,'radius_px':3.,'results':rows})

if __name__=='__main__':main()

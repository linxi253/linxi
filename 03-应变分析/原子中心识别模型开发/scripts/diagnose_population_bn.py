"""Re-estimate BN from frozen training images; evaluate only in eval mode."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import torch
from atom_center.training import checkpoint_for_run
from atom_center.backends import TorchBackend
from atom_center.pipeline import DetectionPipeline
from atom_center.configuration import pipeline_config
from atom_center.data_workflow import verify_dataset
from atom_center.preprocessing import input_tensor
from atom_center.evaluation import evaluate_dataset
from atom_center.storage import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    m,path,h=checkpoint_for_run(a.run,checkpoint='best_points.pt');b=TorchBackend(path,expected_sha256=h,contract=m['contract'],inference=m['config']['inference'],device='cuda:0')
    data=Path(__file__).resolve().parents[1]/'data/processed/real_workflow_20260909_v2';ds,root=verify_dataset(data)
    tensors=[input_tensor(np.array(Image.open(root/c['image'])))[0] for c in ds['crops'] if c['split']=='train']
    initial={k:v.detach().clone() for k,v in b.model.named_parameters()};rng=np.random.default_rng(20260910);rows=[]
    layers=[x for x in b.model.modules() if isinstance(x,torch.nn.BatchNorm2d)]
    for layer in layers:layer.reset_running_stats();layer.momentum=.1
    for epoch in range(1,21):
        b.model.eval()
        for layer in layers:layer.train()
        order=rng.permutation(len(tensors))
        with torch.inference_mode():
            for start in range(0,len(order),4):
                x=torch.from_numpy(np.stack([tensors[i] for i in order[start:start+4]])).to('cuda:0');b.model(x)
        if epoch not in (1,5,20):continue
        b.model.eval();pipe=DetectionPipeline(b,pipeline_config(m['config']));row={'calibration_passes':epoch,'source_sha256':h}
        for split in ('train','val'):
            result=evaluate_dataset(pipe,data,split=split,max_distance_px=2.)
            write_json(a.output/f'passes{epoch}_{split}.json',result)
            row[split]={k:result[k] for k in ('tp','fp','fn','precision','recall','f1','rmse_px')}
            row[split]['min_region_recall']=min(r['tp']/(r['tp']+r['fn']) for r in result['regions'])
        rows.append(row);print(row,flush=True)
    assert all(torch.equal(initial[k],v) for k,v in b.model.named_parameters())
    write_json(a.output/'summary.json',{'diagnostic_only':True,'parameter_weights_unchanged':True,'test_used':False,'calibration_split':'train','evaluation_mode':'eval','checkpoint_saved':False,'results':rows})

if __name__=='__main__':main()

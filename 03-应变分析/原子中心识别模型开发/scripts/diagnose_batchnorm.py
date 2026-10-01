"""Compare fixed learned weights under different BN statistics on training inputs."""
from pathlib import Path
import argparse
from copy import deepcopy
import torch
import numpy as np
from PIL import Image,ImageDraw
from ultralytics import YOLO
from atom_center.preprocessing import input_tensor,prepare_tile
from atom_center.backends import decode_output
from atom_center.metrics import match_points
from atom_center.storage import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args(); run=args.run
    files=sorted((run/'memorization_data/images').glob('*.png'))
    images=[np.array(Image.open(p)) for p in files]
    x=torch.from_numpy(np.stack([input_tensor(i)[0] for i in images])).to('cuda')
    truth=[np.loadtxt(run/'memorization_data/labels'/p.with_suffix('.txt').name,ndmin=2)[:,1:3]*640-.5 for p in files]
    _,transform=prepare_tile(images[0],640)
    model=YOLO(str(run/'training/weights/last.pt')).model.float().to('cuda').eval()
    initial_params={k:v.detach().clone() for k,v in model.named_parameters()}
    rows=[]
    def measure(label,iou=.45):
        with torch.inference_mode(): output=model(x)[0].cpu().numpy()
        cases=[]
        for i,p in enumerate(files):
            pts,_=decode_output(output[i:i+1],transform,conf=.25,iou=iou,max_det=4000)
            m=match_points(pts.points,truth[i],max_distance_px=2.)
            cases.append({'image':p.name,'points':len(pts.points),'tp':m.true_positives,'fp':m.false_positives,'fn':m.false_negatives,
                          'rmse_input_px':float(np.sqrt(np.mean(m.distances_px**2))) if len(m.distances_px) else None,
                          'max_score':float(output[i,4].max())})
            if label=='training_batch_recalibrated_statistics_iou_070':
                pic=Image.fromarray(images[i]).convert('RGB');draw=ImageDraw.Draw(pic)
                for x0,y0 in pts.points:draw.ellipse((x0-3,y0-3,x0+3,y0+3),outline='#00ff88',width=1)
                pic.save(run/f'calibrated_iou070_{p.stem}.png')
        rows.append({'mode':label,'iou':iou,'cases':cases})
    measure('saved_running_statistics')
    for module in model.modules():
        if isinstance(module,torch.nn.BatchNorm2d): module.train()
    measure('current_batch_statistics')
    with torch.inference_mode():
        for _ in range(200):model(x)
    model.eval();measure('training_batch_recalibrated_statistics_201_forwards')
    measure('training_batch_recalibrated_statistics_iou_070',iou=.70)
    assert all(torch.equal(v,initial_params[k]) for k,v in model.named_parameters())
    report={'parameter_weights_identical':True,'checkpoint_modified':False,'scope':'Training-only diagnostic, no optimizer step, not a production inference fix.', 'comparisons':rows}
    write_json(run/'batchnorm_diagnostic.json',report)
    for row in rows:
        print(row['mode'],{k:sum(c[k] for c in row['cases']) for k in ('tp','fp','fn','points')},flush=True)


if __name__=='__main__':main()

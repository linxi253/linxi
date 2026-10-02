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
from atom_center.model_manifest import sha256_file
from atom_center.storage import write_json,read_json

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
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args(); run=args.run
    files=sorted((run/'memorization_data/images').glob('*.png'))
    images=[np.array(Image.open(p)) for p in files]
    x=torch.from_numpy(np.stack([input_tensor(i)[0] for i in images])).to('cuda')
    truth=[np.loadtxt(run/'memorization_data/labels'/p.with_suffix('.txt').name,ndmin=2)[:,1:3]*640-.5 for p in files]
    _,transform=prepare_tile(images[0],640)
    # 审计 79：与 backends.TorchBackend 一致，加载前先按 training.py 记录的 run 状态
    # 校验 last.pt 的 SHA-256，防止 run 目录被替换后无提示地加载被篡改的权重。
    state=read_json(run/'state.json')
    entry=(state.get('checkpoints') or {}).get('last.pt')
    if not entry:
        raise SystemExit('run state.json has no last.pt checkpoint record; refusing to load unverified weights')
    weights=(run/entry['path']).resolve()
    if not weights.is_relative_to(run.resolve()) or sha256_file(weights)!=entry['sha256']:
        raise SystemExit(f'last.pt is missing or its SHA-256 differs from the training record: {weights}')
    model=YOLO(str(weights)).model.float().to('cuda').eval()
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

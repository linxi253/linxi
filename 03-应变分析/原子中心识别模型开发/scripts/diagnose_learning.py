"""Training-only learning diagnosis; never opens reserved test images."""
from pathlib import Path
from copy import copy
import argparse
import csv
import shutil
import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml
from atom_center.storage import read_json, write_json
from atom_center.data_workflow import verify_dataset
from atom_center.training import checkpoint_for_run, _state_dict_digest
from atom_center.training_dataset import VerifiedDataset
from atom_center.preprocessing import prepare_tile, input_tensor
from atom_center.image_io import load_image
from atom_center.metrics import match_points
from atom_center.backends import decode_output
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import init_seeds


class MemorizationTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        # Deliberately no geometric, grayscale or upstream optional augmentation.
        return VerifiedDataset(img_path=img_path, imgsz=self.args.imgsz,
            batch_size=batch, augment=False, hyp=copy(self.args), rect=False,
            cache=None, single_cls=False, stride=32, pad=0., prefix=f"{mode}: ",
            task="detect", classes=None, data=self.data, fraction=1.)

    def read_results_csv(self):
        if not self.csv.exists():
            return {}
        with self.csv.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        return {k: [float(row[k]) for row in rows] for k in rows[0]} if rows else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--box-scale", type=float, default=1.)
    args = parser.parse_args()
    if not np.isfinite(args.box_scale) or args.box_scale <= 0:
        parser.error('--box-scale must be positive and finite')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    frozen, data_root = verify_dataset(root/"data/processed/real_workflow_20260909_v2")
    train, checkpoint, digest = checkpoint_for_run(root/"runs/real-workflow-20260909-v2/yolov8s_640")
    audit = read_json(data_root/"audit_report.json")
    names = {r["record_key"]: Path(r["source_image"]).name for r in audit["records"]}
    chosen = [c for c in frozen["crops"] if c["split"] == "train" and names[c["record_key"]] in {"0004 HAADF.tif", "0007 HAADF.tif", "0005 HAADF.tif", "0059 HAADF.tif"}]
    assert len(chosen) == 4
    dataset = out/"memorization_data"
    for folder in ("images", "labels"):
        (dataset/folder).mkdir(parents=True)
    for c in chosen:
        shutil.copy2(data_root/c["image"], dataset/"images"/Path(c["image"]).name)
        shutil.copy2(data_root/c["label"], dataset/"labels"/Path(c["label"]).name)
        if args.box_scale != 1.:
            label_path=dataset/"labels"/Path(c["label"]).name
            boxes=np.loadtxt(label_path,ndmin=2)
            boxes[:,3:]=np.minimum(boxes[:,3:]*args.box_scale,
                np.minimum(2*boxes[:,1:3],2*(1-boxes[:,1:3])))
            np.savetxt(label_path,boxes,fmt=['%d','%.8f','%.8f','%.8f','%.8f'])
    # Same inputs in train/val is intentional ONLY to measure memorization.
    (dataset/"data.yaml").write_text(yaml.safe_dump({"path":str(dataset), "train":"images", "val":"images", "names":{0:"atom_column"}}), encoding="utf-8")
    write_json(dataset/"provenance.json", {"purpose":"training_only_memorization_diagnostic", "independent_validation":False,
        "source_dataset_sha256":frozen["content_sha256"], "source_crops":chosen, "reserved_test_used":False,
        "label_box_scale":args.box_scale,"centers_unchanged":True})

    init_seeds(train["config"]["seed"], deterministic=True)
    initial = DetectionModel("yolov8s.yaml", nc=1, verbose=False)
    initial_hash = _state_dict_digest(initial)
    expected = read_json(root/"runs/real-workflow-20260909-v2/yolov8s_640/initialization.json")["model_state_sha256"]
    ckpt = torch.load(root/"runs/real-workflow-20260909-v2/yolov8s_640/training/weights/resume.pt", map_location="cpu", weights_only=False)
    model = ckpt["ema"].float()
    changes = {k:float((v.float()-initial.state_dict()[k].float()).abs().max()) for k,v in model.state_dict().items() if v.is_floating_point()}
    report = {"initialization_reproduced":initial_hash==expected, "optimizer_updates":ckpt["updates"],
        "optimizer_step_values":sorted({int(v["step"].item()) for v in ckpt["optimizer"]["state"].values()}),
        "changed_parameter_tensors":sum(changes[k]>0 for k,_ in initial.named_parameters()),
        "first_conv_max_change":changes["model.0.conv.weight"], "parameter_max_change":max(changes.values()),
        "original_checkpoint_sha256":digest, "input_checks":[]}
    hyp = get_cfg(overrides=train["config"]["training"])
    loaded = VerifiedDataset(img_path=str(data_root/"images/train"),imgsz=640,batch_size=4,augment=False,hyp=hyp,
                             rect=False,cache=None,stride=32,pad=0.,task="detect",data={"names":{0:"atom_column"},"nc":1})
    crop_map = {Path(c["image"]).name:c for c in frozen["crops"]}
    for index,file in enumerate(loaded.im_files):
        item=loaded[index]; c=crop_map[Path(file).name]
        source=next(s for s in frozen["sources"] if s["record_key"]==c["record_key"])
        raw=load_image(data_root/source["raw"],normalize=False,preserve_dtype=True).image
        x0,y0,x1,y1=c["roi_xyxy"]
        pixels,transform=prepare_tile(raw[y0:y1,x0:x1],640)
        actual=item["img"].numpy().astype('float32')/255.
        expected_tensor=input_tensor(pixels)[0]
        text=np.loadtxt(data_root/c["label"],ndmin=2)
        boxes=item['bboxes'].numpy()
        point_match=match_points(boxes[:,:2]*640-.5,text[:,1:3]*640-.5,max_distance_px=.001)
        row={"image":names[c['record_key']],"crop":c['export_id'],"loaded_points":len(boxes),
             "expected_points":c['point_count'],"image_max_difference":float(np.max(np.abs(actual-expected_tensor))),
             "label_max_difference_px":float(point_match.distances_px.max()) if len(point_match.distances_px) else None,
             "all_labels_match":point_match.false_positives==point_match.false_negatives==0,
             "min_box_px":float((boxes[:,2:]*640).min()),"input_range":[float(actual.min()),float(actual.max())]}
        report['input_checks'].append(row)
        if c in chosen:
            picture=Image.fromarray(pixels).convert('RGB'); draw=ImageDraw.Draw(picture)
            for bx,by,bw,bh in boxes*640:
                draw.rectangle((bx-.5-bw/2,by-.5-bh/2,bx-.5+bw/2,by-.5+bh/2),outline='#00ff88',width=1)
                draw.ellipse((bx-2,by-2,bx+1,by+1),fill='#ff4040')
            picture.save(out/f"input_{names[c['record_key']].replace('.tif','.png')}")
    assert all(r['image_max_difference']==0 and r['all_labels_match'] and r['loaded_points']==r['expected_points'] for r in report['input_checks'])
    write_json(out/'initial_audit.json',report)
    print('INITIAL_AUDIT', {k:v for k,v in report.items() if k!='input_checks'},flush=True)
    del initial,model,ckpt,loaded

    overrides={**train['config']['training'], 'data':str(dataset/'data.yaml'), 'project':str(out), 'name':'training',
        'exist_ok':False,'epochs':args.epochs,'batch':4,'workers':0,'nbs':4,'warmup_epochs':0.,'patience':args.epochs+1,
        'save_period':-1,'save':True,'plots':False,'pretrained':False,'seed':train['config']['seed'],
        'max_det':4000,'iou':.45,'degrees':0.,'translate':0.,'scale':0.,'fliplr':0.,'flipud':0.,'weight_decay':0.}
    trainer=MemorizationTrainer(overrides=overrides)
    history=[]
    probes=[np.asarray(Image.open(dataset/'images'/Path(c['image']).name)) for c in chosen]
    truths=[np.loadtxt(dataset/'labels'/Path(c['label']).name,ndmin=2)[:,1:3]*640-.5 for c in chosen]
    _,identity=prepare_tile(probes[0],640)
    # Already prepared PNGs: feed directly, without a second percentile normalization.
    probe_tensor=torch.from_numpy(np.stack([input_tensor(p)[0] for p in probes])).to('cuda')
    def on_epoch(tr):
        epoch=tr.epoch+1
        if epoch not in {1,3,10,25,50,100,args.epochs}: return
        tr.ema.ema.eval()
        with torch.inference_mode(): pred=tr.ema.ema(probe_tensor)[0].cpu().numpy()
        cases=[]
        for i,c in enumerate(chosen):
            pts,diag=decode_output(pred[i:i+1],identity,conf=.25,iou=.45,max_det=4000)
            matched=match_points(pts.points,truths[i],max_distance_px=2.)
            cases.append({'crop':c['export_id'],'truth':len(truths[i]),'predicted':len(pts.points),
                          'tp':matched.true_positives,'fp':matched.false_positives,'fn':matched.false_negatives,
                          'max_score':float(pred[i,4].max()),
                          'rmse_px':float(np.sqrt(np.mean(matched.distances_px**2))) if len(matched.distances_px) else None})
            if epoch==args.epochs:
                pic=Image.fromarray(probes[i]).convert('RGB'); draw=ImageDraw.Draw(pic)
                for x,y in pts.points: draw.ellipse((x-3,y-3,x+3,y+3),outline='#00ff88',width=1)
                pic.save(out/f"memorized_{names[c['record_key']].replace('.tif','.png')}")
                write_json(out/f"memorized_{c['export_id']}.json",{'points_xy':pts.points.tolist(),'confidences':pts.confidences.tolist()})
        row={'epoch':epoch,'optimizer_updates':tr.ema.updates,'cases':cases}
        history.append(row); write_json(out/'memorization_history.json',history)
        print('MEMORIZATION',row,flush=True)
    trainer.add_callback('on_fit_epoch_end',on_epoch)
    trainer.train()
    write_json(out/'diagnosis_summary.json',{'initial_audit':report,'memorization':history,'test_images_used':False,'label_box_scale':args.box_scale,
        'scope':'4 training crops, no augmentation, same crops evaluated: learning ability only, not generalization'})


if __name__=='__main__': main()

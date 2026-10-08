"""Package continuous COM11 without changing the verified ONNX graph."""
from pathlib import Path
import argparse
from dataclasses import replace
import shutil
from atom_center.model_manifest import verify_model_bundle
from atom_center.storage import new_artifact_directory,write_json
from atom_center.backends import onnx_pipeline,TorchBackend
from atom_center.training import checkpoint_for_run
from atom_center.pipeline import DetectionPipeline
from atom_center.configuration import pipeline_config
from atom_center.data_workflow import verify_dataset
from atom_center.image_io import load_image
from atom_center.storage import read_json
from atom_center.metrics import match_points
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
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['continuous11','adaptive_blob'],default='continuous11');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    run=root/('runs/generalization-20260910/spacing_noaug250' if args.variant=='adaptive_blob' else 'runs/point-selection-20260909')
    source=run/('onnx/model_manifest.json' if args.variant=='adaptive_blob' else 'onnx-points/model_manifest.json')
    target=root/f'runs/generalization-20260910/{args.variant}-onnx'
    original,graph=verify_model_bundle(source)
    m,p,h=checkpoint_for_run(run,checkpoint='best_points.pt')
    ref={**original.refinement,'window_size':11,'method':'com_continuous'}
    inf=dict(original.inference)
    if args.variant=='adaptive_blob':
        ref.update(method='adaptive_blob',window_size=33,max_shift_px=12.)
        inf['merge_after_refinement']=True
    config={**m['config'],'refinement':ref,'inference':inf}
    pt=DetectionPipeline(TorchBackend(p,expected_sha256=h,contract=m['contract'],inference=inf,device='cpu'),pipeline_config(config))
    data=root/'data/processed/real_workflow_20260909_v2';ds,base=verify_dataset(data)
    with new_artifact_directory(target) as out:
        shutil.copy2(graph,out/graph.name)
        manifest=replace(original,model_id=f'point-selected-{args.variant}-20260910',refinement=ref,inference=inf,
            metrics={**original.metrics,'scientific_acceptance':False,'software_equivalence_passed':False,
                     'variant':args.variant,'refinement_contract':ref,
                     'source_manifest':str(source),'variant_selected_on':'train/val only'})
        manifest.write(out/'model_manifest.json');ort=onnx_pipeline(out/'model_manifest.json');rows=[]
        for s in ds['sources']:
            raw=load_image(base/s['raw'],normalize=False).image;truth=read_json(base/s['truth'])
            for i,roi in enumerate(truth['regions_xyxy']):
                a,b=pt.detect(raw,roi=roi),ort.detect(raw,roi=roi);matches=match_points(a.points,b.points,max_distance_px=.01)
                rows.append({'record_key':s['record_key'],'split':s['split'],'roi_index':i,'torch_points':len(a.points),'onnx_points':len(b.points),
                             'max_delta_px':float(matches.distances_px.max()) if len(matches.distances_px) else None,
                             'unmatched_torch':matches.false_positives,'unmatched_onnx':matches.false_negatives,
                             'passed':matches.false_positives==matches.false_negatives==0})
                if matches.false_positives or matches.false_negatives:
                    from scipy.spatial import cKDTree
                    distances,indices=cKDTree(b.points).query(a.points)
                    rows[-1]['differences']=[{'torch_xy':a.points[j].tolist(),'onnx_xy':b.points[indices[j]].tolist(),'distance_px':float(distances[j])} for j in range(len(a.points)) if distances[j]>.01]
        write_json(target.parent/f'{args.variant}_cpu_parity_diagnostic.json',rows)
        assert all(r['passed'] for r in rows)
        write_json(out/'real_point_parity.json',{'passed':True,'test_used':False,'torch_device':'cpu','onnx_device':'cpu','matching_radius_px':.01,'regions':rows})
        for split in ('train','val'):write_json(out/f'{split}_evaluation.json',evaluate_dataset(ort,data,split=split,max_distance_px=2.))
        manifest=replace(manifest,metrics={**manifest.metrics,'software_equivalence_passed':True,'equivalence_scope':f'unchanged graph; {args.variant} real train/val 22 ROI point parity','nonempty_validation_cases':sum(r['torch_points']>0 for r in rows)})
        manifest.write(out/'model_manifest.json')
    print(target/'model_manifest.json',flush=True)

if __name__=='__main__':main()

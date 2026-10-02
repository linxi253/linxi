"""Explicit supervised adaptation to user-reviewed former test images 04/05.

This is training-fit work, not independent test evaluation. Original files and
the previous frozen dataset remain unchanged. Existing validation groups stay fixed.
"""
from pathlib import Path
import argparse
import copy
import shutil
import numpy as np
from PIL import Image
import tifffile
from atom_center.data_workflow import verify_dataset, dataset_fingerprint, _yolo_lines
from atom_center.storage import read_json, write_json, content_digest, new_artifact_directory
from atom_center.model_manifest import sha256_file
from atom_center.image_io import load_image
from atom_center.geometry import generate_tiles
from atom_center.preprocessing import prepare_tile
from atom_center.source_identity import verify_source
from atom_center.configuration import load_config
from atom_center.training import train_run, checkpoint_for_run

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/target-adaptation-20260910'
DATA=ROOT/'data/processed/reviewed_targets_20260910'


def build():
    old,base=verify_dataset(ROOT/'data/processed/real_workflow_20260910_spacing')
    answers=ROOT/'runs/reviewed-test-20260910'
    plan={'authorization':'User explicitly requested optimizing images with supplied human answers, 04/05; other test images deferred.',
          'former_test_role':'supervised_adaptation_training','independent_test_claim':False,
          'related_0199_field_no_longer_independent_test':True,'original_validation_groups_preserved':True,
          'target_repeats':3,'coverage':'Full-frame supervised fit to all user-provided reference points; 05 explicitly confirmed fully checked. Original exported ROI retained for comparable scoring.',
          'old_dataset_sha256':old['content_sha256']}
    with new_artifact_directory(DATA) as temp:
        for rel in old['artifacts']:
            dest=temp/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/rel,dest)
        payload=copy.deepcopy(old)
        for field in ('artifacts','fingerprint','content_sha256'):payload.pop(field,None)
        payload['adaptation']=plan
        payload['purpose']='supervised_target_adaptation'
        payload['formal_audit_passed']=False
        for number in (4,5):
            answer=read_json(answers/f'{number:02d}_comparison.json')
            source=Path(answer['source_image']);verify_source(source,answer['source_identity'])
            assert sha256_file(Path(answer['csv_file']))==answer['csv_sha256']
            raw=load_image(source,normalize=False,preserve_dtype=True).image
            points=np.asarray(answer['all_csv_truth_xy'],dtype=float)
            key=f'reviewed_target_{number:02d}'
            rp=f'raw/{key}.tif';tp=f'truth/{key}.json'
            tifffile.imwrite(temp/rp,raw,photometric='minisblack')
            truth={'points_xy':points.tolist(),'regions_xyxy':[[0,0,raw.shape[1],raw.shape[0]]],
                   'coordinate_order':'xy','metadata':{'data_origin':'user_reviewed_former_test','original_roi_xyxy':answer['roi_xyxy_inclusive'],
                   'reference_csv':answer['csv_file'],'reference_sha256':answer['csv_sha256'],'preview_number':number}}
            write_json(temp/tp,truth)
            box=8.2
            payload['sources'].append({'record_key':key,'split':'train','isolation_group':'former_test_0199_field',
                'raw':rp,'truth':tp,'modality':'haadf_stem','source_identity':answer['source_identity'],
                'annotation_digest':content_digest(truth),'source_image':str(source),'source_project':str(answers),
                'label_file':answer['csv_file'],'sample_id':'former_test_0199','acquisition_id':'shared_field_unknown',
                'supervision_box_size_px':box})
            for tile in generate_tiles(raw.shape,tile_size=640,overlap=.1):
                crop=raw[tile.slices]
                mask=(points[:,0]+.5>=tile.x0)&(points[:,0]+.5<tile.x1)&(points[:,1]+.5>=tile.y0)&(points[:,1]+.5<tile.y1)
                local=points[mask]-[tile.x0,tile.y0]
                pixels,t=prepare_tile(crop,640,low=1.,high=99.)
                for repeat in range(plan['target_repeats']):
                    eid=f'{key}_x{tile.x0}_y{tile.y0}_rep{repeat}'
                    ip=f'images/train/{eid}.png';lp=f'labels/train/{eid}.txt'
                    Image.fromarray(pixels).save(temp/ip)
                    (temp/lp).write_text(_yolo_lines(t.forward_points(local)-.5,640,640,box*min(t.scale_xy)),encoding='ascii')
                    payload['crops'].append({'export_id':eid,'record_key':key,'split':'train','image':ip,'label':lp,
                        'roi_xyxy':[tile.x0,tile.y0,tile.x1,tile.y1],'point_count':len(local),'repeat':repeat})
        payload['max_train_atoms_per_crop']=max(c['point_count'] for c in payload['crops'] if c['split']=='train')
        payload['group_counts']['train']+=1
        # Regenerate derived provenance; old audit remains the exact historical audit.
        write_json(temp/'adaptation_provenance.json',plan)
        import csv
        with (temp/'source_map.csv').open('w',encoding='utf-8',newline='') as stream:
            fields=['export_id','record_key','split','image','label','point_count','roi_xyxy']
            w=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(payload['crops'])
        payload['artifacts']={p.relative_to(temp).as_posix():sha256_file(p) for p in sorted(temp.rglob('*')) if p.is_file()}
        payload['fingerprint']=dataset_fingerprint(payload);payload['content_sha256']=content_digest(payload['fingerprint'])
        write_json(temp/'dataset_manifest.json',payload)
    verify_dataset(DATA)
    # Source-level raw point truth, existing image/label bytes and split assignments are preserved.
    for p in old['artifacts']:
        if p.startswith(('raw/','truth/','images/','labels/')):assert sha256_file(DATA/p)==old['artifacts'][p]
    write_json(OUT/'data_transition.json',plan)
    return DATA


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--build-only',action='store_true');parser.add_argument('--epochs',type=int,default=100)
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    if not DATA.exists():build()
    if args.build_only:return
    _,checkpoint,digest=checkpoint_for_run(ROOT/'runs/generalization-20260910/spacing_noaug250',checkpoint='best_points.pt')
    config=load_config(ROOT/'configs/experiments/haadf_point_selected_150.yaml',overrides={
        'training.model':str(checkpoint),'training.epochs':args.epochs,'training.patience':args.epochs+1,
        'training.lr0':.0002,'training.workers':2,'training.save_period':10,
        'point_validation.interval_epochs':10,'inference.merge_after_refinement':True,
        'refinement.method':'adaptive_blob','refinement.window_size':33,'refinement.max_shift_px':12.})
    train_run(DATA,OUT/'finetune100',config,initialization_sha256=digest)


if __name__=='__main__':main()

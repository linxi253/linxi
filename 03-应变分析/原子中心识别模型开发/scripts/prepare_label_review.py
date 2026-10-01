"""Copy difficult original labels into an isolated DRAFT review project."""
from pathlib import Path
import shutil
from atom_center.annotations import AnnotationProject
from atom_center.data_workflow import verify_dataset
from atom_center.storage import read_json,write_json

def main():
    root=Path(__file__).resolve().parents[1];out=root/'runs/generalization-20260910/review_projects/difficult_labels';out.mkdir(exist_ok=False)
    (out/'images').mkdir();m,data=verify_dataset(root/'data/processed/real_workflow_20260909_v2')
    audit={r['record_key']:r for r in read_json(data/'audit_report.json')['records']}
    wanted={'387260d9336cd84e4d0d8d56','9dd403f5c82bdb08767d6a17','5b1a35475f1b94c3f9d6e78f','1d423837501b9d2156e829f2','b82fdb0e6dc65c4bb38f76e8'}
    sources={s['record_key']:s for s in m['sources'] if s['record_key'] in wanted}
    for key,s in sources.items():shutil.copy2(data/s['raw'],out/'images'/f'{key}.tif')
    project=AnnotationProject.create(out/'annotation_project.json',image_root=out/'images',modality='haadf_stem',name='困难图像 · 原人工标注复核副本')
    provenance=[]
    for record in project.records:
        key=Path(record.image_path).stem;s=sources[key];truth=read_json(data/s['truth']);doc=project.load_document(record)
        doc.points_xy=[tuple(p) for p in truth['points_xy']];doc.coverage_regions_xyxy=[tuple(r) for r in truth['regions_xyxy']]
        doc.metadata.update({str(k):str(v) for k,v in truth['metadata'].items()})
        doc.metadata.update(original_filename=Path(audit[key]['source_image']).name,original_split=s['split'],
            annotation_source='copy_of_original_manual_labels',notes='原人工坐标的复核副本，未替换为模型坐标。请核对中心、漏标、标注区域和像素标定。')
        doc.review_status='draft';project.save_document(doc)
        provenance.append({'record_key':key,'original_image':audit[key]['source_image'],'original_label':audit[key]['label_file'],'original_label_sha256':audit[key]['label_sha256'],'split':s['split'],'original_points_unchanged':True})
    project.write_manifest();write_json(out/'provenance.json',provenance)
    command=f'& "{root/".venv/Scripts/atom-annotate.exe"}" --project "{project.project_path}"\n'
    (out/'open_review.ps1').write_text(command,encoding='utf-8-sig')
    (out/'README.md').write_text('# 困难标注复核副本\n\n这里保留的是原人工点位，所有记录重置为 draft，便于重新审核。原始项目与冻结数据未修改。请按 provenance.json 对应原文件名，结合 final_train_preview、final_val_preview 中的预测检查。验证集 0030 仍须保留在验证侧。\n',encoding='utf-8')
    print(project.project_path)

if __name__=='__main__':main()

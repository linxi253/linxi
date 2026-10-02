"""Create separate DRAFT annotation projects from measured model predictions."""
from pathlib import Path
import argparse
from atom_center.annotations import AnnotationProject
from atom_center.storage import read_json,write_json
from atom_center.source_identity import verify_source

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main():
    p=argparse.ArgumentParser();p.add_argument('--images',type=Path,required=True);p.add_argument('--preview',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows={}
    for f in a.preview.glob('*.json'):
        d=read_json(f)
        if 'source_image' not in d:continue
        source=Path(d['source_image']).resolve();verify_source(source,d['source_identity']);rows[source]=d
    if not rows:raise ValueError('No verified prediction records')
    a.output.mkdir(exist_ok=False,parents=True)
    project=AnnotationProject.create(a.output/'annotation_project.json',image_root=a.images,modality='haadf_stem',name=a.images.name+' · 模型预标注待人工复核')
    assert len(project.records)==len(rows)
    for record in project.records:
        source=project.image_file(record).resolve();d=rows[source]
        doc=project.load_document(record);doc.points_xy=[tuple(x) for x in d['points_xy']]
        doc.review_status='draft';doc.coverage_regions_xyxy=[]
        doc.metadata.update(annotation_source='model_prediction_unreviewed',model_sha256=d['model_sha256'],
            holdout_role='test',notes='机器预标注，不是人工真值。请逐点增删修正并标明完整标注区域，审核后才能评测。')
        project.save_document(doc)
    project.write_manifest()
    write_json(a.output/'prediction_provenance.json',{'source_preview':str(a.preview.resolve()),'draft_only':True,'test_used_for_training':False,'images':len(rows),'model_sha256s':sorted(set(d['model_sha256'] for d in rows.values()))})
    root=Path(__file__).resolve().parents[1]
    command=f'& "{root / ".venv/Scripts/atom-annotate.exe"}" --project "{project.project_path}"\n'
    (a.output/'open_review.ps1').write_text(command,encoding='utf-8-sig')
    (a.output/'README.md').write_text('# 测试集人工复核\n\n所有点均为模型预标注，状态为 draft，没有把整张图标记为完整标注区。原图仍引用原目录，原始标注未修改。\n\n运行 open_review.ps1 打开标注工具，逐点检查漏检、误检和中心偏差，画出已完整审核的 ROI；完成后再保存为 reviewed。测试数据不得自动并入训练集。\n',encoding='utf-8')
    print(project.project_path,flush=True)

if __name__=='__main__':main()

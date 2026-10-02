"""Round 2 step 2: frozen-model baseline preview of the new, unlabelled originals.

The model and its configuration are the frozen first-round ONNX bundle; nothing is tuned here.
Without human answers this script reports only point counts and network scores: it never derives
precision/recall and never treats a prediction as ground truth.
"""
from pathlib import Path
import json
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from atom_center.backends import onnx_pipeline                            # noqa: E402
from atom_center.image_io import load_image, normalize_percentile         # noqa: E402
from atom_center.source_identity import observe_source, verify_source     # noqa: E402
from atom_center.storage import write_json                                # noqa: E402
from render_test_predictions import text_lines                            # noqa: E402

RUN = ROOT / 'runs/training-update-20260910'
MANIFEST = ROOT / 'runs/target-adaptation-20260910/target_adaptation_onnx/model_manifest.json'
FROZEN_ONNX_SHA256 = '15222438748086ba56fb334926588656fb8057a3b056cdb41f3bd198d910c267'
FONT_PATH = 'C:/Windows/Fonts/msyh.ttc'


def resolve_path(audit, name):
    on_disk = ROOT.parent / '原子标注/测试集2' / name
    if on_disk.is_file():
        return on_disk
    # Audit 35: extraction locations (whose labels carry test-set codenames) are
    # read from the recorded audit instead of being hardcoded here.
    for entry in audit.get('extraction', {}).values():
        directory = entry.get('directory') if isinstance(entry, dict) else None
        if directory and (Path(directory) / name).is_file():
            return Path(directory) / name
    return RUN / 'extracted' / name


def main():
    audit = json.loads((RUN / 'input_audit.json').read_text(encoding='utf-8-sig'))
    targets = {name: value for name, value in audit['candidates'].items()
               if value['status'] == 'new_original_no_labels'}
    output = RUN / 'baseline_preview'
    output.mkdir(parents=True, exist_ok=True)

    pipeline = onnx_pipeline(MANIFEST)
    if pipeline.backend.model_sha256 != FROZEN_ONNX_SHA256:
        raise ValueError('the preview bundle is not the frozen first-round ONNX export')
    refinement = dict(pipeline.backend.manifest.refinement)
    conf = pipeline.backend.inference['conf']
    raw_forward, tile_scores = pipeline.backend.raw, []

    def measured_raw(tensor):
        candidate = raw_forward(tensor)
        tile_scores.append(float(np.max(candidate[:, 4])))
        return candidate

    pipeline.backend.raw = measured_raw
    font = ImageFont.truetype(FONT_PATH, 23)
    small = ImageFont.truetype(FONT_PATH, 19)
    rows, panels = [], []
    for index, (name, value) in enumerate(sorted(targets.items()), 1):
        path = resolve_path(audit, name)
        identity = observe_source(path)
        if identity['file_sha256'] != value['file_sha256']:
            raise ValueError(f'{name}: source changed since the audit')
        raw = load_image(path, normalize=False, preserve_dtype=True).image
        tile_scores.clear()
        started = time.perf_counter()
        result = pipeline.detect(raw)
        seconds = time.perf_counter() - started
        verify_source(path, identity)
        pixels = np.rint(normalize_percentile(raw) * 255).astype(np.uint8)
        picture = Image.fromarray(pixels).convert('RGB')
        mark = ImageDraw.Draw(picture)
        for x, y in result.points:
            mark.ellipse((x - 4, y - 4, x + 4, y + 4), outline='#00ff88', width=2)
            mark.line((x - 2, y, x + 2, y), fill='#00ff88', width=1)
            mark.line((x, y - 2, x, y + 2), fill='#00ff88', width=1)
        count, score = len(result.points), max(tile_scores)
        stem = f'{index:02d}_' + name.replace('.tif', '').replace(' ', '_')
        canvas = Image.new('RGB', (max(900, picture.width), picture.height + 145), '#111827')
        canvas.paste(picture, ((max(900, picture.width) - picture.width) // 2, 145))
        draw = ImageDraw.Draw(canvas)
        y = text_lines(draw, name, (18, 10), font, '#f3f4f6', canvas.width - 36)
        draw.text((18, y), f'第一轮冻结模型 | 阈值 {conf:.2f} | 检测到 {count} 个原子中心', font=font,
                  fill='#fbbf24' if count == 0 else '#00ff88')
        draw.text((18, y + 35), f'网络最高分 {score:.6f} | 绿色圆圈为预测中心 | 无人工真值，不计算精确率/召回率',
                  font=small, fill='#cbd5e1')
        canvas.save(output / f'{stem}.png')
        write_json(output / f'{stem}.json', {
            'source_image': str(path), 'source_identity': identity, 'image_index': index,
            'model_manifest': str(MANIFEST.resolve()), 'model_sha256': result.model_sha256,
            'inference': dict(pipeline.backend.inference), 'refinement': refinement,
            'points_xy': result.points.tolist(), 'confidences': result.confidences.tolist(),
            'metadata': dict(result.metadata), 'network_max_score_before_filtering': score,
            'tile_network_max_scores': list(tile_scores), 'elapsed_seconds': seconds,
            'overlay': f'{stem}.png',
            'annotation_kind': 'model_prediction_not_ground_truth',
            'ground_truth_available': False,
            'note': 'no human coordinate answer exists for this image; the points are a model prediction'})
        rows.append({'name': name, 'image_index': index, 'points': count,
                     'network_max_score': score, 'seconds': seconds,
                     'file_sha256': identity['file_sha256'], 'shape': identity['shape'],
                     'dtype': identity['dtype'], 'overlay': f'{stem}.png',
                     'prediction': f'{stem}.json',
                     'confidence_min': round(float(result.confidences.min()), 4) if count else None,
                     'confidence_median': round(float(np.median(result.confidences)), 4) if count else None})
        panel = Image.new('RGB', (880, 650), '#111827')
        preview = picture.copy()
        preview.thumbnail((844, 490), Image.Resampling.LANCZOS)
        panel.paste(preview, ((880 - preview.width) // 2, 140 + (490 - preview.height) // 2))
        painter = ImageDraw.Draw(panel)
        py = text_lines(painter, f'{index:02d}  {name}', (18, 12), font, '#f3f4f6', 844)
        painter.text((18, py + 3), f'检测 {count} 点 | 阈值 {conf:.2f} | 网络最高分 {score:.6f}', font=font,
                     fill='#fbbf24' if count == 0 else '#00ff88')
        panels.append(panel)
        print(f'{name}: points={count}, max_score={score:.6f}, seconds={seconds:.2f}', flush=True)

    cols = 2
    overview = Image.new('RGB', (cols * 880, 160 + ((len(panels) + cols - 1) // cols) * 650), '#090e18')
    draw = ImageDraw.Draw(overview)
    draw.text((20, 12), '新增原图基线预览 · 第一轮冻结模型', font=ImageFont.truetype(FONT_PATH, 32), fill='white')
    draw.text((20, 62), f'固定阈值 {conf:.2f}，未调参 | 绿色圆圈为模型预测中心 | 这些图没有人工坐标答案，'
                        f'不能计算精确率/召回率，不得当作真值', font=font, fill='#cbd5e1')
    draw.text((20, 100), f'模型 ONNX SHA256 {FROZEN_ONNX_SHA256}', font=small, fill='#94a3b8')
    for i, panel in enumerate(panels):
        overview.paste(panel, ((i % cols) * 880, 160 + (i // cols) * 650))
    overview.save(output / 'overview.png')
    write_json(output / 'summary.json', {
        'purpose': 'baseline_preview_of_new_unlabelled_originals',
        'model_manifest': str(MANIFEST.resolve()), 'model_sha256': pipeline.backend.model_sha256,
        'frozen_model_sha256_confirmed': pipeline.backend.model_sha256 == FROZEN_ONNX_SHA256,
        'fixed_confidence': conf, 'refinement': refinement, 'inference': dict(pipeline.backend.inference),
        'threshold_tuned_on_new_images': False, 'ground_truth_available': False,
        'precision_or_recall_reported': False, 'predictions_used_as_labels': False,
        'source_files_unchanged': True, 'image_count': len(rows),
        'display_normalization': 'full image percentile 1/99; inference uses manifest tile preprocessing',
        'note': 'No human answer exists for these images. Counts and network scores describe model output '
                'only; they are not accuracy metrics and must not enter any training set as labels.',
        'images': rows})
    lines = ['# 新增原图基线预览（第一轮冻结模型）', '',
             f'固定置信度 {conf:.2f}，未在新图上调参。绿色圆圈是模型预测中心，不是人工真值；'
             '这些图没有坐标答案，因此不报告精确率/召回率。', '',
             f'模型 ONNX SHA256：`{FROZEN_ONNX_SHA256}`', '', '![总览](overview.png)', '']
    for row in rows:
        lines.append(f"- [{row['name']}]({row['overlay']})：{row['points']} 点；"
                     f"[坐标 JSON]({row['prediction']})")
    (output / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('total images', len(rows), 'total points', sum(row['points'] for row in rows), flush=True)
    print('wrote', output / 'summary.json', flush=True)


main()

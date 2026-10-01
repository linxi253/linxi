"""Round 2 step 1b: screen the new originals for the same field as any known image.

Same field matters because identical or cropped views must not cross a train/validation split.
The screen uses a coarse block-mean descriptor and Pearson correlation; it is a screening aid,
not a proof of independence.
"""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from atom_center.image_io import load_image                             # noqa: E402
from atom_center.storage import write_json                             # noqa: E402

RUN = ROOT / 'runs/training-update-20260910'
ANN = ROOT.parent / '原子标注'
PREVIEW = ROOT / 'runs/generalization-20260910/final_testset2_preview'
DESCRIPTOR = 64


def descriptor(path):
    plane = np.asarray(load_image(path, normalize=False, preserve_dtype=True).image, dtype=np.float64)
    step_y, step_x = max(1, plane.shape[0] // DESCRIPTOR), max(1, plane.shape[1] // DESCRIPTOR)
    trimmed = plane[:plane.shape[0] // step_y * step_y, :plane.shape[1] // step_x * step_x]
    small = trimmed.reshape(trimmed.shape[0] // step_y, step_y,
                            trimmed.shape[1] // step_x, step_x).mean(axis=(1, 3))
    vector = small.reshape(-1)
    vector = vector - vector.mean()
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def correlation(a, b):
    length = min(len(a), len(b))
    return float(np.dot(a[:length], b[:length]))


def main():
    audit = json.loads((RUN / 'input_audit.json').read_text(encoding='utf-8-sig'))
    candidates = {name: value for name, value in audit['candidates'].items()
                  if value['status'] == 'new_original_no_labels'}

    references = {}
    for label, directory, suffixes in (
            ('测试集2_previewed', ANN / '测试集2', {'.tif'}),
            ('测试集_04_05', ANN / '测试集', {'.tif'}),
            ('HAADF-STEM标注任务_20260905', ANN / 'HAADF-STEM标注任务_20260905_131715/images', {'.tif'}),
            ('HAADF-STEM标注任务_20260909', ANN / 'HAADF-STEM标注任务_20260909_203214/images', {'.tif'}),
            ('标注任务A_20260902',
             ANN / 'AtomCenterAnnotator-v0.2.4-win-x64/HAADF-STEM/HAADF-STEM标注任务_20260902_151017/images', {'.tif'})):
        for path in sorted(Path(directory).glob('*')):
            if path.suffix.lower() in suffixes:
                references[f'{label}/{path.name}'] = path
    for dataset in ('real_workflow_20260909_v2', 'reviewed_targets_20260910'):
        for path in sorted((ROOT / 'data/processed' / dataset / 'raw').glob('*.tif')):
            references[f'{dataset}/raw/{path.name}'] = path

    print(f'{len(candidates)} candidates against {len(references)} references', flush=True)
    cache = {}
    rows = {}
    for name, value in sorted(candidates.items()):
        on_disk = ANN / '测试集2' / name
        path = on_disk if on_disk.is_file() else RUN / 'extracted/测试集3_ncm811-2' / name
        if not path.is_file():
            raise FileNotFoundError(path)
        vector = cache.setdefault(str(path), descriptor(path))
        scored = []
        for label, reference in references.items():
            if reference.resolve() == path.resolve():
                continue
            other = cache.setdefault(str(reference), descriptor(reference)) if reference.is_file() else None
            if other is None:
                continue
            scored.append({'reference': label, 'correlation': round(correlation(vector, other), 4)})
        scored.sort(key=lambda row: -row['correlation'])
        rows[name] = {'sha256': value['file_sha256'], 'shape': value['shape'],
                      'top_matches': scored[:3],
                      'suspected_same_field': [row for row in scored[:3] if row['correlation'] >= 0.9]}
        print(f"{value['file_sha256'][:12]} {name:44s} " + ' | '.join(
            f"{row['correlation']:.3f} {row['reference'].split('/')[-1][:34]}" for row in scored[:3]), flush=True)
    write_json(RUN / 'near_duplicate_screening.json',
               {'purpose': 'same-field screening of the new originals before any future training use',
                'method': f'block-mean {DESCRIPTOR}x{DESCRIPTOR} descriptors, Pearson correlation',
                'caveat': 'a screening aid only; high correlation suggests a shared field but is not proof, '
                          'and low correlation does not rule out a shared acquisition series',
                'candidate_count': len(candidates), 'reference_count': len(references), 'rows': rows})


main()

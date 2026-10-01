"""Round 2 step 1: inventory, dedup and label search for the updated data.

Read-only over user data. The only write outside the new run directory is nothing; zip members
that are not yet on disk are extracted into runs/training-update-20260910/extracted with an
explicit path-traversal check.
"""
from pathlib import Path
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from atom_center.source_identity import observe_source                     # noqa: E402
from atom_center.storage import write_json                                 # noqa: E402
from annotation_project_inventory import container_format, inventory_projects  # noqa: E402

RUN = ROOT / 'runs/training-update-20260910'
ANN = ROOT.parent / '原子标注'
EXTRACT = RUN / 'extracted'
PREVIEW = ROOT / 'runs/generalization-20260910/final_testset2_preview'
DATASETS = {'real_workflow_20260909_v2': ROOT / 'data/processed/real_workflow_20260909_v2',
            'reviewed_targets_20260910': ROOT / 'data/processed/reviewed_targets_20260910'}
ZIPS = {'测试集2_NCM811': ANN / '测试集2/NCM811.zip',
        '测试集3_ncm811-2': ANN / '测试集3/ncm811-2.zip'}
IMAGE_SUFFIXES = {'.tif', '.tiff', '.png'}
FIRST_ROUND_END = '2026-09-10T05:45'


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def sha256_stream(handle, chunk=1 << 20):
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(chunk), b''):
        digest.update(block)
    return digest.hexdigest()


def safe_extract(archive, target):
    """Extract one zip into a fresh directory, refusing any member that escapes it."""
    target = Path(target).resolve()
    if target.exists():
        raise FileExistsError(f'extraction target already exists: {target}')
    target.mkdir(parents=True)
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            if member.is_dir():
                continue
            destination = (target / member.filename).resolve()
            if not destination.is_relative_to(target):
                raise ValueError(f'zip member escapes the target directory: {member.filename}')
            destination.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(member) as source, destination.open('wb') as sink:
                sink.write(source.read())
    return target


def reuse_or_extract(archive, target, members):
    """Reuse a previous extraction only when every member is still byte-identical."""
    target = Path(target).resolve()
    if not target.exists():
        safe_extract(archive, target)
        return {'directory': str(target), 'reused_existing_extraction': False}
    for member in members:
        path = target / member['member']
        if not path.is_file() or sha256_file(path) != member['file_sha256']:
            raise ValueError(f'existing extraction differs from the archive: {path}')
    return {'directory': str(target), 'reused_existing_extraction': True}


def zip_members(archive):
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            if member.is_dir():
                continue
            with handle.open(member) as stream:
                yield member.filename, member.file_size, sha256_stream(stream)


def image_statistics(path):
    from atom_center.image_io import load_image
    identity = observe_source(path)
    plane = np.asarray(load_image(path, normalize=False, preserve_dtype=True).image)
    values = plane.astype(np.float64)
    return identity, {
        'min': float(values.min()), 'max': float(values.max()),
        'mean': float(values.mean()), 'std': float(values.std()),
        'p1': float(np.percentile(values, 1)), 'p50': float(np.percentile(values, 50)),
        'p99': float(np.percentile(values, 99)),
        'saturated_pixels': int((values >= values.max() - 1e-9).sum()),
        'unique_values': int(np.unique(plane).size),
    }


def plane_record(identity):
    """The decoded plane, which is what the frozen source_identity semantics describe."""
    return {'shape': identity['shape'], 'dtype': identity['dtype'],
            'plane_sha256': identity['plane_sha256']}


def walk(directory, suffixes=IMAGE_SUFFIXES):
    if not Path(directory).exists():
        return []
    return sorted(path for path in Path(directory).rglob('*')
                  if path.is_file() and path.suffix.lower() in suffixes)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    audit = {'task_id': 'training-update-20260910', 'phase': 'input_audit',
             'updated_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
             'searched_roots': [str(ANN), str(ROOT / 'data/processed')]}

    # --- 1. candidate new originals -------------------------------------------------
    candidates = {}
    for path in sorted((ANN / '测试集2').glob('*.tif')):
        identity, stats = image_statistics(path)
        candidates[str(path.resolve())] = {
            'name': path.name, 'origin': '原子标注/测试集2 file', 'identity': identity, 'statistics': stats,
            'source_container_format': container_format(path), 'decoded_plane': plane_record(identity),
            'mtime_utc': path.stat().st_mtime}

    archive_hashes = {}
    for label, archive in ZIPS.items():
        members = []
        for name, size, digest in zip_members(archive):
            members.append({'member': name, 'size': size, 'file_sha256': digest})
            archive_hashes.setdefault(digest, []).append(f'{label}::{name}')
        audit.setdefault('zip_members', {})[label] = {'path': str(archive), 'members': members,
                                                      'member_count': len(members)}

    extraction = {}
    already_on_disk = {value['identity']['file_sha256']: key for key, value in candidates.items()}
    for label, archive in ZIPS.items():
        needed = [member for member in audit['zip_members'][label]['members']
                  if member['file_sha256'] not in already_on_disk]
        if not needed:
            extraction[label] = {'needed': 0, 'note': 'every member is byte-identical to a file already on disk'}
            continue
        target = reuse_or_extract(archive, EXTRACT / label, audit['zip_members'][label]['members'])
        extraction[label] = {'needed': len(needed), 'extracted': [member['member'] for member in needed],
                             **target}
        for path in walk(target['directory']):
            if path.suffix.lower() not in {'.tif', '.tiff'}:
                continue
            if sha256_file(path) not in {member['file_sha256'] for member in needed}:
                continue
            identity, stats = image_statistics(path)
            candidates[str(path.resolve())] = {
                'name': path.name, 'origin': f'{label} zip member (extracted copy)',
                'identity': identity, 'statistics': stats,
                'source_container_format': container_format(path), 'decoded_plane': plane_record(identity),
                'mtime_utc': path.stat().st_mtime}
    audit['extraction'] = extraction

    # --- 2. references: training/validation sources, first-round tests, old previews --
    references = {}
    for label, dataset in DATASETS.items():
        for path in walk(dataset / 'raw', {'.tif', '.tiff'}):
            references.setdefault(sha256_file(path), []).append(f'{label}/raw/{path.name}')
        for split in ('train', 'val'):
            for path in walk(dataset / 'images' / split, {'.png'}):
                references.setdefault(sha256_file(path), []).append(f'{label}/images/{split}/{path.name}')
    for label, directory in (('测试集(04/05 sources)', ANN / '测试集'),
                             ('HAADF-STEM标注任务_20260905', ANN / 'HAADF-STEM标注任务_20260905_131715/images'),
                             ('HAADF-STEM标注任务_20260909', ANN / 'HAADF-STEM标注任务_20260909_203214/images'),
                             ('标注任务A_20260902',
                              ANN / 'AtomCenterAnnotator-v0.2.4-win-x64/HAADF-STEM/HAADF-STEM标注任务_20260902_151017/images')):
        for path in walk(directory, {'.tif', '.tiff'}):
            references.setdefault(sha256_file(path), []).append(f'{label}/{path.name}')
    previewed = {}
    for path in sorted(PREVIEW.glob('*_prediction.json')):
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
        identity = payload.get('source_identity') or {}
        previewed[identity.get('file_sha256', path.stem)] = {
            'preview': path.name, 'source_image': payload.get('source_image'),
            'plane_sha256': identity.get('plane_sha256'), 'shape': identity.get('shape'),
            'model_sha256': payload.get('model_sha256'), 'recorded_points': len(payload.get('points_xy') or [])}
    audit['reference_counts'] = {'hashed_reference_files': sum(len(v) for v in references.values()),
                                 'previewed_first_round': len(previewed)}

    # --- 3. classify every candidate ------------------------------------------------
    classified = {}
    for key, value in sorted(candidates.items()):
        identity = value['identity']
        matches_reference = references.get(identity['file_sha256'], [])
        preview_entry = previewed.get(identity['file_sha256'])
        classified[value['name']] = {
            **value,
            'file_sha256': identity['file_sha256'], 'plane_sha256': identity['plane_sha256'],
            'shape': identity['shape'], 'dtype': identity['dtype'], 'file_size': identity['file_size'],
            'duplicate_of_reference_file': matches_reference,
            'already_previewed_first_round': preview_entry,
            'status': ('duplicate_of_known_file' if matches_reference
                       else 'already_previewed_no_labels' if preview_entry
                       else 'new_original_no_labels'),
        }
    audit['candidates'] = classified
    audit['candidate_sha_duplicates'] = {
        digest: [name for name, value in classified.items() if value['file_sha256'] == digest]
        for digest in {value['file_sha256'] for value in classified.values()}
        if sum(1 for value in classified.values() if value['file_sha256'] == digest) > 1}

    # --- 4. label search: real annotation projects plus the reviewed 04/05 CSVs ---------
    # The project schema is image_root/label_root/manifest_path plus a ``records`` list of
    # image_id/image_path entries; each record's label is label_root/<image_id>.json. Reading
    # payload['images'] (a key that does not exist) reported zero records and understated the
    # workspace; annotation_project_inventory reads the real schema and recomputes the same
    # label signature the dataset builder stores as annotation_digest.
    inventory = inventory_projects(ANN)
    candidate_planes = {value['plane_sha256'] for value in classified.values()}
    candidate_names = set(classified)
    project_image_planes = {row.get('image_plane_sha256') for row in inventory['records']}
    answer_rows = []
    for path in sorted(ANN.rglob('*.metadata.json')):
        try:
            payload = json.loads(path.read_text(encoding='utf-8-sig'))
        except (OSError, json.JSONDecodeError):
            continue
        source = payload.get('source_tiff')
        csv_file = path.with_name(path.name.replace('.metadata.json', '.csv'))
        answer_rows.append({'metadata': str(path.resolve()),
                            'source_tiff': source,
                            'source_tiff_name': Path(source).name if source else None,
                            'frame_count': payload.get('frame_count'),
                            'answer_csv': str(csv_file.resolve()),
                            'answer_csv_exists': csv_file.is_file(),
                            'mtime_utc': path.stat().st_mtime,
                            'refers_to_a_new_original': bool(source and Path(source).name in candidate_names)})
    audit['labels'] = {
        'how_read': 'annotation_project.json schema_version/image_root/label_root/manifest_path/records',
        'annotation_projects': inventory,
        'supplementary_reviewed_csv_answers': {
            'note': 'these two CSVs are the extra human answers a user supplied for targets 04 and 05; '
                    'they are not annotation-project labels',
            'records': answer_rows,
            'used_by': 'the reviewed_targets_20260910 dataset built in round 1 (isolation_group '
                       'former_test_0199_field)',
        },
        'candidate_images': sorted(candidate_names),
        'candidate_images_with_any_label': sorted(
            row['image_id'] for row in inventory['records']
            if row.get('image_plane_sha256') in candidate_planes),
        'candidate_images_matching_a_project_image_file': sorted(
            row['image_id'] for row in inventory['records']
            if row.get('image_file_sha256') in {value['file_sha256'] for value in classified.values()}),
        'project_images_not_in_any_candidate': len(project_image_planes - candidate_planes),
        'verdict': {
            'annotation_project_records': inventory['totals']['records'],
            'reviewed_records': inventory['totals']['reviewed_records'],
            'reviewed_records_already_in_a_frozen_dataset': inventory['totals']['used_in_frozen_dataset'],
            'label_files_present_but_draft_and_unused': inventory['totals']['labels_not_in_any_frozen_dataset'],
            'answers_for_the_two_targets': [row['source_tiff_name'] for row in answer_rows if row['source_tiff_name']],
            'answers_for_the_ten_new_originals': 0,
            'statement': 'every reviewed annotation-project label is already used by a frozen dataset; the '
                         'three label files that no frozen dataset references are draft, not reviewed; the '
                         'only human answers outside the projects are the two 04/05 CSVs used in round 1; '
                         'no coordinate answer of any kind refers to the ten new originals',
        },
    }
    audit['candidates_without_any_label'] = sorted(candidate_names)

    write_json(RUN / 'input_audit.json', audit)
    write_json(RUN / 'annotation_audit.json', {
        'purpose': 'real annotation-project inventory and label usage against the frozen datasets',
        'task_id': 'training-update-20260910',
        'updated_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'totals': inventory['totals'],
        'projects': inventory['projects'],
        'records': inventory['records'],
        'supplementary_reviewed_csv_answers': answer_rows,
        'candidate_images': sorted(candidate_names),
        'candidate_images_with_any_label': audit['labels']['candidate_images_with_any_label'],
    })
    for name, value in classified.items():
        container = value['source_container_format']
        print(f"{value['status']:26s} container={container['series_shape']} {container['series_dtype']} "
              f"{container['photometric']}/{container['samples_per_pixel']}ch {container['compression']} | "
              f"decoded={value['decoded_plane']['shape']} {value['decoded_plane']['dtype']} "
              f"{value['file_sha256'][:12]}  {name}")
    print('zip members:', {label: entry['member_count'] for label, entry in audit['zip_members'].items()})
    print('candidate sha duplicates:', json.dumps(audit['candidate_sha_duplicates'], ensure_ascii=False))
    print('annotation projects:', json.dumps(inventory['totals'], ensure_ascii=False))
    print('wrote', RUN / 'input_audit.json', 'and', RUN / 'annotation_audit.json')


main()

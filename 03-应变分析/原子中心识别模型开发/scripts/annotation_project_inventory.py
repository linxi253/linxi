"""Real annotation-project inventory for the round-2 audit.

Reads the actual project schema (``image_root``/``label_root``/``manifest_path`` plus a
``records`` list of ``image_id``/``image_path``), recomputes the same label signature the frozen
dataset builder uses, and compares it with the frozen dataset manifests. Original projects and
labels are opened read-only.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from atom_center.annotations import AnnotationProject                        # noqa: E402
from atom_center.model_manifest import sha256_file                          # noqa: E402
from atom_center.source_identity import observe_source                      # noqa: E402
from atom_center.storage import content_digest                              # noqa: E402

ANN = ROOT.parent / '原子标注'
DATASETS = {'real_workflow_20260909_v2': ROOT / 'data/processed/real_workflow_20260909_v2',
            'reviewed_targets_20260910': ROOT / 'data/processed/reviewed_targets_20260910'}


def label_signature(document):
    """Exactly the signature data_workflow.audit_projects stores as annotation_digest."""
    return content_digest({
        "points": sorted(document.points_xy),
        "regions": sorted(document.coverage_regions_xyxy),
        "modality": document.modality,
        "pixel_size": document.metadata.get("pixel_size"),
        "pixel_size_unit": document.metadata.get("pixel_size_unit")})


def frozen_index():
    """annotation_digest -> frozen dataset usage, from the dataset manifests themselves."""
    index = {}
    for dataset, path in DATASETS.items():
        manifest_path = path / 'dataset_manifest.json'
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        for source in manifest.get('sources', []):
            index.setdefault(source.get('annotation_digest'), []).append({
                'dataset': dataset, 'record_key': source.get('record_key'),
                'split': source.get('split'), 'label_file': source.get('label_file'),
                'isolation_group': source.get('isolation_group'),
                'role': 'supervised_adaptation_target' if source.get('isolation_group') == 'former_test_0199_field'
                        else 'training_or_validation'})
    return index


def container_format(path):
    """What is stored in the TIFF container, as opposed to the plane the reader decodes."""
    import tifffile
    with tifffile.TiffFile(path) as tif:
        page, series = tif.pages[0], tif.series[0]
        return {'format': 'TIFF', 'pages': len(tif.pages), 'series_shape': list(series.shape),
                'series_dtype': str(series.dtype), 'series_axes': series.axes,
                'photometric': str(page.photometric), 'samples_per_pixel': int(page.samplesperpixel),
                'bits_per_sample': int(page.bitspersample), 'compression': str(page.compression),
                'planar_config': str(page.planarconfig)}


def inventory_projects(search_root=ANN, projects=None):
    """Per-project and per-record annotation facts, deduplicated by image pixel identity."""
    paths = projects or sorted(search_root.rglob('annotation_project.json'))
    frozen = frozen_index()
    projects_out, records_out = [], []
    for path in paths:
        project = AnnotationProject.load(path)
        payload = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        entry = {'project': str(path.resolve()), 'project_id': project.project_id,
                 'schema_version': payload.get('schema_version'),
                 'image_root': payload.get('image_root'), 'label_root': payload.get('label_root'),
                 'manifest_path': payload.get('manifest_path'),
                 'record_count': len(list(payload.get('records') or [])),
                 'created_utc': payload.get('created_utc'), 'updated_utc': payload.get('updated_utc'),
                 'records': []}
        for record in project.records:
            image_path, label_path = project.image_file(record), project.label_file(record)
            row = {'image_id': record.image_id, 'image_path': str(image_path),
                   'label_file': str(label_path), 'label_exists': label_path.is_file(),
                   'label_sha256': sha256_file(label_path) if label_path.is_file() else None}
            try:
                identity = observe_source(image_path, series_index=record.series_index,
                                          frame_index=record.frame_index, basis='round2_audit_current_file')
                row.update({'image_file_sha256': identity['file_sha256'],
                            'image_plane_sha256': identity['plane_sha256'],
                            'image_shape': identity['shape'], 'image_dtype': identity['dtype']})
            except (OSError, ValueError) as error:
                row['image_error'] = str(error)
            if label_path.is_file():
                try:
                    document = project.load_document(record)
                    row.update({'review_status': document.review_status,
                                'points': len(document.points_xy),
                                'regions': len(document.coverage_regions_xyxy),
                                'annotation_digest': label_signature(document),
                                'pixel_size': document.metadata.get('pixel_size'),
                                'pixel_size_unit': document.metadata.get('pixel_size_unit')})
                    sample, acquisition, provenance = project.effective_group_ids(document)
                    row.update({'sample_id': sample, 'acquisition_id': acquisition,
                                'group_id_source': provenance})
                    uses = frozen.get(row['annotation_digest'], [])
                    row['frozen_dataset_usage'] = uses
                    row['status'] = ('used_in_frozen_dataset' if uses
                                     else 'label_not_referenced_by_any_frozen_dataset')
                except (ValueError, OSError) as error:
                    row['label_error'] = str(error)
                    row['status'] = 'unreadable_label'
            else:
                row['status'] = 'no_label_file'
            entry['records'].append(row)
            records_out.append({**row, 'project': str(path.resolve())})
        manifest = Path(path).parent / (payload.get('manifest_path') or 'manifest.csv')
        entry['manifest_exists'] = manifest.is_file()
        entry['reviewed_records'] = sum(1 for row in entry['records']
                                        if row.get('review_status') == 'reviewed')
        entry['labels_present'] = sum(1 for row in entry['records'] if row['label_exists'])
        entry['used_in_frozen_dataset'] = sum(1 for row in entry['records']
                                              if row.get('status') == 'used_in_frozen_dataset')
        projects_out.append(entry)

    planes = {}
    for row in records_out:
        if row.get('image_plane_sha256'):
            planes.setdefault(row['image_plane_sha256'], []).append(row['image_id'])
    duplicates = {plane: ids for plane, ids in planes.items() if len(ids) > 1}
    return {'projects': projects_out, 'records': records_out,
            'totals': {'projects': len(projects_out),
                       'records': len(records_out),
                       'reviewed_records': sum(1 for row in records_out if row.get('review_status') == 'reviewed'),
                       'labels_present': sum(1 for row in records_out if row['label_exists']),
                       'used_in_frozen_dataset': sum(1 for row in records_out
                                                     if row.get('status') == 'used_in_frozen_dataset'),
                       'labels_not_in_any_frozen_dataset': sum(
                           1 for row in records_out
                           if row.get('status') == 'label_not_referenced_by_any_frozen_dataset'),
                       'distinct_images_by_plane': len(planes),
                       'duplicate_image_identities': duplicates}}


if __name__ == '__main__':
    result = inventory_projects()
    print(json.dumps(result['totals'], ensure_ascii=False, indent=1))
    for project in result['projects']:
        print(f"{Path(project['project']).parent.name}: records={project['record_count']} "
              f"reviewed={project['reviewed_records']} labels={project['labels_present']} "
              f"used_in_frozen={project['used_in_frozen_dataset']} "
              f"image_root={project['image_root']} label_root={project['label_root']}")

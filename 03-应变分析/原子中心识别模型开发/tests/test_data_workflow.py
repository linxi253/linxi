import json
import shutil
from pathlib import Path
import numpy as np
import pytest
import tifffile
from atom_center.annotations import AnnotationProject
from atom_center.data_workflow import (audit_projects, build_dataset, migrate_project,
                                       verify_dataset, DatasetAuditError)
from atom_center.source_identity import SourceChangedError
from atom_center.storage import read_json


def make_project(root, index, *, raw=None, point=(20.25, 25.75), empty=False):
    image_root = root/"images"
    image_root.mkdir(parents=True)
    if raw is None:
        raw = np.random.default_rng(index).integers(0, 1000, (64, 96), dtype=np.uint16)
    tifffile.imwrite(image_root/"field.tif", raw, photometric="minisblack")
    project = AnnotationProject.create(root/"annotation_project.json",
                                      image_root=image_root, modality="haadf_stem")
    doc = project.load_document(project.records[0])
    doc.metadata.update(sample_id="sample", acquisition_id=f"batch-{index}")
    doc.points_xy = [] if empty else [point]
    doc.coverage_regions_xyxy = [(0., 0., 96., 64.)]
    doc.review_status = "reviewed"
    project.save_document(doc)
    return project


def test_source_replacement_is_detected_even_same_shape(tmp_path):
    project = make_project(tmp_path, 1)
    tifffile.imwrite(project.image_file(project.records[0]), np.ones((64, 96), dtype=np.uint16))
    with pytest.raises(SourceChangedError):
        project.load_document(project.records[0])


def test_spacing_boxes_preserve_truth_images_splits_and_centers(tmp_path):
    projects = [make_project(tmp_path/f"p{i}", i) for i in range(2)]
    for project in projects:
        doc = project.load_document(project.records[0])
        doc.points_xy = [(20.25, 25.75), (60.25, 25.75)]
        project.save_document(doc)
    paths = [p.project_path for p in projects]
    fixed, a = verify_dataset(build_dataset(paths, tmp_path/'fixed', tile_size=64, workflow_only=True))
    adaptive, b = verify_dataset(build_dataset(paths, tmp_path/'spacing', tile_size=64, workflow_only=True, box_policy='spacing'))
    assert fixed['content_sha256'] != adaptive['content_sha256']
    assert [(s['record_key'],s['split']) for s in fixed['sources']] == [(s['record_key'],s['split']) for s in adaptive['sources']]
    assert all(s['supervision_box_size_px'] == 16. for s in adaptive['sources'])
    for path in fixed['artifacts']:
        if path.startswith(('raw/','truth/','images/')):
            assert fixed['artifacts'][path] == adaptive['artifacts'][path]
        elif path.startswith('labels/'):
            if not (a/path).read_text().strip():
                assert not (b/path).read_text().strip()
                continue
            old = np.loadtxt(a/path, ndmin=2); new = np.loadtxt(b/path, ndmin=2)
            if old.size:
                np.testing.assert_array_equal(old[:,:3], new[:,:3])
                assert np.all(new[:,3:] >= old[:,3:])
    with pytest.raises(ValueError, match='box_policy'):
        build_dataset(paths, tmp_path/'invalid', box_policy='unknown')


def test_workflow_only_preserves_provenance_failures_and_whole_task_groups(tmp_path):
    projects = [make_project(tmp_path/f"p{i}", i) for i in range(2)]
    for project in projects:
        doc = project.load_document(project.records[0])
        doc.metadata["acquisition_id"] = ""
        project.save_document(doc)
        payload = read_json(project.project_path)
        payload["name"] = project.project_path.parent.name
        for record in payload["records"]:
            record.pop("source_identity")
        project.project_path.write_text(json.dumps(payload), encoding="utf-8")
    paths = [p.project_path for p in projects]
    with pytest.raises(DatasetAuditError):
        build_dataset(paths, tmp_path/"formal", tile_size=64)
    result = build_dataset(paths, tmp_path/"workflow", tile_size=64, workflow_only=True)
    manifest, _ = verify_dataset(result)
    assert manifest["purpose"] == "workflow_validation"
    assert manifest["group_counts"] == {"train": 1, "val": 1}
    assert not manifest["formal_audit_passed"]
    assert {e["code"] for e in manifest["deferred_provenance_errors"]} == {"missing_acquisition", "current_source_only"}
    assert not (result.parent/"images/test").exists()


def test_duplicate_versions_require_selection_and_cannot_split(tmp_path):
    a = make_project(tmp_path/"a", 1)
    raw = tifffile.imread(a.image_file(a.records[0]))
    b = make_project(tmp_path/"b", 2, raw=raw, point=(21., 26.))
    report = audit_projects([a.project_path, b.project_path])
    assert not report["ok"]
    assert any(e["code"] == "annotation_conflict" for e in report["errors"])
    assert report["isolation_groups"] == 1
    duplicate = report["duplicates"][0]
    with pytest.raises(DatasetAuditError):
        build_dataset([a.project_path, b.project_path], tmp_path/"conflicted-workflow",
                      tile_size=64, workflow_only=True)
    chosen = audit_projects([a.project_path, b.project_path],
        selections={duplicate["plane_sha256"]: duplicate["record_keys"][0]})
    assert chosen["ok"]
    assert chosen["selected_records"] == 1


def test_legacy_migration_preserves_labels_and_records_limited_identity(tmp_path):
    project = make_project(tmp_path/"old", 3)
    payload = read_json(project.project_path)
    payload["schema_version"] = 1
    for row in payload["records"]:
        row.pop("source_identity", None)
    project.project_path.write_text(json.dumps(payload), encoding="utf-8")
    before = project.label_file(project.records[0]).read_bytes()
    migrated = AnnotationProject.load(migrate_project(project.project_path, tmp_path/"new"))
    assert migrated.label_file(migrated.records[0]).read_bytes() == before
    assert migrated.records[0].source_identity["basis"] == "migration_current_file"
    assert not audit_projects([migrated.project_path])["ok"]
    assert audit_projects([migrated.project_path], confirm_current_sources=True)["ok"]


def test_dataset_reproducibility_portability_background_and_tampering(tmp_path):
    projects = [make_project(tmp_path/f"p{i}", i, empty=(i == 4)) for i in range(5)]
    paths = [p.project_path for p in projects]
    a = build_dataset(paths, tmp_path/"a", tile_size=64)
    b = build_dataset(paths[::-1], tmp_path/"b", tile_size=64)
    first, _ = verify_dataset(a)
    second, _ = verify_dataset(b)
    assert first["content_sha256"] == second["content_sha256"]
    assert first["group_counts"]["test"] >= 2
    assert any(c["point_count"] == 0 for c in first["crops"])
    unexpected = tmp_path/"a/images/train/unlisted.npy"
    unexpected.write_bytes(b"stale image cache")
    with pytest.raises(DatasetAuditError, match="unexpected/missing"):
        verify_dataset(a)
    unexpected.unlink()
    moved = tmp_path/"moved"
    shutil.copytree(tmp_path/"a", moved)
    verify_dataset(moved)
    manifest = read_json(moved/"dataset_manifest.json")
    manifest["sources"][0]["split"] = "test" if manifest["sources"][0]["split"] != "test" else "train"
    (moved/"dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetAuditError, match="fingerprint"):
        verify_dataset(moved)
    target = tmp_path/"b"/second["crops"][0]["label"]
    target.write_text("", encoding="ascii")
    # Choose an artifact with a label if the first happens to be an empty ROI.
    image = tmp_path/"b"/second["crops"][0]["image"]
    image.write_bytes(b"changed")
    with pytest.raises(DatasetAuditError):
        verify_dataset(b)


def test_record_and_label_frame_identity_cannot_drift(tmp_path):
    from dataclasses import replace
    project = make_project(tmp_path, 12)
    with pytest.raises(ValueError, match="observed source"):
        replace(project.records[0], frame_index=1)
    document = project.load_document(project.records[0])
    document.frame_index = 1
    with pytest.raises(ValueError, match="source frame"):
        project.save_document(document)


def test_data_grouping_links_parent_image_even_across_acquisitions(tmp_path):
    a = make_project(tmp_path/"a", 10)
    b = make_project(tmp_path/"b", 11)
    document = b.load_document(b.records[0])
    document.metadata["parent_source_sha256"] = a.records[0].source_identity["file_sha256"]
    b.save_document(document)
    report = audit_projects([a.project_path, b.project_path])
    assert report["ok"] and report["isolation_groups"] == 1

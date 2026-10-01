"""Reviewable annotation migration, source grouping, and immutable datasets."""
from __future__ import annotations

import csv
import math
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile
import yaml

from .annotations import AnnotationProject, validate_document
from .coordinates import points_to_yolo
from .preprocessing import prepare_tile
from .image_io import load_image, normalize_percentile
from .model_manifest import sha256_file
from .source_identity import observe_source, verify_source
from .splitting import grouped_split_records
from .storage import (content_digest, new_artifact_directory, read_json, relative_path,
                      resolve_path, write_json)


class DatasetAuditError(ValueError):
    pass


def migrate_project(project_path, output):
    """Copy labels unchanged; newly observed hashes do not certify historical files."""
    source = AnnotationProject.load(project_path)
    output = Path(output).resolve()
    with new_artifact_directory(output) as temporary:
        records = []
        for record in source.records:
            verify_source(source.image_file(record), record.source_identity)
            identity = record.source_identity or observe_source(
                source.image_file(record), series_index=record.series_index,
                frame_index=record.frame_index, basis="migration_current_file")
            records.append(replace(record, source_identity=identity))
            label = source.label_file(record)
            if label.exists():
                (temporary/"labels").mkdir(exist_ok=True)
                shutil.copy2(label, temporary/"labels"/label.name)
        payload = source.to_dict()
        payload.update(schema_version=2, records=[r.to_dict() for r in records],
                       image_root=relative_path(source.image_root, output),
                       label_root="labels", manifest_path="manifest.csv")
        write_json(temporary/"annotation_project.json", payload)
        AnnotationProject.load(temporary/"annotation_project.json").write_manifest()
        write_json(temporary/"migration_report.json", {
            "source_project": str(source.project_path),
            "source_project_sha256": sha256_file(source.project_path),
            "records": len(records), "labels_copied_without_changes": True,
            "identity_note": "Missing hashes observe current files only, not annotation-time identity."})
    return output/"annotation_project.json"


def _key(project, record):
    return content_digest({"project_id": project.project_id, "modality": project.modality, "image_id": record.image_id})[:24]


def audit_projects(project_paths, *, selections=None, confirm_current_sources=False):
    """Read-only audit; selections maps a plane digest to one approved record key."""
    selections = dict(selections or {})
    records, errors, warnings = [], [], []
    seen = set()
    for path in sorted({str(Path(p).resolve()) for p in project_paths}):
        project = AnnotationProject.load(path)
        for record in project.records:
            key = _key(project, record)
            try:
                document = project.load_document(record)
                if document.review_status != "reviewed":
                    continue
                validate_document(document, for_review=True)
                if key in seen:
                    raise DatasetAuditError("duplicate project identity; supply one copy of each project")
                seen.add(key)
                identity = record.source_identity or observe_source(
                    project.image_file(record), series_index=record.series_index,
                    frame_index=record.frame_index, basis="audit_current_file")
                sample, acquisition, provenance = project.effective_group_ids(document)
                label_signature = content_digest({
                    "points": sorted(document.points_xy),
                    "regions": sorted(document.coverage_regions_xyxy),
                    "modality": document.modality,
                    "pixel_size": document.metadata.get("pixel_size"),
                    "pixel_size_unit": document.metadata.get("pixel_size_unit")})
                records.append({
                    "record_key": key, "source_project": str(project.project_path),
                    "source_image": str(project.image_file(record)),
                    "label_file": str(project.label_file(record)),
                    "label_sha256": sha256_file(project.label_file(record)),
                    "source_identity": identity, "sample_id": sample,
                    "acquisition_id": acquisition, "group_id_source": provenance,
                    "modality": project.modality, "points_xy": document.points_xy,
                    "regions_xyxy": document.coverage_regions_xyxy,
                    "metadata": dict(document.metadata), "annotation_digest": label_signature})
            except (ValueError, OSError) as error:
                errors.append({"record_key": key, "code": "invalid_source_or_label", "message": str(error)})
    records.sort(key=lambda r: r["record_key"])
    if not records:
        errors.append({"code": "no_reviewed_data", "message": "没有可使用的已审核标注"})
    if len({r["modality"] for r in records}) > 1:
        errors.append({"code": "mixed_modality", "message": "HAADF-STEM 与 HRTEM 必须分别组装"})
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    token_owner = {}
    planes = {}
    for i, record in enumerate(records):
        identity, metadata = record["source_identity"], record["metadata"]
        tokens = [
            ("file", identity["file_sha256"]),
            ("plane", identity["plane_sha256"]),
            ("acquisition", record["sample_id"], record["acquisition_id"])]
        if metadata.get("parent_field_id", "").strip():
            tokens.append(("field", record["sample_id"], metadata["parent_field_id"].strip()))
        if metadata.get("parent_source_sha256", "").strip():
            tokens.append(("file", metadata["parent_source_sha256"].strip().lower()))
        for token in tokens:
            if token in token_owner:
                parent[find(i)] = find(token_owner[token])
            else:
                token_owner[token] = i
        planes.setdefault(identity["plane_sha256"], []).append(i)
    components = {}
    for i, record in enumerate(records):
        components.setdefault(find(i), []).append(record["record_key"])
    for i, record in enumerate(records):
        record["isolation_group"] = content_digest(sorted(components[find(i)]))[:24]
        record["selected"] = True

    duplicates = []
    for plane, indices in planes.items():
        if len(indices) == 1:
            continue
        versions = {records[i]["annotation_digest"] for i in indices}
        keys = [records[i]["record_key"] for i in indices]
        choice = selections.get(plane)
        if choice is not None and choice not in keys:
            errors.append({"code": "invalid_selection", "plane_sha256": plane, "message": "选择的标注版本不存在"})
        elif len(versions) > 1 and choice is None:
            errors.append({"code": "annotation_conflict", "plane_sha256": plane,
                           "record_keys": keys, "message": "重复原图有不同标注，请指定正式版本或先合并审核"})
        else:
            choice = choice or min(keys)
            for i in indices:
                records[i]["selected"] = records[i]["record_key"] == choice
        duplicates.append({"plane_sha256": plane, "record_keys": keys,
                           "different_annotations": len(versions) > 1, "selected_record": choice})
    for unused in sorted(set(selections)-set(planes)):
        errors.append({"code": "unknown_selection", "message": f"unknown plane digest: {unused}"})
    for record in records:
        if not record["selected"]:
            continue
        if record["group_id_source"] != "provided_or_path":
            errors.append({"record_key": record["record_key"], "code": "missing_acquisition",
                           "message": "正式划分前需要真实 sample_id 和 acquisition_id"})
        if record["source_identity"]["basis"] != "observed_at_discovery":
            note = {"record_key": record["record_key"], "code": "current_source_only",
                    "message": "原图身份仅在当前观察；未证明标注时的原图身份"}
            (warnings if confirm_current_sources else errors).append(note)
        if not record["metadata"].get("pixel_size", "").strip():
            warnings.append({"record_key": record["record_key"], "code": "missing_pixel_size",
                             "message": "缺少像素标定，只能报告像素误差，不能换算 pm"})
    return {"schema_version": 1, "ok": not errors, "records": records, "errors": errors,
            "warnings": warnings, "duplicates": duplicates,
            "selected_records": sum(r["selected"] for r in records),
            "isolation_groups": len({r["isolation_group"] for r in records if r["selected"]})}


def _yolo_lines(points, width, height, box_size):
    centers = points_to_yolo(points, width, height)
    lines = []
    for cx, cy in centers:
        if not (0 < cx < 1 and 0 < cy < 1):
            raise DatasetAuditError("原子中心位于图像像素支撑边缘之外，请复核标注/完整 ROI 边界")
        # Shrink symmetrically at the image edge, never move the center.
        bw = min(box_size/width, 2*cx, 2*(1-cx))
        bh = min(box_size/height, 2*cy, 2*(1-cy))
        lines.append(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
    return "\n".join(lines) + ("\n" if lines else "")


def build_dataset(
    project_paths, output, *, tile_size=640, box_size_px=8.,
    percentiles=(1., 99.), seed=20260831, ratios=None, selections=None,
    confirm_current_sources=False, development=False, workflow_only=False,
    box_policy="fixed",
):
    if not isinstance(tile_size, int) or tile_size < 32:
        raise ValueError("tile_size must be an integer >= 32")
    if not math.isfinite(box_size_px) or box_size_px <= 0:
        raise ValueError("box_size_px must be positive and finite")
    if box_policy not in {"fixed", "spacing"}:
        raise ValueError("box_policy must be fixed or spacing")
    low, high = (float(v) for v in percentiles)
    if not 0 <= low < high <= 100:
        raise ValueError("invalid normalization percentiles")
    audit = audit_projects(project_paths, selections=selections,
                           confirm_current_sources=confirm_current_sources)
    deferred = [e for e in audit["errors"] if workflow_only and e["code"] in {"missing_acquisition", "current_source_only"}]
    blocking = [e for e in audit["errors"] if e not in deferred]
    if blocking:
        raise DatasetAuditError("数据审计未通过：" + "; ".join(e["message"] for e in audit["errors"]))
    selected = [r for r in audit["records"] if r["selected"]]
    if not development and any(r["metadata"].get("data_origin") == "synthetic" for r in selected):
        raise DatasetAuditError("模拟数据必须标记 development，不能冒充正式实验数据")
    if workflow_only:
        if ratios is not None and ratios != {"train": .8, "val": .2}:
            raise ValueError("workflow_only uses train/val source groups and no test split")
        ratios = {"train": .8, "val": .2}
    else:
        ratios = ratios or {"train": .8, "val": .1, "test": .1}
    split = grouped_split_records(
        [{"record_key": r["record_key"], "group": r["isolation_group"]} for r in selected],
        group_key="group", ratios=ratios, seed=seed,
        min_groups_per_split={} if workflow_only else {"test": 1 if development else 2})
    membership = {r["record_key"]: name for name, rows in split.records_by_split.items() for r in rows}
    target = Path(output).resolve()
    sources, crops, artifacts = [], [], {}
    with new_artifact_directory(target) as temp:
        for directory in ("raw", "truth", "images", "labels"):
            (temp/directory).mkdir()
        for name in ratios:
            (temp/"images"/name).mkdir()
            (temp/"labels"/name).mkdir()
        for row in selected:
            identity = row["source_identity"]
            verify_source(row["source_image"], identity)
            raw = load_image(row["source_image"], series_index=identity["series_index"],
                             frame_index=identity["frame_index"], normalize=False,
                             preserve_dtype=True).image
            key, name = row["record_key"], membership[row["record_key"]]
            if sha256_file(row["label_file"]) != row["label_sha256"]:
                raise DatasetAuditError(f"标注在审计后变化：{row['label_file']}")
            raw_path = f"raw/{key}.tif"
            truth_path = f"truth/{key}.json"
            tifffile.imwrite(temp/raw_path, raw, photometric="minisblack")
            write_json(temp/truth_path, {"points_xy": row["points_xy"], "regions_xyxy": row["regions_xyxy"],
                                        "metadata": row["metadata"], "coordinate_order": "xy"})
            sources.append({
                "record_key": key, "split": name, "isolation_group": row["isolation_group"],
                "raw": raw_path, "truth": truth_path, "modality": row["modality"],
                "source_identity": identity, "annotation_digest": row["annotation_digest"],
                "source_project": relative_path(row["source_project"], target),
                "source_image": relative_path(row["source_image"], target),
                "label_file": relative_path(row["label_file"], target),
                "sample_id": row["sample_id"], "acquisition_id": row["acquisition_id"]})
            points = np.asarray(row["points_xy"], dtype=float).reshape(-1, 2)
            source_box_size = float(box_size_px)
            if box_policy == "spacing":
                # Synthetic supervision boxes only: never alter measured centers.
                # Bound sparse-field estimates so incomplete labels cannot create
                # arbitrarily large targets. Original point truth remains the metric.
                if len(points) >= 2:
                    from scipy.spatial import cKDTree
                    spacing = cKDTree(points).query(points, k=2)[0][:, 1]
                    source_box_size = float(np.clip(.4*np.median(spacing), box_size_px, 3*box_size_px))
                sources[-1]["supervision_box_size_px"] = source_box_size
            for region_index, bounds in enumerate(row["regions_xyxy"]):
                x0, y0, x1, y1 = (int(v) for v in bounds)
                owned = points[(points[:, 0] >= x0) & (points[:, 0] < x1)
                               & (points[:, 1] >= y0) & (points[:, 1] < y1)]
                if len(owned) and (np.any(owned[:, 0]+.5 >= x1) or np.any(owned[:, 1]+.5 >= y1)):
                    raise DatasetAuditError(f"{key}: 完整 ROI 最后一列/行的亚像素点超出像素支撑边缘，请扩展并审核 ROI")
                for top in range(y0, y1, tile_size):
                    for left in range(x0, x1, tile_size):
                        bottom, right = min(y1, top+tile_size), min(x1, left+tile_size)
                        crop = raw[top:bottom, left:right]
                        local = owned[(owned[:, 0]+.5 >= left) & (owned[:, 0]+.5 < right)
                                      & (owned[:, 1]+.5 >= top) & (owned[:, 1]+.5 < bottom)]-[left, top]
                        export_id = f"{key}_r{region_index}_x{left}_y{top}"
                        image_path, label_path = f"images/{name}/{export_id}.png", f"labels/{name}/{export_id}.txt"
                        pixels, transform = prepare_tile(crop, tile_size, low=low, high=high)
                        network_points = transform.forward_points(local)-.5
                        Image.fromarray(pixels).save(temp/image_path)
                        # Box widths follow the resize; raw floating point truth stays unchanged.
                        box_scale = float(min(transform.scale_xy))
                        (temp/label_path).write_text(_yolo_lines(network_points, tile_size, tile_size, source_box_size*box_scale), encoding="ascii")
                        crops.append({"export_id": export_id, "record_key": key, "split": name,
                                      "image": image_path, "label": label_path,
                                      "roi_xyxy": [left, top, right, bottom], "point_count": len(local)})
            verify_source(row["source_image"], identity)
        data = {name: f"images/{name}" for name in ratios}
        data["names"] = {0: "atom_column"}
        (temp/"data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        for file in sorted(temp.rglob("*")):
            if file.is_file():
                artifacts[file.relative_to(temp).as_posix()] = sha256_file(file)
        payload = {
            "schema_version": 1, "modality": selected[0]["modality"],
            "purpose": "workflow_validation" if workflow_only else "development" if development else "experimental",
            "formal_audit_passed": audit["ok"], "deferred_provenance_errors": deferred,
            "seed": seed, "split_ratios": ratios, "group_counts": split.group_counts,
            "tile_size": tile_size, "box_size_px": box_size_px,
            "preprocessing_contract": "raw_tile_percentile_uint8_letterbox_v1",
            "normalization": {"method": "percentile", "low": low, "high": high, "scope": "raw_tile"},
            "coordinate_convention": "integer_pixel_centers_xy",
            "sources": sources, "crops": crops, "artifacts": artifacts,
            "max_train_atoms_per_crop": max((c["point_count"] for c in crops if c["split"] == "train"), default=0),
            "audit_warnings": audit["warnings"]}
        # External locations and observation basis are provenance, not content.
        if box_policy != "fixed":
            payload["box_policy"] = {"name": box_policy, "spacing_fraction": .4,
                                     "minimum_px": box_size_px, "maximum_px": 3*box_size_px}
        fingerprint = dataset_fingerprint(payload)
        payload["content_sha256"] = content_digest(fingerprint)
        payload["fingerprint"] = fingerprint
        write_json(temp/"dataset_manifest.json", payload)
        write_json(temp/"audit_report.json", audit)
        with (temp/"source_map.csv").open("w", encoding="utf-8", newline="") as stream:
            fields = ["export_id", "record_key", "split", "image", "label", "point_count", "roi_xyxy"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(crops)
    return target/"dataset_manifest.json"


def verify_dataset(path):
    path = Path(path).resolve()
    manifest_path = path if path.name == "dataset_manifest.json" else path.parent/"dataset_manifest.json" if path.is_file() else path/"dataset_manifest.json"
    manifest = read_json(manifest_path)
    root = manifest_path.parent
    if manifest.get("schema_version") != 1:
        raise DatasetAuditError("unsupported dataset manifest")
    if (dataset_fingerprint(manifest) != manifest["fingerprint"]
            or content_digest(manifest["fingerprint"]) != manifest["content_sha256"]):
        raise DatasetAuditError("dataset fingerprint has changed")
    for relative, expected in manifest["artifacts"].items():
        file = resolve_path(relative, root)
        if not file.is_relative_to(root) or not file.is_file() or sha256_file(file) != expected:
            raise DatasetAuditError(f"dataset artifact missing or changed: {relative}")
    # A directory loader also sees later additions, including stale .npy image
    # caches. Verifying only the originally listed bytes would miss these.
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            prefix = f"{kind}/{split}/"
            expected_files = {p for p in manifest["artifacts"] if p.startswith(prefix)}
            actual_files = {p.relative_to(root).as_posix() for p in (root/kind/split).rglob("*") if p.is_file()}
            if actual_files != expected_files:
                raise DatasetAuditError(f"unexpected/missing dataset files in {prefix}: {sorted(actual_files ^ expected_files)}")
    groups = {}
    for source in manifest["sources"]:
        group, name = source["isolation_group"], source["split"]
        if group in groups and groups[group] != name:
            raise DatasetAuditError("source group leaked across splits")
        groups[group] = name
    return manifest, root


def dataset_fingerprint(payload):
    fingerprint = {k: v for k, v in payload.items()
                   if k not in {"sources", "audit_warnings", "fingerprint", "content_sha256"}}
    fingerprint["sources"] = [
        {k: v for k, v in s.items()
         if k not in {"source_project", "source_image", "label_file", "source_identity"}}
        | {"file_sha256": s["source_identity"]["file_sha256"],
           "plane_sha256": s["source_identity"]["plane_sha256"]} for s in payload["sources"]]
    return fingerprint

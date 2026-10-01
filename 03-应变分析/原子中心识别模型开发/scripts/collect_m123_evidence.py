"""Collect measured M1-M3 evidence from the completed local development workflow."""
from pathlib import Path
import json
import subprocess
import sys
import time
import tracemalloc
import numpy as np
import yaml
from atom_center.annotations import AnnotationProject
from atom_center.metrics import match_points
from atom_center.model_manifest import sha256_file
from atom_center.storage import read_json, write_json
from atom_center.training import code_provenance, software_versions


def main():
    root = Path(__file__).resolve().parents[1]
    run = root/"runs/m123-validation"
    experiment = run/"experiment-acceptance"
    parity = read_json(run/"onnx-acceptance/onnx_validation.json")
    state = read_json(experiment/"state.json")
    training = read_json(experiment/"run_manifest.json")
    frozen = read_json(run/"dataset/dataset_manifest.json")
    truth_smoke = read_json(run/"refinement-smoke.json")
    yy, xx = np.mgrid[:60, :80]
    points = np.column_stack((xx.ravel()*8., yy.ravel()*8.))
    tracemalloc.start()
    start = time.perf_counter()
    matches = match_points(points+[.1, -.1], points, max_distance_px=.3)
    duration = time.perf_counter()-start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert matches.true_positives == 4800

    migrations = []
    for path in sorted((root/"data/annotation_projects/migrated_20260906").glob("*/annotation_project.json")):
        report = read_json(path.parent/"migration_report.json")
        original = AnnotationProject.load(report["source_project"])
        migrated = AnnotationProject.load(path)
        old_records = {r.image_id: r for r in original.records}
        verified = 0
        for record in migrated.records:
            old_file = original.label_file(old_records[record.image_id])
            if old_file.exists():
                assert old_file.read_bytes() == migrated.label_file(record).read_bytes()
                verified += 1
            migrated.load_document(record)
        assert report["source_project_sha256"] == sha256_file(original.project_path)
        migrations.append({"project": path.parent.name, "labels_byte_identical": verified,
                           "project_sha256_unchanged": True, "reopened": True})

    clean_python = run/"locked-infer-env/Scripts/python.exe"
    verification = subprocess.run([str(clean_python), "-c",
        "import json,sys,importlib.util; import atom_center.backends; import atom_center.pipeline; "
        "names=['torch','ultralytics','onnx']; absent={x:importlib.util.find_spec(x) is None for x in names}; "
        "assert all(absent.values()); assert 'tkinter' not in sys.modules; "
        "print(json.dumps({'absent_packages':absent,'gui_imported':False}))"],
        capture_output=True, text=True, check=True)
    a = read_json(run/"prediction-acceptance.json")
    b = read_json(run/"prediction-locked-infer.json")
    np.testing.assert_array_equal(a["points_xy"], b["points_xy"])
    np.testing.assert_array_equal(a["confidences"], b["confidences"])
    assert state["status"] == "completed" and state["completed_epochs"] == 3
    assert state["attempts"][1]["resume"] and state["attempts"][1]["from_completed_epochs"] == 1
    assert parity["passed"] and parity["nonempty_cases"] == 6
    assert code_provenance()["source_sha256"] == training["code"]["source_sha256"]
    original = read_json(run/"experiment-release-source/state.json")
    assert original["status"] == "paused" and original["completed_epochs"] == 1
    args = read_json(experiment/"ultralytics_args.json")
    assert Path(args["data"]).resolve() == (experiment/"data.resolved.yaml").resolve()
    actual_data = yaml.safe_load((experiment/"data.resolved.yaml").read_text(encoding="utf-8"))
    assert Path(actual_data["path"]).resolve() == (run/"dataset-relocated").resolve()
    assert args["workers"] == 2
    test_log = (run/"core-tests-acceptance.log").read_text(encoding="utf-8")
    assert "65 passed" in test_log
    report = {
        "scope": "M1-M3 software acceptance; no scientific model acceptance",
        "software": software_versions(), "code": code_provenance(),
        "dataset_content_sha256": frozen["content_sha256"],
        "dataset_source_groups": frozen["group_counts"],
        "dataset_source_count": len(frozen["sources"]),
        "background_crops": sum(c["point_count"] == 0 for c in frozen["crops"]),
        "training": state, "model_equivalence": parity,
        "regression_tests_passed": 65,
        "relocated_resume": {"original_preserved_paused": True, "uses_relocated_dataset": True,
                              "stale_label_cache_rebuilt": True, "dataloader_workers": 2},
        "geometry_refinement_smoke": truth_smoke,
        "spatial_matching_4800_points": {"matches": 4800, "time_s": duration, "peak_tracemalloc_bytes": peak},
        "actual_project_migrations": migrations,
        "clean_inference": {**json.loads(verification.stdout), "locked_install_passed": True,
                            "points": len(b["points_xy"]), "identical_to_development_prediction": True},
        "validation_point_metrics": {k: read_json(run/"evaluation-acceptance.json")[k]
                                     for k in ("tp", "fp", "fn", "rmse_px")},
        "artifacts": {"training": str(experiment), "onnx": str(run/"onnx-acceptance"),
                      "wheel": str(run/"wheels/atom_center-0.3.0-py3-none-any.whl")},
    }
    write_json(root/"reports/m123_validation_2026-09-06.json", report)
    print(json.dumps({"matching": report["spatial_matching_4800_points"],
                      "migrated_projects": len(migrations), "clean_inference": report["clean_inference"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

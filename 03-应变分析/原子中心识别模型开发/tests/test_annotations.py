import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from atom_center.annotations import (
    AnnotationProject,
    export_yolo_dataset,
    make_image_id,
    project_statistics,
    validate_document,
)


def _write_image(path: Path, shape: tuple[int, int] = (10, 12)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.arange(shape[0] * shape[1], dtype=np.uint16).reshape(shape)
    tifffile.imwrite(path, image, photometric="minisblack", metadata={"axes": "YX"})


def _make_project(tmp_path: Path) -> AnnotationProject:
    image_root = tmp_path / "raw"
    _write_image(image_root / "sample-a" / "run-01" / "field.tif")
    return AnnotationProject.create(
        tmp_path / "projects" / "haadf.json",
        image_root=image_root,
        label_root=tmp_path / "labels",
        manifest_path=tmp_path / "manifest.csv",
        modality="haadf_stem",
    )


def _fill_haadf_review_metadata(document: object) -> None:
    document.metadata.update(
        {
            "sample_id": "sample-a",
            "acquisition_id": "run-01",
            "pixel_size": "0.2",
            "accelerating_voltage_kv": "200",
            "detector_inner_angle_mrad": "68",
            "detector_outer_angle_mrad": "280",
        }
    )


def test_project_discovers_images_and_infers_group_ids(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    assert len(project.records) == 1
    record = project.records[0]
    assert record.image_shape == (10, 12)
    document = project.load_document(record)
    assert document.metadata["sample_id"] == "sample-a"
    assert document.metadata["acquisition_id"] == "run-01"
    assert project.sync_images() == (0, 0)

    reloaded = AnnotationProject.load(project.project_path)
    assert reloaded.records == project.records
    assert make_image_id("a/file.tif", series_index=0, frame_index=None) == make_image_id(
        "a/file.tif", series_index=0, frame_index=None
    )


def test_multiframe_tiff_becomes_independent_records(tmp_path: Path) -> None:
    image_root = tmp_path / "raw"
    image_root.mkdir()
    stack = np.stack(
        [np.full((7, 8), value, dtype=np.uint16) for value in (10, 20, 30)]
    )
    tifffile.imwrite(
        image_root / "stack.tif",
        stack,
        photometric="minisblack",
        metadata={"axes": "QYX"},
    )
    project = AnnotationProject.create(
        tmp_path / "project.json",
        image_root=image_root,
        modality="hrtem",
    )
    assert [record.frame_index for record in project.records] == [0, 1, 2]
    assert {record.image_shape for record in project.records} == {(7, 8)}


def test_review_requires_coverage_but_allows_blank_metadata(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    document = project.load_document(project.records[0])
    document.metadata["sample_id"] = ""
    document.metadata["acquisition_id"] = ""
    with pytest.raises(ValueError, match="完整标注区域"):
        validate_document(document, for_review=True)

    document.coverage_regions_xyxy = [(0.0, 0.0, 12.0, 10.0)]
    validate_document(document, for_review=True)

    document.metadata["pixel_size"] = "not-a-number"
    with pytest.raises(ValueError, match="pixel_size"):
        validate_document(document, for_review=True)
    document.metadata["pixel_size"] = ""
    _fill_haadf_review_metadata(document)
    document.points_xy = [(4.25, 3.75)]
    validate_document(document, for_review=True)

    document.coverage_regions_xyxy = [(0.0, 0.0, 3.0, 3.0)]
    with pytest.raises(ValueError, match="区域之外"):
        validate_document(document, for_review=True)


def test_overlapping_coverage_regions_are_rejected(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    document = project.load_document(project.records[0])
    document.coverage_regions_xyxy = [
        (0.0, 0.0, 6.0, 6.0),
        (5.0, 5.0, 10.0, 9.0),
    ]
    with pytest.raises(ValueError, match="overlap"):
        validate_document(document, for_review=False)

    document.coverage_regions_xyxy = [(0.25, 0.0, 6.0, 6.0)]
    with pytest.raises(ValueError, match="integer pixel bounds"):
        validate_document(document, for_review=False)


def test_hrtem_review_allows_blank_imaging_conditions(tmp_path: Path) -> None:
    image_root = tmp_path / "raw"
    _write_image(image_root / "sample-h" / "run-h" / "field.tif")
    project = AnnotationProject.create(
        tmp_path / "hrtem.json",
        image_root=image_root,
        modality="hrtem",
    )
    document = project.load_document(project.records[0])
    document.coverage_regions_xyxy = [(0.0, 0.0, 12.0, 10.0)]
    for key in tuple(document.metadata):
        if key != "pixel_size_unit":
            document.metadata[key] = ""
    validate_document(document, for_review=True)

    document.metadata.update(
        {
            "pixel_size": "0.15",
            "accelerating_voltage_kv": "300",
            "defocus_nm": "-12.5",
            "spherical_aberration_mm": "0.001",
            "contrast_polarity": "dark",
        }
    )
    validate_document(document, for_review=True)
    document.metadata["spherical_aberration_mm"] = "not-a-number"
    with pytest.raises(ValueError, match="spherical_aberration_mm"):
        validate_document(document, for_review=True)


def test_blank_group_ids_get_stable_project_fallbacks(tmp_path: Path) -> None:
    image_root = tmp_path / "raw"
    _write_image(image_root / "field.tif")
    project = AnnotationProject.create(
        tmp_path / "project.json",
        image_root=image_root,
        modality="haadf_stem",
    )
    record = project.records[0]
    document = project.load_document(record)
    document.coverage_regions_xyxy = [(0.0, 0.0, 12.0, 10.0)]
    document.points_xy = [(4.0, 5.0)]
    document.review_status = "reviewed"
    project.save_document(document)

    first_manifest = project.write_manifest()
    with first_manifest.open(encoding="utf-8", newline="") as handle:
        first_row = next(csv.DictReader(handle))
    project.write_manifest()
    with first_manifest.open(encoding="utf-8", newline="") as handle:
        second_row = next(csv.DictReader(handle))

    assert first_row["sample_id"].startswith("auto-sample-")
    assert first_row["acquisition_id"].startswith("auto-acquisition-")
    assert first_row["group_id_source"] == "project_fallback"
    assert "pixel_size" in first_row["missing_metadata_fields"]
    assert second_row["sample_id"] == first_row["sample_id"]
    assert second_row["acquisition_id"] == first_row["acquisition_id"]

    output = tmp_path / "export"
    export_yolo_dataset(project, output)
    with (output / "source_map.csv").open(encoding="utf-8", newline="") as handle:
        export_row = next(csv.DictReader(handle))
    assert export_row["acquisition_id"] == first_row["acquisition_id"]
    assert export_row["group_id_source"] == "project_fallback"


def test_save_manifest_statistics_and_yolo_roi_export(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    record = project.records[0]
    document = project.load_document(record)
    _fill_haadf_review_metadata(document)
    document.coverage_regions_xyxy = [(2.0, 1.0, 8.0, 7.0)]
    document.points_xy = [(3.0, 2.0), (7.0, 6.0)]
    document.atom_diameter_px = 8.0
    document.review_status = "reviewed"
    project.save_document(document)
    manifest = project.write_manifest()

    label_payload = json.loads(project.label_file(record).read_text(encoding="utf-8"))
    assert label_payload["coordinate_order"] == "xy"
    assert label_payload["points_xy"][0] == {"x": 3.0, "y": 2.0}
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["review_status"] == "reviewed"
    assert rows[0]["point_count"] == "2"
    assert float(rows[0]["coverage_fraction"]) == pytest.approx(0.3)

    stats = project_statistics(project)
    assert stats.reviewed == 1
    assert stats.reviewed_points == 2
    assert stats.samples == 1
    assert stats.acquisitions == 1

    output = tmp_path / "derived" / "snapshot"
    summary = export_yolo_dataset(project, output, box_size_px=4.0)
    assert summary.source_images == 1
    assert summary.derived_images == 1
    assert summary.points == 2

    exported_image = Image.open(next((output / "images").glob("*.png")))
    assert exported_image.size == (6, 6)
    label_lines = next((output / "labels").glob("*.txt")).read_text().splitlines()
    first = [float(value) for value in label_lines[0].split()[1:]]
    second = [float(value) for value in label_lines[1].split()[1:]]
    assert first == pytest.approx([0.25, 0.25, 4 / 6, 4 / 6])
    assert second == pytest.approx([5.5 / 6, 5.5 / 6, 4 / 6, 4 / 6])
    export_manifest = json.loads(
        (output / "export_manifest.json").read_text(encoding="utf-8")
    )
    assert "acquisition_id" in export_manifest["split_warning"]

    with pytest.raises(FileExistsError):
        export_yolo_dataset(project, output)


def test_gui_module_can_be_imported_without_opening_window() -> None:
    from atom_center.annotator_gui import AnnotationApp

    assert AnnotationApp.AUTOSAVE_DELAY_MS > 0


def test_window_geometry_fits_requested_content_and_small_screens() -> None:
    from atom_center.annotator_gui import calculate_window_geometry

    assert calculate_window_geometry(
        (720, 510),
        (1920, 1080),
        minimum_size=(600, 380),
    ) == (720, 510, 600, 285)
    assert calculate_window_geometry(
        (1580, 940),
        (1000, 800),
        minimum_size=(900, 600),
    ) == (940, 704, 30, 48)


def test_task_folder_selection_is_forgiving_but_unambiguous(tmp_path: Path) -> None:
    from atom_center.annotator_gui import find_task_project

    task = tmp_path / "received-task"
    project_file = task / "annotation_project.json"
    project_file.parent.mkdir()
    project_file.write_text("{}", encoding="utf-8")
    nested = task / "images" / "sample-a" / "run-a"
    nested.mkdir(parents=True)

    assert find_task_project(task) == project_file.resolve()
    assert find_task_project(nested) == project_file.resolve()
    assert find_task_project(tmp_path) == project_file.resolve()

    legacy_file = tmp_path / "legacy" / "haadf_stem.json"
    legacy_file.parent.mkdir()
    legacy_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "legacy",
                "modality": "haadf_stem",
                "image_root": "images",
                "label_root": "labels",
                "manifest_path": "manifest.csv",
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    assert find_task_project(legacy_file.parent) == legacy_file.resolve()

    second = tmp_path / "second-task" / "annotation_project.json"
    second.parent.mkdir()
    second.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="多个标注任务"):
        find_task_project(tmp_path)


def test_new_task_uses_safe_unique_child_folder(tmp_path: Path) -> None:
    from atom_center.annotator_gui import next_task_directory

    first = next_task_directory(
        tmp_path,
        modality="haadf_stem",
        timestamp="20260831_120000",
    )
    assert first.name == "HAADF-STEM标注任务_20260831_120000"
    first.mkdir()
    second = next_task_directory(
        tmp_path,
        modality="haadf_stem",
        timestamp="20260831_120000",
    )
    assert second.name == "HAADF-STEM标注任务_20260831_120000_2"


def test_startup_layout_fits_at_double_font_scaling() -> None:
    import tkinter as tk

    from atom_center.annotator_gui import StartupDialog

    root = tk.Tk()
    root.withdraw()
    try:
        root.tk.call("tk", "scaling", 2.0)
        dialog = StartupDialog(root)
        assert dialog.content_fits is True
        assert dialog.window_geometry[1] >= dialog.required_size[1]
        assert [button.cget("text") for button in dialog.primary_buttons] == [
            "打开收到的标注任务文件夹…",
            "新建 HAADF-STEM 任务…",
            "新建 HRTEM 任务…",
        ]
    finally:
        root.destroy()


def test_main_sidebar_remains_visible_and_scrollable_on_small_window(
    tmp_path: Path,
) -> None:
    import tkinter as tk
    from types import SimpleNamespace

    from atom_center.annotator_gui import AnnotationApp, create_portable_task

    project = create_portable_task(tmp_path / "small-screen-task", modality="hrtem")
    root = tk.Tk()
    try:
        app = AnnotationApp(root, project)
        root.geometry("900x600")
        root.update_idletasks()
        root.update()
        assert "defocus_nm" in app.metadata_vars
        assert "detector_inner_angle_mrad" not in app.metadata_vars
        assert app.main_frame.pack_slaves()[0] is app.sidebar_shell
        assert app.sidebar_canvas.winfo_width() >= 300
        scroll_bounds = app.sidebar_canvas.bbox("all")
        assert scroll_bounds is not None
        assert scroll_bounds[3] > app.sidebar_canvas.winfo_height()
        before = app.sidebar_canvas.yview()
        app._on_sidebar_mousewheel(SimpleNamespace(delta=-120))
        root.update()
        assert app.sidebar_canvas.yview() != before
    finally:
        root.destroy()


def test_main_sidebar_has_no_horizontal_clipping_at_double_scaling(
    tmp_path: Path,
) -> None:
    import tkinter as tk

    from atom_center.annotator_gui import AnnotationApp, create_portable_task

    project = create_portable_task(tmp_path / "high-dpi-task", modality="haadf_stem")
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 2.0)
        app = AnnotationApp(root, project)
        root.geometry("1200x700")
        root.update_idletasks()
        root.update()
        assert "detector_inner_angle_mrad" in app.metadata_vars
        assert "defocus_nm" not in app.metadata_vars
        content = app.sidebar_canvas.nametowidget(
            app.sidebar_canvas.itemcget(app._sidebar_window, "window")
        )
        assert content.winfo_reqwidth() <= app.sidebar_canvas.winfo_width()
    finally:
        root.destroy()


def test_exported_labeled_image_can_switch_to_next_unannotated_image(
    tmp_path: Path,
) -> None:
    import tkinter as tk

    from atom_center.annotator_gui import AnnotationApp, create_portable_task

    task = tmp_path / "two-image-task"
    _write_image(task / "images" / "01-labeled.tif", shape=(40, 60))
    _write_image(task / "images" / "02-unannotated.tif", shape=(48, 72))
    project = create_portable_task(task, modality="haadf_stem")

    first = project.load_document(project.records[0])
    first.points_xy = [(20.0, 20.0)]
    first.coverage_regions_xyxy = [(0.0, 0.0, 60.0, 40.0)]
    first.review_status = "reviewed"
    project.save_document(first)
    export_yolo_dataset(project, tmp_path / "reviewed-export")

    root = tk.Tk()
    try:
        app = AnnotationApp(root, project)
        root.update_idletasks()
        root.update()
        assert app._point_artist is not None
        assert len(app._region_artists) == 1

        app._next()
        root.update_idletasks()
        root.update()

        assert app.current_index == 1
        assert app.record is project.records[1]
        assert app.raw_image is not None
        assert app.raw_image.shape == (48, 72)
        assert np.asarray(app.axes.images[0].get_array()).shape == (48, 72)
        assert app._point_artist is None
        assert app._region_artists == []

        app._previous()
        root.update_idletasks()
        root.update()
        assert app.current_index == 0
        assert app.raw_image.shape == (40, 60)
        assert app._point_artist is not None
        assert len(app._region_artists) == 1
    finally:
        root.destroy()


def test_portable_task_paths_survive_folder_transfer(tmp_path: Path) -> None:
    from atom_center.annotator_gui import (
        _parser,
        create_portable_task,
        open_or_create_project,
    )

    original = tmp_path / "task-original"
    _write_image(original / "images" / "sample-p" / "run-p" / "field.tif")
    project = create_portable_task(original, modality="haadf_stem")
    payload = json.loads(project.project_path.read_text(encoding="utf-8"))
    assert payload["image_root"] == "images"
    assert payload["label_root"] == "labels"
    assert payload["manifest_path"] == "manifest.csv"

    transferred = tmp_path / "task-transferred"
    shutil.copytree(original, transferred)
    reloaded = AnnotationProject.load(transferred / "annotation_project.json")
    assert reloaded.image_root == (transferred / "images").resolve()
    assert reloaded.image_file(reloaded.records[0]).is_file()
    dragged = open_or_create_project(_parser().parse_args([str(transferred)]))
    assert dragged is not None
    assert dragged.project_path == (transferred / "annotation_project.json").resolve()


def test_packaged_smoke_path_exercises_gui_and_export(tmp_path: Path) -> None:
    import pytest

    from atom_center.annotator_gui import bundled_font_path, run_packaged_smoke_test

    report_path = tmp_path / "smoke.json"
    assert run_packaged_smoke_test(report_path) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["launcher_constructed"] is True
    assert report["launcher_content_fits"] is True
    assert report["launcher_buttons_present"] is True
    assert report["sidebar_scrollable"] is True
    assert report["sidebar_packed_first"] is True
    assert report["sidebar_horizontal_fits"] is True
    assert report["sidebar_vertical_overflow"] is True
    # Audit 56: the .otf is an optional asset that the repository does not ship.
    # Only require successful font registration when the asset actually exists;
    # otherwise the whole check is moot on a clean checkout.
    if report["bundled_font_asset_present"] or bundled_font_path().is_file():
        assert report["bundled_font_registered"] is True
    else:
        pytest.skip("optional bundled font asset not present in this checkout")
    assert report["loaded_shape"] == [64, 80]
    assert report["next_loaded_shape"] == [72, 96]
    assert report["image_switch_passed"] is True
    assert report["exported_points"] == 2

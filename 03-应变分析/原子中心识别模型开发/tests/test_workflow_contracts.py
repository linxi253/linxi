import random
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from atom_center.augmentation import GrayscaleAugment
from atom_center.backends import decode_output
from atom_center.configuration import load_config
from atom_center.coordinates import Letterbox
from atom_center.preprocessing import prepare_tile, input_tensor


def test_layered_configuration_and_invalid_keys_before_model(tmp_path):
    a, b = tmp_path/"common.yaml", tmp_path/"mode.yaml"
    a.write_text("training:\n  epochs: 9\naugmentation:\n  brightness: 0.1\n", encoding="utf-8")
    b.write_text("modality: hrtem\ntraining:\n  epochs: 5\n", encoding="utf-8")
    c = load_config(a, b, {"training.epochs": 2})
    assert c["modality"] == "hrtem" and c["training"]["epochs"] == 2
    assert c["augmentation"]["brightness"] == .1
    with pytest.raises(ValueError, match="unknown configuration"):
        load_config(overrides={"training.brightness": .2})
    with pytest.raises(ValueError, match="unknown configuration"):
        load_config(overrides={"inference.max_dte": 1000})


def test_grayscale_augmentation_seed_and_geometry():
    augment = GrayscaleAugment(probability=1)
    raw = np.arange(100, dtype=np.uint8).reshape(10, 10)
    image = np.repeat(raw[..., None], 3, axis=2)
    points = np.array([[3.25, 5.5]])
    random.seed(321)
    a = augment({"img": image.copy(), "points": points})
    random.seed(321)
    b = augment({"img": image.copy(), "points": points})
    np.testing.assert_array_equal(a["img"], b["img"])
    np.testing.assert_array_equal(a["img"][..., 0], a["img"][..., 2])
    np.testing.assert_array_equal(a["points"], points)


def test_dense_decode_over_300_and_cap_diagnostic():
    points = np.array([(x, y) for y in range(5, 126, 5) for x in range(5, 126, 5)], dtype=float)
    transform = Letterbox.create((160, 180), (192, 192))
    output = np.zeros((1, 5, len(points)), dtype=np.float32)
    output[0, :2] = transform.forward_points(points).T
    output[0, 2:4] = 2.
    output[0, 4] = .9
    result, diagnostics = decode_output(output, transform, conf=.5, iou=.45, max_det=1000)
    assert len(result.points) == 625
    np.testing.assert_allclose(result.points, points, atol=1e-5)
    assert not diagnostics["hit_max_det"]
    with pytest.warns(RuntimeWarning, match="max_det=500"):
        result, diagnostics = decode_output(output, transform, conf=.5, iou=.45, max_det=500)
    assert len(result.points) == 500 and diagnostics["truncated"]


def test_preprocessing_non_square_roundtrip_and_channels():
    raw = np.arange(81*125, dtype=np.uint16).reshape(81, 125)
    gray, transform = prepare_tile(raw, 128)
    tensor = input_tensor(gray)
    assert tensor.shape == (1, 3, 128, 128) and tensor.dtype == np.float32
    np.testing.assert_array_equal(tensor[0, 0], tensor[0, 2])
    points = np.array([[0., 0.], [124., 80.], [13.25, 45.7]])
    np.testing.assert_allclose(transform.inverse_centers(transform.forward_points(points)), points, atol=1e-12)


def test_backend_rejects_an_unsupported_model_contract():
    from atom_center.backends import _Backend
    from atom_center.configuration import model_contract
    from atom_center.model_manifest import ModelVerificationError
    config = load_config()
    contract = model_contract(config)
    contract["padding"] = 0
    with pytest.raises(ModelVerificationError, match="contract"):
        _Backend()._configure(contract, config["inference"])


def test_inference_import_does_not_load_training_or_gui():
    result = subprocess.run([sys.executable, "-c",
        "import sys; import atom_center.backends; import atom_center.pipeline; "
        "assert not any(m in sys.modules for m in ('torch','ultralytics','tkinter','matplotlib.pyplot')); print('ok')"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_legacy_benchmark_keeps_background_and_undefined_rmse(tmp_path, monkeypatch):
    ppa = Path(__file__).resolve().parents[2]/"原子级应力分析-PPA"
    monkeypatch.syspath_prepend(str(ppa))
    from atom_detector.infer import benchmark, detector
    from PIL import Image
    images, labels = tmp_path/"images"/"test", tmp_path/"labels"/"test"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(images/"empty.png")
    (labels/"empty.txt").write_text("")
    fake = SimpleNamespace(_load_image_file=lambda p: np.zeros((32, 32)),
                           detect=lambda img: ([(5.2, 6.3)], [.9]))
    monkeypatch.setattr(detector, "AtomDetector", lambda *a, **kw: fake)
    result = benchmark.benchmark_yolo("unused.pt", benchmark.resolve_eval_dirs(tmp_path))
    assert result["total_fp"] == 1 and result["total_gt"] == 0
    assert np.isnan(result["rmse_px"])
    monkeypatch.setattr(sys, "argv", ["benchmark", "--model", "unused.pt", "--test_data", str(tmp_path)])
    benchmark.main()


def test_legacy_results_half_pixel_max_det_and_raw_roi(monkeypatch):
    ppa = Path(__file__).resolve().parents[2]/"原子级应力分析-PPA"
    monkeypatch.syspath_prepend(str(ppa))
    from atom_detector.infer.detector import AtomDetector
    def array(value):
        return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array(value)))
    recorded = {}
    def predict(raw, **kwargs):
        recorded.update(kwargs)
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=array([[2., 3., 8., 9.]]), conf=array([.8])))]
    monkeypatch.setattr(AtomDetector, "_load_model", lambda self: setattr(self, "_model", SimpleNamespace(predict=predict)))
    detector = AtomDetector("unused", refine=False, max_det=4000, device="cpu")
    detector.model_sha256 = None
    result = detector.detect(np.ones((32, 32)), roi=(2, 4, 22, 24))
    np.testing.assert_allclose(result[0], [[6.5, 9.5]])
    assert recorded["max_det"] == 4000

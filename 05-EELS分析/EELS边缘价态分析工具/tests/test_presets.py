"""预设加载与 CLI 参数合并逻辑的测试。"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from eels_edge_analyzer.models import AnalysisConfig, DistanceBin, ReferenceSpec
from eels_edge_analyzer.presets import (
    default_preset_path,
    load_preset,
    parse_distance_bins_text,
)


def test_builtin_preset_loads_and_validates() -> None:
    overrides = load_preset()
    assert overrides["fit_min_ev"] == 925.0
    assert overrides["fit_max_ev"] == 970.0
    assert overrides["background_min_ev"] == 850.0
    bins = overrides["distance_bins"]
    assert isinstance(bins, tuple) and all(isinstance(item, DistanceBin) for item in bins)
    assert [item.label for item in bins] == ["E1", "E2", "E3", "E4", "Bulk"]
    assert bins[-1].high_nm is None and bins[0].low_nm == 0.0

    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        source = root / "input.dm4"
        source.touch()
        references = tuple(
            ReferenceSpec(f"Cu{state}", state, root / f"cu{state}.dm4") for state in range(3)
        )
        for state in range(3):
            (root / f"cu{state}.dm4").touch()
        config = AnalysisConfig(
            input_path=source,
            output_dir=root / "out",
            references=references,
            **overrides,
        )
        config.validate()


def test_load_preset_rejects_unknown_keys() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "bad.json"
        path.write_text(
            json.dumps({"fit_min_ev": 900.0, "fit_min_Ev_typo": 1.0}), encoding="utf-8"
        )
        try:
            load_preset(path)
        except ValueError as exc:
            assert "fit_min_Ev_typo" in str(exc)
        else:
            raise AssertionError("未知键未被拒绝。")


def test_load_preset_reads_custom_values() -> None:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "custom.json"
        path.write_text(
            json.dumps(
                {
                    "name": "custom",
                    "fit_range_ev": None,  # 未知键，应被拒绝
                }
            ),
            encoding="utf-8",
        )
        try:
            load_preset(path)
        except ValueError:
            pass
        else:
            raise AssertionError("非法键 fit_range_ev 未被拒绝。")

        path.write_text(
            json.dumps(
                {
                    "name": "custom",
                    "fit_min_ev": 900.0,
                    "fit_max_ev": 980.0,
                    "zlp_window_ev": [-4.0, 4.0],
                    "distance_bins_nm": [["E1", 0.0, 3.0], ["Bulk", 3.0, None]],
                    "notes": ["ignored"],
                }
            ),
            encoding="utf-8",
        )
        overrides = load_preset(path)
        assert overrides["fit_min_ev"] == 900.0
        assert overrides["zlp_window_ev"] == (-4.0, 4.0)
        assert [item.label for item in overrides["distance_bins"]] == ["E1", "Bulk"]


def test_parse_distance_bins_text() -> None:
    bins = parse_distance_bins_text("E1:0:2.2, E2:2.2:4.4,Bulk:11")
    assert [item.label for item in bins] == ["E1", "E2", "Bulk"]
    assert (bins[0].low_nm, bins[0].high_nm) == (0.0, 2.2)
    assert bins[-1].low_nm == 11.0 and bins[-1].high_nm is None
    # 空上限等价于开区间。
    bins = parse_distance_bins_text("Bulk:11:")
    assert bins[0].high_nm is None
    for bad in ("", "E1", "E1:0:2.2:3", "E1:a:2"):
        try:
            parse_distance_bins_text(bad)
        except ValueError:
            continue
        raise AssertionError(f"非法距离分层未被拒绝: {bad!r}")


def test_cli_override_merge_logic(tmp_path) -> None:
    """显式参数必须覆盖预设值；运行控制参数始终以 CLI 为准。"""

    from eels_edge_analyzer import cli

    parser = cli._parser()
    args = parser.parse_args(
        [
            "in.dm4",
            "--fit-min-ev",
            "900.0",
            "--no-injection",
            "--fast",
        ]
    )
    overrides = cli._config_overrides(parser, args)
    assert overrides["fit_min_ev"] == 900.0
    assert overrides["bootstrap_resamples"] == 100
    assert overrides["injection_simulations"] == 0
    assert overrides["run_sensitivity"] is True

    # 不带显式参数时使用内置预设的拟合窗口。
    args = parser.parse_args(["in.dm4"])
    overrides = cli._config_overrides(parser, args)
    assert overrides["fit_min_ev"] == 925.0
    assert overrides["bootstrap_resamples"] == 500
    assert overrides["injection_simulations"] == 200


def test_distance_bins_parse_matches_expected_geometry() -> None:
    bins = parse_distance_bins_text("E1:0:2.2,E2:2.2:4.4")
    assert np.allclose([item.low_nm for item in bins], [0.0, 2.2])
    assert default_preset_path().is_file()

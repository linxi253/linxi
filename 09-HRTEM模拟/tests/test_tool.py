"""工具层单元测试：参数校验 / 系列列表解析 / 取向渲染 / 导出元数据 /
惰性 multislice 一致性 / 结构读取健壮性 / STEM 角度校验。

运行：python -m pytest tests/ -v（无外部数据依赖）
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hrtem_tool.params import SimParams  # noqa: E402
from hrtem_tool.render import oriented_image  # noqa: E402
from hrtem_tool.series import parse_list  # noqa: E402
from hrtem_tool.sim_core import SimResult, _cache_put, _STRUCTURE_CACHE_MAX  # noqa: E402
from tem_sim import Microscope, Structure, multislice, multislice_series  # noqa: E402
from tem_sim.multislice import build_slice_transmissions  # noqa: E402


# ----------------------------------------------------------------------
# 参数模型
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kw", [
    dict(zone=(0, 0, 0)),
    dict(aperture_mrad=-1.0),
    dict(gpts_mode="2048"),
    dict(output_sampling_a=0.01),
    dict(sampling_a=0.001),
    dict(table="kirkland"),
    dict(display_lo_pct=90.0, display_hi_pct=10.0),
    dict(display_hi_pct=101.0),
    dict(thickness_nm=0.0),
    dict(voltage_kv=0.0),
])
def test_params_validate_rejects(kw):
    with pytest.raises(ValueError):
        SimParams(**kw).validate()


def test_params_validate_accepts_defaults():
    SimParams().validate()  # 默认值必须合法


def test_params_from_dict_coercion():
    p = SimParams.from_dict({"zone": [1.0, 1.0, 0.0], "bogus_key": 1})
    assert p.zone == (1, 1, 0)  # 浮点带轴被规整为整数
    with pytest.raises(ValueError):
        SimParams.from_dict({"zone": 5})  # 标量无法拆成 3 个指数


def test_params_replace_unknown_key():
    with pytest.raises(KeyError):
        SimParams().replace(no_such_field=1)


def test_params_scope_unit_conversion():
    p = SimParams(voltage_kv=300.0, cs_mm=1.0, defocus_nm=-5.0)
    scope = p.scope()
    assert scope.cs == 1.0e7      # mm → Å
    assert scope.defocus == -50.0  # nm → Å
    assert scope.aperture_outer == 20.0


# ----------------------------------------------------------------------
# 系列列表解析
# ----------------------------------------------------------------------
def test_parse_list_basic():
    assert parse_list("2, 4，6") == [2.0, 4.0, 6.0]
    assert parse_list("-80,-60") == [-80.0, -60.0]
    assert parse_list(" 3.5 , 7 ") == [3.5, 7.0]


@pytest.mark.parametrize("text", ["", ",,", "nan", "inf", "-inf", "1e999", "2 4 6", "abc"])
def test_parse_list_rejects(text):
    with pytest.raises(ValueError):
        parse_list(text)


def test_parse_list_count_limit():
    with pytest.raises(ValueError, match="上限"):
        parse_list(",".join(["1"] * 401))


# ----------------------------------------------------------------------
# 取向渲染
# ----------------------------------------------------------------------
def test_oriented_image_shape_and_mirror():
    rng = np.random.default_rng(7)
    arr = rng.uniform(0, 1, size=(64, 64))
    out = oriented_image(arr, 0.1, angle_deg=0.0, mirror=False,
                         output_sampling_a=0.1, output_shape=(64, 64))
    assert out.shape == (64, 64)
    # 镜像 + 中心裁剪可交换：镜像输出的最大值位置应关于中轴对称
    out_m = oriented_image(arr, 0.1, angle_deg=0.0, mirror=True,
                           output_sampling_a=0.1, output_shape=(64, 64))
    y0, x0 = np.unravel_index(np.argmax(out), out.shape)
    y1, x1 = np.unravel_index(np.argmax(out_m), out_m.shape)
    assert (y0, x0) == (y1, out.shape[1] - 1 - x1)


def test_oriented_image_constant_rotation_invariant():
    arr = np.full((48, 72), 0.5)
    out = oriented_image(arr, 0.1, angle_deg=37.5, mirror=False,
                         output_sampling_a=0.1, output_shape=(32, 32))
    assert out.shape == (32, 32)
    assert np.allclose(out, 0.5, atol=1e-6)


@pytest.mark.parametrize("outsam", [0.01, -0.1, float("nan"), float("inf")])
def test_oriented_image_sampling_guard(outsam):
    arr = np.zeros((16, 16))
    with pytest.raises(ValueError):
        oriented_image(arr, 0.1, angle_deg=0.0, mirror=False,
                       output_sampling_a=outsam, output_shape=(16, 16))


def test_oriented_image_shift_bounds():
    arr = np.zeros((16, 16))
    with pytest.raises(ValueError, match="越界"):
        oriented_image(arr, 0.1, angle_deg=0.0, mirror=False,
                       output_sampling_a=0.1, output_shape=(16, 16),
                       shift_a=(50.0, 0.0))


# ----------------------------------------------------------------------
# 惰性 multislice：与物化路径逐位一致
# ----------------------------------------------------------------------
def _random_cell_structure(seed=42, n=15):
    rng = np.random.default_rng(seed)
    cell = np.diag([8.0, 8.0, 8.0])
    pos = rng.uniform(0, 8, size=(n, 3))
    return Structure(["Au", "Fe", "O"] * (n // 3), pos, cell=cell)


def test_multislice_lazy_bit_exact():
    st = _random_cell_structure()
    scope = Microscope(voltage_kv=300.0)
    w_lazy = multislice(st, scope, sampling=0.2, slice_thickness=2.0, verbose=False)
    trans, (sx, sy), ext = build_slice_transmissions(
        st, scope, sampling=0.2, slice_thickness=2.0
    )
    w_mat = multislice(st, scope, sampling=0.2, slice_thickness=2.0,
                       transmissions=trans, verbose=False)
    assert np.array_equal(w_lazy.array, w_mat.array)
    assert w_lazy.sampling == w_mat.sampling == (sx, sy)
    assert w_lazy.extent == w_mat.extent == (ext[0], ext[1])


def test_multislice_series_lazy_matches_single():
    """厚度序列在 t=全厚时的捕获波应与单次 multislice 逐位一致。"""
    st = _random_cell_structure(seed=3)
    scope = Microscope(voltage_kv=300.0)
    lz = 8.0  # cell z 向长度
    w_single = multislice(st, scope, sampling=0.2, slice_thickness=2.0, verbose=False)
    series = multislice_series(st, scope, thicknesses=[lz],
                               sampling=0.2, slice_thickness=2.0)
    assert len(series) == 1
    t_a, w_series = series[0]
    assert t_a == pytest.approx(lz)
    assert np.array_equal(w_single.array, w_series.array)


# ----------------------------------------------------------------------
# 导出元数据
# ----------------------------------------------------------------------
def _fake_result():
    p = SimParams(zone=(1, 1, 0), table="gauss3",
                  display_lo_pct=1.0, display_hi_pct=99.0)
    img = np.full((16, 16), 0.5)
    return SimResult(
        params=p, raw=img, oriented=img, exit_wave=None,
        thickness_actual_nm=1.234, sampling=0.1, out_sampling=0.05,
        n_atoms=8, formula="Fe3O4", zone_periods=dict(lx=1.0, ly=1.0, lz=1.0),
    )


def test_tiff_description_fields():
    from hrtem_tool.export import tiff_description

    res = _fake_result()
    desc = tiff_description(res.params, res)
    for token in ("Fe3O4", "out_sampling=0.05", "table=gauss3",
                  "contrast=1-99 pct (PNG)", "norm=0.05-99.95 pct (this TIFF)",
                  "engine=tem_sim"):
        assert token in desc, f"TIFF 描述串缺少 {token!r}: {desc}"


def test_export_all_roundtrip(tmp_path):
    from hrtem_tool.export import export_all

    res = _fake_result()
    paths = export_all(res, tmp_path, "case",
                       {"tiff": True, "png": True, "npy": True, "json": True})
    assert set(paths) == {"tiff", "png", "npy", "json"}
    assert all(Path(p).exists() for p in paths.values())

    from PIL import Image

    with Image.open(paths["tiff"]) as im:
        assert im.tag_v2.get(270, "").startswith("Fe3O4")  # ImageDescription
        assert im.size == (16, 16)
    data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert data["meta"]["table"] == "gauss3"
    assert data["meta"]["engine_version"]
    assert data["meta"]["tool_version"]
    assert data["result"]["formula"] == "Fe3O4"
    arr = np.load(paths["npy"])
    assert arr.dtype == np.float32 and arr.shape == (16, 16)


@pytest.mark.skipif(os.name == "nt", reason="Windows 目录只读属性被系统忽略，无法触发")
def test_export_all_permission_error(tmp_path):
    import os as _os

    from hrtem_tool.export import export_all

    out = tmp_path / "ro"
    out.mkdir()
    out.chmod(0o500)  # 去掉写权限（POSIX）
    try:
        with pytest.raises(PermissionError):
            export_all(_fake_result(), out, "x", {"png": True})
    finally:
        out.chmod(0o700)  # 恢复，便于清理


# ----------------------------------------------------------------------
# 结构读取健壮性
# ----------------------------------------------------------------------
def test_read_xyz_truncation_warning(tmp_path):
    from tem_sim.structure import read_xyz

    p = tmp_path / "bad.xyz"
    p.write_text("2\ncomment\nFe 0 0 0\nFe 1 1 1\nFe 2 2 2\n", encoding="ascii")
    with pytest.warns(UserWarning, match="已截取"):
        st = read_xyz(p)
    assert len(st) == 2


def test_read_xyz_empty_file(tmp_path):
    from tem_sim.structure import read_xyz

    p = tmp_path / "empty.xyz"
    p.write_text("", encoding="ascii")
    with pytest.raises(ValueError, match="空文件"):
        read_xyz(p)


def test_read_xyz_bad_header(tmp_path):
    from tem_sim.structure import read_xyz

    p = tmp_path / "bad_header.xyz"
    p.write_text("abc\ncomment\nFe 0 0 0\n", encoding="ascii")
    with pytest.raises(ValueError, match="原子数"):
        read_xyz(p)


def test_read_pdb_missing_element_column_warning(tmp_path):
    from tem_sim.structure import read_pdb

    # 按 PDB 固定列构造：原子名 13-16 列为 "1HG "（氢），77-78 元素列缺失
    line = (
        "HETATM"      # 1-6
        "    1"       # 7-11 serial
        " "           # 12
        "1HG "        # 13-16 atom name
        " "           # 17
        "LIG"         # 18-20 resname
        " "           # 21
        "A"           # 22 chain
        "   1"        # 23-26 resseq
        "    "        # 27-30
        "   1.000"    # 31-38 x
        "   2.000"    # 39-46 y
        "   3.000"    # 47-54 z
        "  1.00"      # 55-60 occupancy
        "  0.00"      # 61-66 tempFactor
        "          "  # 67-76（元素列 77-78 缺失）
    )
    p = tmp_path / "noelem.pdb"
    p.write_text(line + "\n", encoding="ascii")
    with pytest.warns(UserWarning, match="元素列"):
        st = read_pdb(p)
    # 已知启发式风险：无元素列时 1HG 被解析为 Hg（汞）——这正是告警要提醒的
    assert st.symbols == ["Hg"]
    assert np.allclose(st.positions, [[1.0, 2.0, 3.0]])


# ----------------------------------------------------------------------
# 结构缓存淘汰
# ----------------------------------------------------------------------
def test_structure_cache_eviction():
    from hrtem_tool import sim_core

    for i in range(_STRUCTURE_CACHE_MAX + 5):
        _cache_put((f"path{i}", float(i)), None)
    assert len(sim_core._structure_cache) <= _STRUCTURE_CACHE_MAX
    # 最旧的键被淘汰
    assert ("path0", 0.0) not in sim_core._structure_cache
    assert ("path4", 4.0) not in sim_core._structure_cache
    assert ("path5", 5.0) in sim_core._structure_cache


# ----------------------------------------------------------------------
# STEM 角度 vs 网格奈奎斯特
# ----------------------------------------------------------------------
def test_stem_adf_probe_beyond_nyquist():
    from tem_sim.imaging import stem_adf

    st = Structure(["Au"], np.array([[2.0, 2.0, 2.0]]), cell=np.diag([4.0, 4.0, 4.0]))
    with pytest.raises(ValueError, match="奈奎斯特"):
        stem_adf(st, Microscope(voltage_kv=300.0), probe_semiangle=200.0,
                 sampling=0.5, scan_pts=2, verbose=False)


def test_stem_adf_detector_beyond_nyquist_warns():
    from tem_sim.imaging import stem_adf

    st = Structure(["Au"], np.array([[2.0, 2.0, 2.0]]), cell=np.diag([4.0, 4.0, 4.0]))
    # 盒子 4 Å → 网格 32 px（_grid_size 下限）→ 实际采样 0.125 Å/px
    # → 奈奎斯特 4 Å⁻¹ ≈ 78.7 mrad @300 kV；
    # 探针 20 mrad 合法，探测器外角 200 mrad 超出 → 告警
    with pytest.warns(UserWarning, match="奈奎斯特"):
        image, step_x, step_y = stem_adf(
            st, Microscope(voltage_kv=300.0), probe_semiangle=20.0,
            inner_angle=60.0, outer_angle=200.0,
            sampling=0.4, scan_pts=2, verbose=False,
        )
    assert image.shape == (2, 2)
    assert step_x == pytest.approx(4.0 / 2)  # lx / scan_pts
    assert step_y == pytest.approx(4.0 / 2)


# ----------------------------------------------------------------------
# 引擎：padding 警告
# ----------------------------------------------------------------------
def test_small_padding_warns_for_aperiodic():
    st = Structure(["Au"], np.array([[0.0, 0.0, 0.0]]), cell=None)
    scope = Microscope(voltage_kv=300.0)
    with pytest.warns(UserWarning, match="真空边距"):
        multislice(st, scope, sampling=0.5, slice_thickness=2.0,
                   padding=2.0, verbose=False)

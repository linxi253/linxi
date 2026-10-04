"""针对 2026-09 代码审查修复项的回归测试。

覆盖:
  - point_size 僵尸状态移除后, "手动加点 → 自动检测 → 追加" 不再崩溃 (P0)
  - 检测阈值百分位语义与非法输入拒绝
  - _clear_analysis_results 递增任务代际 (分析竞态闭环)
  - benchmark.match_points 竞争匹配跳过而非 NameError
  - prepare_dataset 按原图分组划分 (防 train/test 泄漏)
  - augment small_rot 使用 (N-1)/2 像素中心约定
  - detector 参数校验先于重型导入
  - ppa_stats 可选列按表头名解析 (列序无关)
  - utf-8-sig (带 BOM) CSV 双向兼容
  - semi_auto_label 背景自适应阈值对暗原子的公平性
  - 自动检测 worker 强制主线程快照 use_preprocessed (工单75, Tk 非线程安全)
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import sys  # noqa: E402
# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



# ---------------------------------------------------------------- ppa.py --

class ThresholdParsingTests(unittest.TestCase):
    def test_valid_percentile_and_empty(self):
        import ppa
        self.assertAlmostEqual(ppa.parse_detection_threshold("0.8"), 0.8)
        self.assertIsNone(ppa.parse_detection_threshold("  "))
        self.assertIsNone(ppa.parse_detection_threshold(""))

    def test_absolute_and_degenerate_values_rejected(self):
        import ppa
        # 图像已归一化到 [0,1]: >=1 的"绝对阈值"是死功能, 0/负数会放过全部像素
        for bad in ("1.0", "1.5", "0", "0.0", "-0.2", "abc", "nan", "inf"):
            with self.assertRaises(ValueError, msg=bad):
                ppa.parse_detection_threshold(bad)


class ClearAnalysisBumpsGenerationTests(unittest.TestCase):
    def test_clear_invalidates_in_flight_jobs(self):
        import ppa
        app = ppa.AtomMarkerApp.__new__(ppa.AtomMarkerApp)
        app._job_generation = 7
        app._clear_analysis_results()
        self.assertEqual(app._job_generation, 8)
        self.assertIsNone(app.strain_xx)
        self.assertEqual(app.strain_grid, {})


class DetectAppendAfterEditTests(unittest.TestCase):
    """P0 回归: 手动编辑点表后追加检测结果不得崩溃。"""

    def _make_app(self):
        import tkinter as tk
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        root.withdraw()
        import ppa
        app = ppa.AtomMarkerApp(root)
        app.image = np.zeros((50, 50), dtype=np.float64)
        app.image_path = "synthetic.tif"
        return root, app

    def test_append_after_manual_add(self):
        root, app = self._make_app()
        try:
            app.points = [(5.0, 5.0)]  # 手动加过 1 个点 (旧代码会使 point_size 失配)
            with mock.patch.object(__import__('ppa').messagebox,
                                   "askyesnocancel", return_value=True) as ask:
                app._finish_auto_detect([(10.0, 10.0), (20.0, 20.0)])
            self.assertEqual(ask.call_count, 1)
            self.assertEqual(len(app.points), 3)
            self.assertEqual(app.points[-1], (20.0, 20.0))
        finally:
            root.destroy()

    def test_cancel_discards_results(self):
        root, app = self._make_app()
        try:
            app.points = [(5.0, 5.0)]
            with mock.patch.object(__import__('ppa').messagebox,
                                   "askyesnocancel", return_value=None):
                app._finish_auto_detect([(10.0, 10.0)])
            self.assertEqual(app.points, [(5.0, 5.0)])
        finally:
            root.destroy()


class DetectWorkerTkSnapshotTests(unittest.TestCase):
    """工单75: worker 线程内不得读 tk 变量, use_preprocessed 必须为主线程快照。

    本组用例锁定修复后的完整形态：_detect_peaks 在主线程完成快照、
    worker 以普通数值形参接收（此前两者都直接读 tk 变量）。
    """

    def _make_app(self):
        import tkinter as tk
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        root.withdraw()
        import ppa
        app = ppa.AtomMarkerApp(root)
        app.image = np.zeros((50, 50), dtype=np.float64)
        app.image_path = "synthetic.tif"
        return root, app

    def test_use_preprocessed_is_required_positional(self):
        root, app = self._make_app()
        try:
            # 快照参数必填: 缺参在调用层即 TypeError, 不再可能经 None 分支
            # 在 worker 线程现场读 tk 变量 (Tk 非线程安全)
            with self.assertRaises(TypeError):
                app._detect_peaks_worker(4, 0.0, 5, None, True)
        finally:
            root.destroy()

    def test_worker_uses_snapshot_not_tk_var(self):
        root, app = self._make_app()
        try:
            # 快照 False → 用原图 (全零图无峰)
            app.processed_image = None
            pts, stats = app._detect_peaks_worker(4, 0.0, 5, None, True, False)
            self.assertEqual(pts, [])
            self.assertIsNone(stats)
            # 快照 True → 用预处理图 (单点亮斑), 与 use_preprocessed tk 变量无关
            proc = np.zeros((50, 50), dtype=np.float64)
            proc[25, 25] = 1.0
            app.processed_image = proc
            pts2, _ = app._detect_peaks_worker(4, 0.0, 5, None, True, True)
            self.assertEqual(pts2, [(25.0, 25.0)])
        finally:
            root.destroy()


# ------------------------------------------------------- ppa_stats.py ----

class StrainCsvColumnResolutionTests(unittest.TestCase):
    def test_optional_columns_resolved_by_header_name(self):
        import ppa_stats
        # 边缘列故意放在 GL 列之前: 旧的固定下标解析会把 GL 数值读进 edge
        header = ("三角形编号,位置x,位置y,ε_xx,ε_yy,ε_xy (张量剪应变),"
                  "ε_eq (von Mises),极分解旋转 θ (rad),边缘三角形,"
                  "ε_xx (GL),ε_yy (GL),ε_xy (GL),ε_eq (GL von Mises)")
        rows = [
            "1,10.0,20.0,0.010,0.020,0.005,0.030,0.001,1,0.011,0.021,0.006,0.031",
            "2,30.0,40.0,-0.010,-0.020,-0.005,0.030,-0.001,0,-0.011,-0.021,-0.006,0.031",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strain.csv"
            path.write_text("\n".join([header] + rows), encoding="utf-8-sig")
            result = ppa_stats.load_strain_csv(path)
        self.assertEqual(int(result['n']), 2)
        np.testing.assert_array_equal(result['edge_mask'], [True, False])
        np.testing.assert_allclose(result['gl_xx'], [0.011, -0.011])
        np.testing.assert_allclose(result['gl_eq'], [0.031, 0.031])

    def test_bom_displacement_csv_roundtrip(self):
        import ppa_stats
        header = "编号,实际x（物理）,实际y（物理）,参考x（物理）,参考y（物理）,位移dx（物理）,位移dy（物理）,位移幅值"
        row = "1,10.0,-5.0,9.5,-4.5,0.5,-0.5,0.7071"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "disp.csv"
            path.write_text("\n".join([header, row]), encoding="utf-8-sig")
            meta = Path(tmp) / "disp.metadata.json"
            meta.write_text(json.dumps({
                "coordinate_convention":
                    "physical-cartesian-x-right-y-up; image-display-y-down",
                "algorithm_id": "peak-pairs-local-all-points-v2",
            }), encoding="utf-8")
            result = ppa_stats.load_displacement_csv(path)
        # 物理 y=-5 翻转为显示 y=+5; dx 不变
        self.assertAlmostEqual(result['actual_pos'][0, 1], 5.0)
        self.assertAlmostEqual(result['displacements'][0, 0], 0.5)


# ------------------------------------------------------ atom_detector ----

class MatchPointsTests(unittest.TestCase):
    def test_competing_predictions_skip_instead_of_crashing(self):
        from atom_detector.infer.benchmark import match_points
        # 两个预测都在同一 GT 的 match_dist 内 → 第二个必须作为 FP 跳过,
        # 而不是触发旧代码的 NameError
        preds = [(10.0, 10.0), (13.0, 10.0)]
        gts = [(10.0 / 100, 10.0 / 100)]  # 归一化 → 像素 (10, 10)
        tp, fp, fn, errors = match_points(preds, gts, (100, 100), match_dist=5.0)
        self.assertEqual(tp, 1)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 0)
        self.assertEqual(len(errors), 1)

    def test_two_preds_two_gts_partial_overlap(self):
        from atom_detector.infer.benchmark import match_points
        preds = [(10.0, 10.0), (30.0, 10.0), (60.0, 60.0)]
        gts = [(10.0 / 100, 10.0 / 100), (31.0 / 100, 10.0 / 100)]
        tp, fp, fn, errors = match_points(preds, gts, (100, 100), match_dist=3.0)
        self.assertEqual(tp, 2)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 0)


class DetectorParamValidationTests(unittest.TestCase):
    def test_invalid_conf_rejected_before_heavy_imports(self):
        from atom_detector.infer.detector import AtomDetector
        # conf 越界必须抛 ValueError (校验先于 atom_center 导入),
        # 而不是 ModuleNotFoundError 或静默接受
        with self.assertRaises(ValueError):
            AtomDetector("whatever.pt", conf=1.5)
        with self.assertRaises(ValueError):
            AtomDetector("whatever.pt", iou=0.0)
        with self.assertRaises(ValueError):
            AtomDetector("whatever.pt", imgsz=16)


class AugmentRotationCenterTests(unittest.TestCase):
    def test_small_rot_uses_pixel_center_convention(self):
        from atom_detector.dataset import augment
        w = h = 61
        img = np.zeros((h, w))
        angle = 5.0
        with mock.patch.object(augment.np.random, "uniform", return_value=angle):
            _, labels = augment.augment_image_and_labels(
                img, [[0, 0.8, 0.3, 0.1, 0.1]], "small_rot")
        # scipy.ndimage.rotate 绕 ((N-1)/2, (N-1)/2) 旋转; 标签变换必须同中心。
        # 独立按该约定解析析期望值 (若有人回退到 w/2 中心, 此断言失败)。
        cx0, cy0 = (w - 1) / 2.0, (h - 1) / 2.0
        px, py = 0.8 * w, 0.3 * h
        rad = np.radians(-angle)
        expected_x = ((px - cx0) * np.cos(rad) - (py - cy0) * np.sin(rad) + cx0) / w
        expected_y = ((px - cx0) * np.sin(rad) + (py - cy0) * np.cos(rad) + cy0) / h
        self.assertAlmostEqual(labels[0][1], expected_x, places=9)
        self.assertAlmostEqual(labels[0][2], expected_y, places=9)

    def test_augment_dataset_skips_existing_augmented_copies(self):
        from atom_detector.dataset import augment
        with tempfile.TemporaryDirectory() as tmp:
            img_dir = Path(tmp) / "images"
            lbl_dir = Path(tmp) / "labels"
            img_dir.mkdir()
            lbl_dir.mkdir()
            (img_dir / "a.png").write_bytes(b"fake")
            (lbl_dir / "a.txt").write_text("0 0.5 0.5 0.1 0.1")
            (img_dir / "a_aug_flip_h.png").write_bytes(b"fake")
            (lbl_dir / "a_aug_flip_h.txt").write_text("0 0.5 0.5 0.1 0.1")
            with mock.patch.object(augment, "load_image", return_value=np.zeros((32, 32))), \
                    mock.patch.object(augment, "save_image") as save_mock:
                augment.augment_dataset(img_dir, lbl_dir, factor=1, seed=1)
            saved_names = [str(call.args[1]) for call in save_mock.call_args_list]
            # 只允许从原图 a 生成 a_aug_<type>; 不得出现二次增强 a_aug_flip_h_aug_*
            self.assertTrue(saved_names, "应至少生成一个增强副本")
            for name in saved_names:
                self.assertNotIn("aug_flip_h_aug", name,
                                 f"增强副本被再次增强: {name}")
                self.assertIn("a_aug", name)


class PrepareDatasetGroupSplitTests(unittest.TestCase):
    def test_augmented_variants_stay_in_one_split(self):
        from atom_detector.dataset.prepare_dataset import prepare_dataset
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            img_dir = tmp / "raw_images"
            lbl_dir = tmp / "labels"
            out_dir = tmp / "dataset"
            img_dir.mkdir()
            lbl_dir.mkdir()
            # 4 张原图, 其中 img1 有两个增强变体
            stems = ["img1", "img1_aug_flip_h", "img1_aug_noise", "img2", "img3", "img4"]
            for stem in stems:
                (img_dir / f"{stem}.png").write_bytes(b"fake")
                (lbl_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1")
            prepare_dataset(img_dir, lbl_dir, out_dir,
                            train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=7)
            splits = {}
            for split in ("train", "val", "test"):
                names = {p.stem for p in (out_dir / "labels" / split).glob("*.txt")}
                for name in names:
                    splits[name] = split
            self.assertEqual(splits["img1"], splits["img1_aug_flip_h"])
            self.assertEqual(splits["img1"], splits["img1_aug_noise"])
            # 每个子集都非空 (6 组 → 3/1/2)
            self.assertTrue((out_dir / "labels" / "train").glob)  # 目录存在
            self.assertGreater(len(list((out_dir / "labels" / "train").glob("*.txt"))), 0)
            self.assertGreater(len(list((out_dir / "labels" / "test").glob("*.txt"))), 0)

    def test_ratio_sum_enforced(self):
        from atom_detector.dataset.prepare_dataset import prepare_dataset
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                prepare_dataset(Path(tmp), Path(tmp), Path(tmp) / "out",
                                train_ratio=0.8, val_ratio=0.1, test_ratio=0.2)


class BackgroundThresholdTests(unittest.TestCase):
    @staticmethod
    def _lattice_image(n_bright, seed=3):
        """同一背景噪声尺度下、不同亮原子数量的合成图 (各含 1 个暗原子)。"""
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(seed)
        img = np.full((320, 320), 0.2)
        # 亮原子网格 (跳过暗原子邻域, 避免峰重叠)
        dim_y, dim_x = 162, 162
        step = max(6, int(320 / max(1, int(np.sqrt(n_bright)))))
        count = 0
        for y in range(10, 315, step):
            for x in range(10, 315, step):
                if count >= n_bright:
                    break
                if abs(y - dim_y) < 4 and abs(x - dim_x) < 4:
                    continue
                yy, xx = np.mgrid[y - 2:y + 3, x - 2:x + 3]
                img[yy, xx] = np.clip(
                    img[yy, xx] + 0.8 * np.exp(-(((yy - y) ** 2 + (xx - x) ** 2) / 2.0)), 0, 1)
                count += 1
        # 暗原子: 幅度 0.25 (模拟低 Z 原子列)
        yy, xx = np.mgrid[dim_y - 2:dim_y + 3, dim_x - 2:dim_x + 3]
        img[yy, xx] += 0.25 * np.exp(-(((yy - dim_y) ** 2 + (xx - dim_x) ** 2) / 2.0))
        img = np.clip(img + rng.normal(0, 0.005, img.shape), 0, 1)
        return gaussian_filter(img, 0.5), (dim_y, dim_x)

    def test_threshold_tracks_noise_not_occupancy(self):
        """背景阈值应随噪声尺度变化, 而不是与原子占空比耦合。"""
        from atom_detector.annotate.semi_auto_label import background_threshold
        sparse, _ = self._lattice_image(n_bright=64)
        dense, _ = self._lattice_image(n_bright=900)
        new_sparse = background_threshold(sparse)
        new_dense = background_threshold(dense)
        # 同一噪声水平 → 背景阈值近似不变 (对占空比不敏感)
        self.assertLess(abs(new_sparse - new_dense), 0.05,
                        f"背景阈值不应随原子占空比漂移: {new_sparse:.3f} vs {new_dense:.3f}")

    def test_threshold_formula_matches_noise_floor(self):
        """纯背景+高斯噪声下, 阈值 = 中位数 + 6·1.4826·MAD, 随 σ 线性放大。"""
        from atom_detector.annotate.semi_auto_label import background_threshold
        rng = np.random.default_rng(11)
        bg_low = 0.2 + rng.normal(0, 0.005, (200, 200))
        bg_high = 0.2 + rng.normal(0, 0.020, (200, 200))
        t_low = background_threshold(bg_low)
        t_high = background_threshold(bg_high)
        self.assertAlmostEqual(t_low, np.median(bg_low) + 6.0 * 1.4826 *
                               np.median(np.abs(bg_low - np.median(bg_low))), places=6)
        # σ 从 0.005 → 0.020 (4×): 抬升 = 6·1.4826·ΔMAD = 6·Δσ ≈ 0.090
        expected_rise = 6.0 * (0.020 - 0.005)
        self.assertGreater(t_high - t_low, expected_rise * 0.7)
        self.assertLess(t_high - t_low, expected_rise * 1.3)

    def test_dim_atom_detected_in_dense_field(self):
        from atom_detector.annotate.semi_auto_label import detect_atoms_traditional
        img_filt, (dim_y, dim_x) = self._lattice_image(n_bright=400)
        points = detect_atoms_traditional(img_filt, sigma=0.0, min_dist=3, window=5)
        dist = [np.hypot(x - dim_x, y - dim_y) for x, y in points]
        self.assertTrue(min(dist) < 1.0 if dist else False,
                        f"暗原子 (低 Z 列) 必须被检出; 最近检测距离 {min(dist) if dist else None}")


class _Widget:
    """最小 Tk 控件替身（只实现 status.config）。"""

    def config(self, **kwargs):
        return None

    configure = config


class _InlineThread:
    """同步执行 target 的 Thread 替身（用于捕获 worker 参数）。"""

    def __init__(self, *, target, **kwargs):
        self.target = target

    def start(self):
        self.target()


class LatticeConditionThresholdChainTests(unittest.TestCase):
    """晶格条件数阈值必须贯穿整条调用链（回归 2026-10-03）。

    参考晶格校验、参考区基矢精化、晶格索引分配、局部 PPA 与参考加载必须使用
    **同一个**用户阈值。此前只有参考校验处可传阈值，其余仍回落硬编码 30，
    用户放宽后会在后续步骤被静默挡下。
    """

    def setUp(self):
        import ppa as ppa_module
        from ppa_core.strain import LATTICE_CONDITION_GUIDANCE
        self.ppa = ppa_module
        self.guidance = LATTICE_CONDITION_GUIDANCE
        indices = np.array([(i, j) for i in range(3) for j in range(3)])
        self.indices = indices
        # 条件数 40 的基矢：默认 30 拒绝，放宽到 50 通过
        self.ref = indices @ np.diag([40.0, 1.0])
        self.cond = 40.0
        self.a_vec = np.array([40.0, 0.0])
        self.b_vec = np.array([0.0, 1.0])

    def test_default_stays_30(self):
        """默认阈值必须保持历史行为（general=30）。"""
        from ppa_core.strain import validate_reference_lattice
        self.assertEqual(self.guidance["general"], 30.0)
        # cond=40 > 30 → 默认必须拒绝（这是历史行为，不能被"合并"放宽）
        with self.assertRaises(Exception):
            validate_reference_lattice([40, 0], [0, 1])
        # 合法基矢在默认阈值下正常通过，返回值是条件数
        self.assertAlmostEqual(validate_reference_lattice([10, 0], [0, 10]), 1.0)

    def test_reference_and_local_share_custom_threshold(self):
        """cond=40 的基矢：阈值 50 时参考校验与局部 PPA 都必须通过。"""
        from ppa_core.strain import (compute_local_peak_pair_strain,
                                     validate_reference_lattice)
        self.assertAlmostEqual(
            validate_reference_lattice([40, 0], [0, 1], max_condition=50.0), self.cond)
        result = compute_local_peak_pair_strain(
            self.indices, self.ref, self.ref, max_condition=50.0)
        self.assertTrue(np.any(result.quality_mask),
                        "放宽阈值后局部 PPA 不应把全部站点判为无效")

    def test_assign_unique_lattice_indices_accepts_threshold(self):
        """索引分配此前隐藏使用默认 30，放宽后必须能通过。"""
        points = self.ref.copy()
        assignment = self.ppa.assign_unique_lattice_indices(
            points, np.zeros(2), self.a_vec, self.b_vec, max_condition=50.0)
        self.assertEqual(len(assignment.lattice_indices), len(points))

    def test_refine_region_accepts_threshold(self):
        """参考区精化内部的索引分配也必须使用同一阈值。"""
        points = self.ref.copy()
        mask = np.ones(len(points), dtype=bool)
        out = self.ppa.refine_lattice_basis_in_region(
            points, np.zeros(2), self.a_vec, self.b_vec, mask,
            min_atoms=3, max_condition=50.0)
        self.assertEqual(len(out), 5)

    def test_estimate_chains_accepts_threshold(self):
        """双原子列估计也必须接受同一阈值。"""
        a_points = np.stack([np.array([40.0 * k, 0.0]) for k in range(4)])
        b_points = np.stack([np.array([0.0, 1.0 * k]) for k in range(4)])
        a_fit, b_fit = self.ppa.estimate_reference_vectors_from_chains(
            a_points, b_points, min_points=3, max_condition=50.0)
        self.assertIsNotNone(a_fit.vector)
        self.assertIsNotNone(b_fit.vector)

    def test_threshold_validation_rejects_non_finite_and_non_positive(self):
        """NaN/inf/0/负值必须在任何计算前明确拒绝。"""
        from ppa_core.strain import (compute_local_peak_pair_strain,
                                     validate_max_condition,
                                     validate_reference_lattice)
        for bad in (float("nan"), float("inf"), float("-inf"), 0.0, -1.0, "x", None):
            with self.subTest(bad=bad):
                with self.assertRaises(Exception):
                    validate_max_condition(bad)
                with self.assertRaises(Exception):
                    validate_reference_lattice([40, 0], [0, 1], max_condition=bad)
                with self.assertRaises(Exception):
                    compute_local_peak_pair_strain(
                        self.indices, self.ref, self.ref, max_condition=bad)

    def _bare_app(self, threshold):
        """构造只含 run_ppa_analysis 所需字段的实例（不跑 __init__/Tk）。"""
        app = self.ppa.AtomMarkerApp.__new__(self.ppa.AtomMarkerApp)
        app.image = np.zeros((128, 128), dtype=np.float32)
        app.points = [tuple(row) for row in self.ref]
        app.reference_vecs = (self.a_vec, self.b_vec)
        app.ref_origin = np.array([0.0, 0.0])
        app.lattice_max_condition = threshold
        app.von_mises_coeff = 4.0 / 9.0
        app.analysis_method = "peak_pairs"
        app._job_generation = 0
        app.status = _Widget()
        app.root = type("Root", (), {"update_idletasks": staticmethod(lambda: None)})()
        app._atoms_in_ref_region_mask = lambda: None
        app._clear_analysis_results = lambda: None
        app._ensure_worker_polling = lambda: None
        return app

    def test_gui_worker_receives_snapshot_threshold(self):
        """run_ppa_analysis 派发给 worker 的阈值必须等于配置值（不回落默认）。"""
        app = self._bare_app(50.0)
        dispatched = []
        app._ppa_analysis_worker = lambda **kwargs: dispatched.append(kwargs)
        with mock.patch.object(self.ppa.threading, "Thread", _InlineThread), \
                mock.patch.object(self.ppa.messagebox, "showerror") as err:
            app.run_ppa_analysis()
        self.assertEqual(len(dispatched), 1, f"worker 未派发: {err.call_args_list}")
        self.assertEqual(dispatched[0]["lattice_max_condition"], 50.0)

    def test_gui_analysis_rejects_invalid_threshold_before_any_work(self):
        """非法阈值必须在派发 worker 之前报错（不得进入计算）。"""
        app = self._bare_app(30.0)
        dispatched = []
        app._ppa_analysis_worker = lambda **kwargs: dispatched.append(kwargs)
        for bad in (float("nan"), float("inf"), 0.0, -1.0):
            with self.subTest(bad=bad):
                dispatched.clear()
                app.lattice_max_condition = bad
                with mock.patch.object(self.ppa.messagebox, "showerror") as err:
                    app.run_ppa_analysis()
                self.assertEqual(dispatched, [], f"非法阈值 {bad} 仍派发了 worker")
                self.assertTrue(err.called, f"非法阈值 {bad} 未提示")


if __name__ == "__main__":
    unittest.main()

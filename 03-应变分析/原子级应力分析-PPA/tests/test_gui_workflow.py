from __future__ import annotations

import time
import tkinter as tk
import unittest
from unittest import mock

import numpy as np

import ppa


class FullPointPpaWorkflowTests(unittest.TestCase):
    def test_index_collision_keeps_every_confirmed_atom_and_shows_one_report(self):
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        root.withdraw()
        try:
            app = ppa.AtomMarkerApp(root)
            app.image = np.zeros((90, 90), dtype=np.float64)
            app.image_path = "synthetic.tif"
            app.points = [
                (10.0 + 10 * i, 10.0 + 10 * j)
                for i in range(5)
                for j in range(5)
            ] + [(10.4, 10.1)]
            app.reference_vecs = [np.array([10.0, 0.0]), np.array([0.0, 10.0])]
            app.ref_origin = np.array([10.0, 10.0])

            with mock.patch.object(ppa.messagebox, "showwarning") as warning, \
                    mock.patch.object(ppa.messagebox, "showinfo") as info, \
                    mock.patch.object(ppa.messagebox, "showerror") as error:
                app.run_ppa_analysis()
                for _ in range(300):
                    root.update()
                    if info.call_count + warning.call_count:
                        break
                    time.sleep(0.01)

            self.assertEqual(len(app.points), 26)
            self.assertEqual(len(app.matched_actual), 26)
            self.assertEqual(len(app.ideal_grid), 26)
            self.assertEqual(len(app.displacements), 26)
            self.assertEqual(len(app.lattice_indices), 26)
            self.assertEqual(len({tuple(row) for row in app.lattice_indices}), 26)
            np.testing.assert_array_equal(app.analysis_point_indices, np.arange(26))
            self.assertFalse(hasattr(app, "duplicate_index_points"))
            self.assertEqual(error.call_count, 0)
            self.assertEqual(info.call_count + warning.call_count, 1)
            self.assertIsNotNone(app.strain_xx)
        finally:
            root.destroy()

    def _make_app_with_strain_result(self, root):
        """构建带 5×5 完整晶格分析结果的应用 (不弹出任何对话框)。"""
        app = ppa.AtomMarkerApp(root)
        app.image = np.zeros((90, 90), dtype=np.float64)
        app.image_path = "synthetic.tif"
        app.points = [(10.0 + 10 * i, 10.0 + 10 * j) for i in range(5) for j in range(5)]
        app.reference_vecs = [np.array([10.0, 0.0]), np.array([0.0, 10.0])]
        app.ref_origin = np.array([10.0, 10.0])
        with mock.patch.object(ppa.messagebox, "showwarning"), \
                mock.patch.object(ppa.messagebox, "showinfo"), \
                mock.patch.object(ppa.messagebox, "showerror"):
            app.run_ppa_analysis()
            for _ in range(300):
                root.update()
                if app.strain_grid:
                    break
                time.sleep(0.01)
        return app

    def test_strain_cloud_style_controls_apply_to_interpolated_view(self):
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        root.withdraw()
        app = None
        try:
            app = self._make_app_with_strain_result(root)
            self.assertTrue(app.strain_grid)

            # 默认样式: 分量自动配色 + 0.75 不透明度
            app._view_changed("strain_eq")
            artist = app._strain_cloud_artist
            self.assertIsNotNone(artist)
            self.assertAlmostEqual(artist.get_alpha(), 0.75)
            self.assertEqual(artist.get_cmap().name, "turbo")

            # 用户选择 viridis: 配色覆盖生效, 等效应变不再强制对称色标
            app.cloud_cmap_var.set("viridis")
            app._on_cloud_style_changed()
            self.assertEqual(app.strain_cloud_cmap, "viridis")
            self.assertEqual(app._strain_cloud_artist.get_cmap().name, "viridis")

            # 透明度滑杆: 就地更新 artist, 不需要整帧重建
            app.cloud_alpha_var.set(0.3)
            app._on_cloud_alpha_changed()
            self.assertAlmostEqual(app.strain_cloud_alpha, 0.3)
            self.assertAlmostEqual(app._strain_cloud_artist.get_alpha(), 0.3)

            # 发散分量使用发散色图时保持对称色标语义
            app._view_changed("strain_xy")
            self.assertEqual(app._strain_cloud_artist.get_cmap().name, "viridis")

            # 回到自动: 状态清空, 剪应变恢复默认 RdBu_r
            app.cloud_cmap_var.set(ppa.STRAIN_CLOUD_AUTO_CMAP_LABEL)
            app._on_cloud_style_changed()
            self.assertIsNone(app.strain_cloud_cmap)
            self.assertEqual(app._strain_cloud_artist.get_cmap().name, "RdBu_r")
        finally:
            # 取消挂起的后台轮询, 避免 destroy 后 after 回调产生 Tcl 告警
            if app is not None and app._poll_after_id is not None:
                try:
                    root.after_cancel(app._poll_after_id)
                except tk.TclError:
                    pass
            root.destroy()

    def test_point_style_controls_apply_to_displacement_views(self):
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        root.withdraw()
        app = None
        try:
            app = self._make_app_with_strain_result(root)

            # 默认: turbo 配色 + 40 pt² 点大小
            app._view_changed("colored")
            artist = app._points_scatter_artist
            self.assertIsNotNone(artist)
            self.assertEqual(artist.get_cmap().name, "turbo")
            self.assertTrue(np.all(np.asarray(artist.get_sizes()) == 40))

            # 选择自定义红蓝色带: 色带与标题同步生效
            app.point_cmap_var.set(ppa.POINT_CMAP_CHOICES[1])
            app._on_point_cmap_changed()
            self.assertEqual(app.point_cmap_key, "rb")
            self.assertEqual(app._points_scatter_artist.get_cmap().name, "ppa_red_blue")

            # 点大小滑杆: 就地更新 sizes, 不重建 artist
            artist_before = app._points_scatter_artist
            app.point_size_var.set(70)
            app._on_point_size_changed()
            self.assertEqual(app.atom_dot_size, 70)
            self.assertIs(app._points_scatter_artist, artist_before)
            self.assertTrue(np.all(np.asarray(artist_before.get_sizes()) == 70))

            # 矢量场视图同样遵循点样式 (点大小 + 红蓝色带)
            app._view_changed("vector")
            self.assertEqual(app._points_scatter_artist.get_cmap().name, "ppa_red_blue")

            # 回到默认配色
            app.point_cmap_var.set(ppa.POINT_CMAP_CHOICES[0])
            app._on_point_cmap_changed()
            self.assertEqual(app._points_scatter_artist.get_cmap().name, "turbo")
        finally:
            if app is not None and app._poll_after_id is not None:
                try:
                    root.after_cancel(app._poll_after_id)
                except tk.TclError:
                    pass
            root.destroy()


if __name__ == "__main__":
    unittest.main()

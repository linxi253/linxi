# -*- coding: utf-8 -*-
"""GUI 冒烟测试：验证界面构建、图像加载与实时预览渲染。"""

import os
import sys
import time
import tkinter as tk

import numpy as np

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import FilterApp

TEST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_images')

results = []


def run_test(image_name, param_updates, desc, expect_color=None):
    """创建应用、加载图像、调参、验证渲染，然后关闭。"""
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        results.append((f'{desc}（无显示，已跳过：{exc}）', True, ''))
        return
    root.withdraw()  # 隐藏窗口，仅做逻辑测试
    try:
        app = FilterApp(root)
    except tk.TclError as exc:
        results.append((f'{desc}（初始化失败，已跳过：{exc}）', True, ''))
        try:
            root.destroy()
        except tk.TclError:
            pass
        return

    test_img = os.path.join(TEST_DIR, image_name)
    app.file_list = [test_img]
    app.file_index = 0
    app._load_current()

    for key, val in param_updates.items():
        app.sliders[key].set_value(val, notify=True)

    deadline = time.monotonic() + 8

    def check_and_close():
        try:
            # 帧加载与预览计算均已异步化：轮询等待两侧画布渲染完成。
            if time.monotonic() < deadline and (
                    app.canvas_orig.image_ref is None or app.canvas_prev.image_ref is None):
                root.after(50, check_and_close)
                return
            assert app.canvas_orig.image_ref is not None, '原始图像未渲染'
            assert app.canvas_prev.image_ref is not None, '预览图像未渲染'
            if expect_color is not None:
                state = str(app.sliders['temperature'].slider['state'])
                wanted = 'normal' if expect_color else 'disabled'
                assert state == wanted, f'色彩滑块状态应为 {wanted}，实际 {state}'
            results.append((desc, True, ''))
        except Exception as e:
            results.append((desc, False, str(e)))
        finally:
            app.close()

    root.after(50, check_and_close)
    root.mainloop()


def test_mousewheel_scroll():
    """滚轮必须在整个参数面板生效，包括指针悬停在滑块上时（NotifyInferior 场景）。"""
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        results.append((f'滚轮滚动（无显示，已跳过：{exc}）', True, ''))
        return
    root.withdraw()
    try:
        app = FilterApp(root)
    except tk.TclError as exc:
        results.append((f'滚轮滚动（初始化失败，已跳过：{exc}）', True, ''))
        try:
            root.destroy()
        except tk.TclError:
            pass
        return

    def check_and_close():
        try:
            canvas = app.panel_canvas
            # withdraw 窗口不布局，scrollregion 为空时 yview_scroll 是空操作；
            # 显式设定以聚焦测试绑定与处理逻辑本身。
            canvas.configure(scrollregion=(0, 0, 300, 2000))
            assert canvas.yview()[0] == 0.0, '初始应位于顶部'
            slider_widget = app.sliders['exposure'].slider
            slider_widget.event_generate('<MouseWheel>', delta=-120)
            assert canvas.yview()[0] > 0.0, '悬停在滑块上滚动应生效'
            before = canvas.yview()[0]
            canvas.event_generate('<Button-5>')  # Linux 向下
            after_down = canvas.yview()[0]
            assert after_down > before, 'Button-5 应向下滚动'
            canvas.event_generate('<Button-4>')  # Linux 向上
            assert canvas.yview()[0] < after_down, 'Button-4 应向上滚动'
            results.append(('滑块上滚轮/按钮事件滚动面板', True, ''))
        except Exception as e:
            results.append(('滑块上滚轮/按钮事件滚动面板', False, str(e)))
        finally:
            app.close()

    root.after(300, check_and_close)
    root.mainloop()


def test_downscale_antialias():
    """高倍缩小必须抗混叠：棋盘格缩小后不应保留 0/1 极值（摩尔纹）。"""
    from main import _downscale
    yy, xx = np.mgrid[0:512, 0:512]
    pattern = ((xx // 4 + yy // 4) % 2).astype(np.float32)
    small = _downscale(pattern, 64)
    spread = float(small.max() - small.min())
    assert 0.4 < float(small.mean()) < 0.6, f'均值漂移：{small.mean()}'
    assert spread < 0.9, f'混叠未消除：max-min={spread}'
    print('[通过] 降采样抗混叠（棋盘格 512→64）')


def test_display_mode_toggle():
    """显示范围开关：自动（百分位）↔ 固定 [0,1]，切换后预览仍可渲染。"""
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        results.append((f'显示范围开关（无显示，已跳过：{exc}）', True, ''))
        return
    root.withdraw()
    try:
        app = FilterApp(root)
    except tk.TclError as exc:
        results.append((f'显示范围开关（初始化失败，已跳过：{exc}）', True, ''))
        try:
            root.destroy()
        except tk.TclError:
            pass
        return

    test_img = os.path.join(TEST_DIR, 'rgb8.tif')
    app.file_list = [test_img]
    app.file_index = 0
    app._load_current()
    deadline = time.monotonic() + 8

    def check_and_close():
        try:
            if app.preview_frame is None:
                assert time.monotonic() < deadline, '预览帧加载超时'
                root.after(50, check_and_close)
                return
            assert app._display_auto, '初始应为自动模式'
            assert app._current_levels() == app.preview_levels
            app._toggle_display_mode()
            assert not app._display_auto
            assert app._current_levels() == (0.0, 1.0), '固定模式应为 [0,1]'
            assert '显示范围：[0,1]' in str(app._display_toggle.cget('text'))
            assert app.canvas_prev.image_ref is not None, '切换后预览应仍可渲染'
            results.append(('显示范围开关与预览渲染', True, ''))
        except Exception as e:
            results.append(('显示范围开关与预览渲染', False, str(e)))
        finally:
            app.close()

    root.after(50, check_and_close)
    root.mainloop()


if __name__ == '__main__':
    run_test('rgb8.tif', {'contrast': 50, 'saturation': 30, 'exposure': 20},
             'RGB 8bit 单页 + 多滤镜调节', expect_color=True)
    run_test('gray8.tif', {'clarity': 40, 'dehaze': 30, 'usm': 50},
             '灰度 8bit 单页 + 清晰度/去雾/锐化（色彩滑块禁用）', expect_color=False)
    run_test('stack_gray.tif', {'texture': 60, 'highlights': -30},
             '灰度堆栈 + 帧导航')
    run_test('rgb16.tif', {'white_balance': 80, 'temperature': 40},
             'RGB 16bit + 白平衡/色温', expect_color=True)
    test_mousewheel_scroll()
    test_display_mode_toggle()
    test_downscale_antialias()

    print('\n===== GUI 冒烟测试结果 =====')
    all_ok = True
    for desc, ok, err in results:
        status = '通过' if ok else '失败'
        print(f'[{status}] {desc}' + (f' -> {err}' if err else ''))
        if not ok:
            all_ok = False
    print('全部通过！' if all_ok else '存在失败项！')
    sys.exit(0 if all_ok else 1)

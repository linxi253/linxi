# -*- coding: utf-8 -*-
"""
电镜晶体/非晶区域统计分析工具

程序入口：初始化 ttkbootstrap 深色主题窗口并启动主应用。

项目结构：
    main.py              - 入口文件
    constants.py         - 全局常量
    core/                - 核心算法
        segmentation.py  - 图像分割（5种方法）
        measurement.py   - 面积/周长/形状因子测量
        analysis.py      - 生长速率/统计分析
        logger.py        - 线程安全日志
    gui/                 - 界面组件
        app.py           - 主窗口
        parameter_panel.py - 参数面板
        canvas_viewer.py - 画布查看器
    io_utils/            - 输入输出
        tiff_handler.py  - TIFF 读写
        exporter.py      - CSV/Excel 导出
        app_config.py    - 配置持久化

说明：打包后的模块全部位于 PyInstaller 的 PYZ 归档内，无需向 sys.path
注入 exe 所在目录（避免目录内同名模块被优先导入的植入面）。
"""

import sys
import traceback


def main():
    """应用主入口"""
    try:
        import ttkbootstrap as ttkb
        from gui.app import EMImageAnalyzerApp

        root = ttkb.Window(themename="darkly")
        app = EMImageAnalyzerApp(root)
        root.mainloop()

    except ImportError as e:
        error_msg = (
            f"缺少依赖库: {e}\n\n"
            f"请运行 install_deps.bat 安装依赖，或执行:\n"
            f"  pip install -r requirements.txt"
        )
        print(error_msg)
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("启动失败", error_msg)
            root.destroy()
        except Exception:
            pass
        sys.exit(1)

    except Exception as e:
        print(f"程序启动失败: {e}")
        traceback.print_exc()
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("启动失败", f"程序启动失败:\n{e}")
            root.destroy()
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()

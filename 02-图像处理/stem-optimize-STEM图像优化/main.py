"""
HRTEM/STEM 图像增强工具 v2.1 — 入口点

启动应用并配置日志系统。
"""

import logging
import os
import sys
import traceback
from logging.handlers import RotatingFileHandler

from version import APP_NAME, APP_VERSION

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



def setup_logging():
    """配置日志系统，同时输出到文件和控制台。"""
    # 日志文件路径：
    #   打包后 (sys.frozen) 写入 %APPDATA%/STEM_Enhancer/，避免写入临时解压目录
    #   开发模式写入脚本所在目录
    if getattr(sys, "frozen", False):
        log_dir = os.path.join(
            os.environ.get("APPDATA", os.path.expanduser("~")), "STEM_Enhancer"
        )
    else:
        log_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "stem_enhancer.log")

    handlers = [
        RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
    ]
    if sys.stdout is not None:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("%s v%s 启动", APP_NAME, APP_VERSION)
    logger.info(f"日志文件: {log_file}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"工作目录: {os.getcwd()}")
    logger.info("=" * 50)
    return logger


def main():
    """主入口函数。"""
    logger = setup_logging()
    self_test_mode = sys.argv[1:] == ["--self-test"]

    try:
        if self_test_mode:
            from self_test import run_self_test

            result = run_self_test()
            logger.info("打包运行时自检通过: %s", result)
            return

        from main_window import MainWindow

        app = MainWindow()
        app.run()
    except ImportError as e:
        logger.error(f"依赖模块导入失败: {e}")
        if self_test_mode:
            sys.exit(2)
        logger.error("请确保已安装所有依赖: pip install -r requirements.txt")
        # 尝试显示 GUI 错误提示
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "启动失败",
                f"依赖模块缺失:\n{e}\n\n请运行: pip install -r requirements.txt",
            )
            root.destroy()
        except Exception:
            logger.debug("无法显示依赖错误对话框", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序发生致命错误:\n{traceback.format_exc()}")
        if self_test_mode:
            sys.exit(2)
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "致命错误", f"程序发生错误:\n{e}\n\n详细信息请查看日志文件。"
            )
            root.destroy()
        except Exception:
            logger.debug("无法显示致命错误对话框", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("程序退出")


if __name__ == "__main__":
    main()

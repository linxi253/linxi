"""PyInstaller 运行时兼容 NumPy 1.x/2.x 的核心扩展位置。"""

try:
    import numpy._core._multiarray_tests  # noqa: F401
except ImportError:
    try:
        import numpy.core._multiarray_tests  # noqa: F401
    except ImportError:
        pass


"""Runtime hook to fix numpy module loading in PyInstaller bundles.

NumPy 2.x moved the private test extension from ``numpy.core`` to
``numpy._core``; import whichever exists so the bundle starts on both.
"""
try:
    import numpy._core._multiarray_tests  # noqa: F401  (NumPy >= 2.0)
except ImportError:
    try:
        import numpy.core._multiarray_tests  # noqa: F401  (NumPy 1.x)
    except ImportError:
        pass  # 测试扩展缺失不致命, 不应阻止应用启动

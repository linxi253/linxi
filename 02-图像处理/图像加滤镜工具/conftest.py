# -*- coding: utf-8 -*-
"""pytest 兼容层：会话级生成 test_images，使测试可被 pytest 直接收集。

官方入口仍是 ``python -X utf8 test_xxx.py``（build_release.py 按此执行），
本文件让 ``pytest`` 在 clean checkout 上同样可用，不必先手动跑 test_core。
"""

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


@pytest.fixture(scope='session', autouse=True)
def _generated_test_images():
    import test_core
    test_core.make_test_images()

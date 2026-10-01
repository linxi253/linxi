# -*- coding: utf-8 -*-
"""后台消息轮询的健壮性与日志测试。"""
from __future__ import annotations

import tkinter as tk
import unittest

from atomic_app import APP_VERSION, _setup_logging
from tests.tk_session import get_tk_session


class PollResilienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.root, cls.app = get_tk_session()
        except tk.TclError as error:
            raise unittest.SkipTest(f"Tk 不可用：{error}") from error

    def test_malformed_message_does_not_kill_polling(self) -> None:
        app = self.app
        app._worker_queue.put(("progress", app._job_generation, None))  # 解包会抛 TypeError
        app._poll_worker_queue()  # 不应向外抛异常
        self.assertTrue(app._poll_active)
        self.assertIsNotNone(app._poll_after_id, "轮询异常后必须继续重新注册 after")
        self.assertIn("内部错误", app.status.cget("text"))

    def test_normal_message_processed_after_malformed_one(self) -> None:
        app = self.app
        app._worker_queue.put(("progress", app._job_generation, None))
        app._poll_worker_queue()
        app._worker_queue.put(("progress", app._job_generation, (1, 1, 0, 5)))
        app._poll_worker_queue()
        self.assertIn("处理中", app.status.cget("text"))

    def test_logging_setup_returns_stable_path(self) -> None:
        logger_first, path_first = _setup_logging()
        logger_second, path_second = _setup_logging()
        self.assertIs(logger_first, logger_second)
        self.assertEqual(path_first, path_second)

    def test_app_version_is_v13(self) -> None:
        self.assertEqual(APP_VERSION, "1.3")
        self.assertIn("v1.3", self.app.root.title())


if __name__ == "__main__":
    unittest.main()

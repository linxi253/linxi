# -*- coding: utf-8 -*-
"""
日志模块

提供线程安全的日志记录功能，支持：
- GUI 文本框输出（通过 Tkinter after 机制保证线程安全）
- 状态栏更新
- 日志保存至文件
"""

import os
from datetime import datetime


class AppLogger:
    """应用日志管理器，线程安全地更新 Tkinter UI 组件"""

    MAX_ENTRIES = 5000  # 最大日志条目数，防止内存溢出

    def __init__(self, root, log_text_widget, status_var):
        """
        Args:
            root: Tkinter 根窗口（用于 after 调度）
            log_text_widget: 日志显示的 Text 控件
            status_var: 状态栏 StringVar
        """
        self._root = root
        self._log_text = log_text_widget
        self._status_var = status_var
        self._log_entries = []  # 内存中保存日志记录

    def log(self, message: str, level: str = "INFO"):
        """
        记录一条日志（线程安全）

        Args:
            message: 日志内容
            level: 日志级别 (INFO / WARN / ERROR)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"INFO": "", "WARN": "⚠ ", "ERROR": "✖ "}.get(level, "")
        log_entry = f"[{timestamp}] {prefix}{message}\n"

        # 保存到内存（限制最大条目数）
        self._log_entries.append(log_entry)
        if len(self._log_entries) > self.MAX_ENTRIES:
            self._log_entries = self._log_entries[-self.MAX_ENTRIES // 2:]

        # 调度到主线程更新 UI
        def _update_ui():
            try:
                self._log_text.insert('end', log_entry)
                # Text 控件与内存列表同策略裁剪（2026-09-28 审查修正：
                # 此前只有内存列表裁剪，长批量时控件无限增长，且"保存日志"
                # 落盘内容比屏幕显示的少）
                line_count = int(self._log_text.index('end-1c').split('.')[0])
                if line_count > self.MAX_ENTRIES:
                    self._log_text.delete('1.0', f'{line_count - self.MAX_ENTRIES // 2}.0')
                self._log_text.see('end')
                self._status_var.set(message)
            except Exception:
                pass  # 窗口关闭后忽略

        self._root.after(0, _update_ui)

    def warn(self, message: str):
        """记录警告日志"""
        self.log(message, level="WARN")

    def error(self, message: str):
        """记录错误日志"""
        self.log(message, level="ERROR")

    def clear(self):
        """清空日志显示"""
        def _clear_ui():
            self._log_text.delete('1.0', 'end')
        self._root.after(0, _clear_ui)
        self._log_entries.clear()
        self.log("日志已清空")

    def save(self, output_folder: str) -> str:
        """
        保存日志到文件

        Args:
            output_folder: 输出目录路径

        Returns:
            保存的文件路径

        Raises:
            ValueError: 日志为空或目录无效
            IOError: 写入失败
        """
        if not self._log_entries:
            raise ValueError("日志为空，没有内容可保存")

        if not os.path.isdir(output_folder):
            raise ValueError(f"输出目录不存在: {output_folder}")

        filename = f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = os.path.join(output_folder, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(self._log_entries)

        self.log(f"日志已保存到: {filepath}")
        return filepath

    def get_all_entries(self) -> list:
        """获取所有日志条目"""
        return list(self._log_entries)

# -*- coding: utf-8 -*-
"""
应用配置持久化

以 JSON 保存/恢复界面状态（文件夹、像素尺寸、分割参数等），
让科研批处理的参数跨会话可复用。写入用临时文件 + os.replace
保证断电/崩溃不产生半写配置；任何异常静默降级为"不持久化"，
绝不影响主流程。
"""

import json
import os

_CONFIG_DIR_NAME = 'EMAmorphTool'


def _config_path() -> str:
    base = os.environ.get('APPDATA') or os.path.expanduser('~')
    return os.path.join(base, _CONFIG_DIR_NAME, 'config.json')


def load_config() -> dict:
    """读取配置；文件缺失/损坏时返回空 dict。"""
    path = _config_path()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_config(data: dict) -> bool:
    """保存配置；失败返回 False（不抛出，持久化失败不应影响分析）。"""
    try:
        cfg_dir = os.path.dirname(_config_path())
        os.makedirs(cfg_dir, exist_ok=True)
        path = _config_path()
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return True
    except Exception:
        return False

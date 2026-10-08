# -*- coding: utf-8 -*-
"""配置持久化测试。"""

import io_utils.app_config as app_config


class TestAppConfig:
    def test_roundtrip(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / 'config.json'
        monkeypatch.setattr(app_config, '_config_path', lambda: str(cfg_file))

        data = {
            'input_folder': str(tmp_path / 'data'),
            'output_folder': str(tmp_path / 'out'),
            'pixel_size': 0.5,
            'pixel_calibrated': True,
            'segmentation': {'method': 'otsu', 'threshold': 100},
        }
        assert app_config.save_config(data) is True
        assert app_config.load_config() == data

    def test_load_missing_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            app_config, '_config_path', lambda: str(tmp_path / 'nope.json'))
        assert app_config.load_config() == {}

    def test_load_corrupted_returns_empty(self, tmp_path, monkeypatch):
        bad = tmp_path / 'bad.json'
        bad.write_text('{not json', encoding='utf-8')
        monkeypatch.setattr(app_config, '_config_path', lambda: str(bad))
        assert app_config.load_config() == {}

    def test_save_failure_returns_false_not_raises(self, tmp_path, monkeypatch):
        # 指向一个目录路径使其必然写入失败，持久化失败不得抛异常
        monkeypatch.setattr(app_config, '_config_path', lambda: str(tmp_path))
        assert app_config.save_config({'a': 1}) is False

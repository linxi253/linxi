"""Legacy training entry: the supported workflow lives in the atom-center package."""
import sys

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


MESSAGE = """
旧训练入口已迁移到“原子中心识别模型开发”，避免未经审计的数据和全局 best.pt 覆盖。
请使用该项目的 Python 环境：
  python -m atom_center.cli audit <annotation_project.json> --output audit.json
  python -m atom_center.cli build <project1.json> <project2.json> ... --output <new_dataset>
  python -m atom_center.cli train --data <new_dataset> --run <new_run> --common configs/common.yaml --modality configs/haadf_stem.yaml
续训：python -m atom_center.cli resume --run <new_run>
旧 YOLO 导出应从原始 JSON 标注重新审计、生成。brightness/contrast 现为独立 augmentation 配置。
""".strip()


def train(*args, **kwargs):
    raise RuntimeError(MESSAGE)


def main():
    print(MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

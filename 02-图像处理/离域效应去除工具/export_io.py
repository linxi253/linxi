"""Shared source protection and atomic single-file export."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile


def protect_source(path, *sources):
    target = os.path.normcase(os.path.realpath(os.path.abspath(path)))
    for source in sources:
        if not source:
            continue
        canonical = os.path.normcase(os.path.realpath(os.path.abspath(source)))
        same = target == canonical
        if not same and os.path.exists(path) and os.path.exists(source):
            same = os.path.samefile(path, source)
        if same:
            raise ValueError("拒绝覆盖源文件，请换一个输出路径")


@contextmanager
def atomic_path(path, *sources):
    protect_source(path, *sources)
    target = Path(path).absolute()
    fd, tmp = tempfile.mkstemp(prefix=".deloc_save_", suffix=target.suffix, dir=target.parent)
    os.close(fd)
    try:
        yield tmp
        protect_source(path, *sources)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_json(path, data, *sources):
    with atomic_path(path, *sources) as tmp:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, allow_nan=False)

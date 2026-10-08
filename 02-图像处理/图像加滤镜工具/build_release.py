"""Test, package, and checksum the Windows release using this interpreter."""

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

from version import __version__

# Windows 中文控制台/重定向（GBK/cp936）环境下，print 中文、✓ 等字符会触发 UnicodeEncodeError
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass



def main():
    project = Path(__file__).resolve().parent
    for test in ('test_core.py', 'test_batch.py', 'test_gui.py', 'test_packaging.py'):
        subprocess.run([sys.executable, '-X', 'utf8', test], cwd=project, check=True)
    subprocess.run(
        [sys.executable, '-m', 'PyInstaller', '--clean', '--noconfirm', 'TIF_FilterTool.spec'],
        cwd=project, check=True,
    )
    dist = project / 'dist'
    executable = dist / 'TIF_FilterTool.exe'
    versioned = dist / f'TIF_FilterTool-v{__version__}.exe'
    shutil.copy2(executable, versioned)
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    (dist / 'SHA256SUMS.txt').write_text(
        f'{digest}  {executable.name}\n{digest}  {versioned.name}\n', encoding='utf-8',
    )
    print(f'Release ready: {versioned}')


if __name__ == '__main__':
    main()

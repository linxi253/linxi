"""Pytest path configuration for src-layout without pip install -e."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
# 项目根目录也要可导入：tests/test_facade.py 需要引入根目录的 butter.py facade。
sys.path.insert(0, str(_ROOT))

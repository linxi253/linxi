"""Validated, immutable processing parameters shared by UI, JSON and workers."""
from dataclasses import asdict, dataclass
import math

ALGORITHM_VERSION = "2.0.0"
SCHEMA_VERSION = 2
MAX_PIXELS = 4096 * 4096
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_SHAPES = 1000
MAX_POINTS = 100000


def finite_number(value, name, low, high):
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是数字，不能是布尔值")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是有限数字") from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} 必须在 {low} ~ {high} 之间")
    return number


@dataclass(frozen=True)
class Parameters:
    fmin: float = 0.058
    fmax: float = 0.092
    dilate: float = 6.0
    feather: float = 8.0
    strength: float = 1.0
    # Millicycles/pixel, independent of image size and FFT/DCT grid.
    band_feather: float = 2.0

    def __post_init__(self):
        for name, bounds in {"fmin": (1e-6, .5), "fmax": (1e-6, .5),
                             "dilate": (0, 60), "feather": (0, 40),
                             "strength": (0, 1), "band_feather": (0, 6)}.items():
            object.__setattr__(self, name, finite_number(getattr(self, name), name, *bounds))
        if self.fmin >= self.fmax:
            raise ValueError("频带要求 0 < 下限 < 上限 ≤ 0.5")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data, defaults=None):
        if not isinstance(data, dict):
            raise ValueError("参数必须是 JSON 对象")
        values = (defaults or cls()).to_dict()
        values.update({k: data[k] for k in values if k in data})
        return cls(**values)

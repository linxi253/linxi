from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from atom_center.image_io import (
    AmbiguousImageError,
    inspect_image_series,
    load_image,
    normalize_percentile,
)


def test_percentile_normalization_is_bounded() -> None:
    image = np.arange(100, dtype=np.float64).reshape(10, 10)
    output = normalize_percentile(image, low=1, high=99)
    assert output.dtype == np.float32
    assert output.min() == 0.0
    assert output.max() == 1.0


def test_multiframe_tiff_requires_explicit_frame(tmp_path: Path) -> None:
    path = tmp_path / "stack.tif"
    stack = np.stack(
        [np.full((8, 9), 10, dtype=np.uint16), np.full((8, 9), 20, dtype=np.uint16)]
    )
    tifffile.imwrite(path, stack, photometric="minisblack", metadata={"axes": "QYX"})
    with pytest.raises(AmbiguousImageError):
        load_image(path, normalize=False)
    selected = load_image(path, frame_index=1, normalize=False)
    assert selected.image.shape == (8, 9)
    assert np.all(selected.image == 20)
    assert selected.info.frame_index == 1

    descriptions = inspect_image_series(path)
    assert len(descriptions) == 1
    assert descriptions[0].frame_count == 2
    assert descriptions[0].plane_shape == (8, 9)


def test_rgb_is_not_mistaken_for_a_stack(tmp_path: Path) -> None:
    path = tmp_path / "rgb.png"
    rgb = np.zeros((6, 7, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    Image.fromarray(rgb).save(path)
    loaded = load_image(path, normalize=False)
    assert loaded.image.shape == (6, 7)
    assert np.allclose(loaded.image, 255 * 0.2126)

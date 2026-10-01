import numpy as np

from atom_center.geometry import (
    clip_roi,
    generate_tiles,
    merge_close_points,
    tiles_cover_image,
)
from atom_center.interfaces import CandidateSet
from atom_center.pipeline import DetectionPipeline, PipelineConfig


class BrightPixelBackend:
    name = "bright-pixel-test"
    model_sha256 = None

    def predict(self, image: np.ndarray) -> CandidateSet:
        ys, xs = np.nonzero(image > 0.5)
        points = np.column_stack((xs, ys)).astype(np.float64)
        return CandidateSet(points, np.full(len(points), 0.9))


def test_tiles_cover_non_divisible_image() -> None:
    tiles = generate_tiles((101, 137), tile_size=(64, 64), overlap=0.25)
    assert tiles_cover_image(tiles, (101, 137))
    assert max(tile.x1 for tile in tiles) == 137
    assert max(tile.y1 for tile in tiles) == 101


def test_roi_clips_and_uses_half_open_bounds() -> None:
    roi = clip_roi((-2.2, 3.2, 12.1, 30.0), (20, 10))
    assert (roi.x0, roi.y0, roi.x1, roi.y1) == (0, 3, 10, 20)


def test_point_nms_keeps_highest_confidence() -> None:
    merged = merge_close_points(
        [[10.0, 10.0], [10.5, 10.0], [20.0, 20.0]],
        [0.6, 0.9, 0.8],
        min_distance=1.0,
    )
    assert len(merged.points) == 2
    assert [10.5, 10.0] in merged.points.tolist()


def test_adaptive_nms_merges_large_blob_duplicates_without_merging_small_neighbors():
    from atom_center.geometry import point_nms_indices
    points = [[0.,0.],[5.,0.],[40.,0.],[48.,0.],[70.,0.]]
    keep=point_nms_indices(points,[.9,.8,.9,.7,.9],min_distance=3.,radii=[3.,3.,12.,12.,12.])
    assert set(keep)=={0,1,2,4}


def test_adaptive_refinement_merges_same_blob_and_preserves_neighbor_quality():
    class FixedBackend:
        name = 'wide-blob-test'
        def predict(self, image):
            return CandidateSet([[19.,24.],[23.,24.],[47.,24.]], [.9,.7,.8])
    yy,xx=np.mgrid[:64,:72]
    raw=100+500*np.exp(-((xx-21.)**2+(yy-24.)**2)/32)+500*np.exp(-((xx-47.)**2+(yy-24.)**2)/32)
    pipeline=DetectionPipeline(FixedBackend(),PipelineConfig(tile_size=96,refinement_method='adaptive_blob',
        refinement_polarity='bright',refinement_window=33,refinement_max_shift_px=12.,merge_after_refinement=True))
    result=pipeline.detect(raw)
    assert len(result.points)==2
    assert result.metadata['duplicates_removed_after_refinement']==1
    assert len(result.metadata['point_quality'])==2
    np.testing.assert_allclose(result.points,[[21.,24.],[47.,24.]],atol=.15)
    np.testing.assert_array_equal(result.confidences,[.9,.8])


def test_pipeline_restores_tile_and_roi_offsets() -> None:
    image = np.zeros((110, 120), dtype=np.float64)
    image[44, 37] = 1.0
    image[79, 91] = 1.0
    pipeline = DetectionPipeline(
        BrightPixelBackend(),
        PipelineConfig(tile_size=48, tile_overlap=0.25, merge_distance_px=1.0, refine=False),
    )
    full = pipeline.detect(image)
    assert sorted(map(tuple, full.points.tolist())) == [(37.0, 44.0), (91.0, 79.0)]

    roi_result = pipeline.detect(image, roi=(20, 30, 70, 70))
    assert roi_result.points.tolist() == [[37.0, 44.0]]
    assert roi_result.metadata["roi_xyxy"] == [20, 30, 70, 70]

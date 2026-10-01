"""Tests for the round-2 rejected experiment module, not for the production API.

The fit gate lives in scripts/experiment_fit_gate.py precisely because candidate C1 was not
adopted. These tests assert the experiment's own behaviour: the gate is inert unless enabled and
the production pipeline module is always restored.
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import atom_center.pipeline as pipeline_module                            # noqa: E402
from atom_center.interfaces import CandidateSet                           # noqa: E402
from atom_center.pipeline import DetectionPipeline, PipelineConfig        # noqa: E402
from atom_center.refinement import refine_with_diagnostics                # noqa: E402
from experiment_fit_gate import DEFAULT_QUANTILE, apply_fit_gate, fit_gate_enabled  # noqa: E402


class FixedBackend:
    name = "fit-gate-experiment-test"
    model_sha256 = None

    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float64)

    def predict(self, image):
        return CandidateSet(self.points, np.full(len(self.points), 0.9))


def synthetic_field():
    """One clean column plus an asymmetric merged pair that a single Gaussian explains poorly."""
    yy, xx = np.mgrid[:72, :120]
    raw = 100. + 900*np.exp(-((xx-30.)**2 + (yy-30.)**2)/(2*1.5**2))
    raw += 900*np.exp(-((xx-86.)**2 + (yy-30.)**2)/(2*1.5**2))
    raw += 300*np.exp(-((xx-94.)**2 + (yy-30.)**2)/(2*1.5**2))
    return raw


def run(quantile=None):
    backend = FixedBackend([[30., 30.], [90., 30.]])
    config = PipelineConfig(tile_size=160, refinement_method='gaussian', refinement_polarity='bright',
                            refinement_window=11, refinement_max_shift_px=4.,
                            merge_after_refinement=True)
    pipeline = DetectionPipeline(backend, config)
    if quantile is None:
        return pipeline.detect(synthetic_field())
    with fit_gate_enabled(quantile):
        return pipeline.detect(synthetic_field())


def test_production_api_has_no_gate_option():
    assert 'refinement_fit_gate_quantile' not in PipelineConfig.__dataclass_fields__
    assert pipeline_module.refine_with_diagnostics is refine_with_diagnostics


def test_pipeline_is_restored_after_the_experiment_context():
    before = pipeline_module.refine_with_diagnostics
    with fit_gate_enabled(DEFAULT_QUANTILE):
        assert pipeline_module.refine_with_diagnostics is not before
    assert pipeline_module.refine_with_diagnostics is before


def test_gate_leaves_the_coarse_candidate_only_where_the_fit_is_unreliable():
    gated, plain = run(DEFAULT_QUANTILE), run(None)
    statuses = [entry['status'] for entry in gated.metadata['point_quality']]
    assert statuses.count('fit_gate_reverted') == 1
    kept, reverted = statuses.index('refined'), statuses.index('fit_gate_reverted')
    np.testing.assert_allclose(gated.points[kept], plain.points[kept], atol=1e-9)
    np.testing.assert_allclose(gated.points[reverted], [90., 30.], atol=1e-9)
    assert not np.allclose(gated.points[reverted], plain.points[reverted], atol=.5)


def test_apply_fit_gate_is_inert_below_two_usable_fits():
    diagnostics = ({'status': 'refined', 'normalized_rmse': 0.2},)
    points = np.asarray([[1., 1.]])
    out, updated = apply_fit_gate(points, np.asarray([[5., 5.]]), diagnostics, DEFAULT_QUANTILE)
    np.testing.assert_array_equal(out, points)
    assert updated[0]['status'] == 'refined'

"""Smoke tests for the refactored 4D-STEM processor core.

These tests use synthetic data only, so they run on any machine without
the original F:\\ datasets.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import dm4_io
from core.dpc_core import (compute_com_robust, idpc_reconstruct,
                           find_alpha_from_radial)
from core.dimension_utils import fix_dimensions
from core.ssb_core import ssb_reconstruct
from core.ptychography import run_ptychography


def make_synthetic_datacube(scan=16, det=24, alpha=5, shift=(1.5, -1.0),
                            seed=0):
    """A BF disk whose centre shifts with scan position, plus a periodic
    atomic-column-like modulation so the BF image has real periodicity."""
    rng = np.random.RandomState(seed)
    ky, kx = np.meshgrid(np.arange(det), np.arange(det), indexing='ij')
    sy, sx = np.meshgrid(np.arange(scan), np.arange(scan), indexing='ij')
    data = np.zeros((scan, scan, det, det), dtype=np.float32)
    for i in range(scan):
        for j in range(scan):
            cy = det / 2 + shift[0] * (i - scan / 2) / (scan / 2)
            cx = det / 2 + shift[1] * (j - scan / 2) / (scan / 2)
            r2 = (ky - cy) ** 2 + (kx - cx) ** 2
            periodic = 1 + 0.5 * np.cos(2 * np.pi * i / 4) * np.cos(
                2 * np.pi * j / 4)
            data[i, j] = periodic * (r2 <= alpha ** 2).astype(np.float32)
            data[i, j] += 0.01 * rng.rand(det, det).astype(np.float32)
    return data


def test_dm4_dtype_mapping():
    assert np.dtype(dm4_io.numpy_dtype_from_code(2)) == np.dtype('<i2')
    assert np.dtype(dm4_io.numpy_dtype_from_code(4)) == np.dtype('<u2')
    assert np.dtype(dm4_io.numpy_dtype_from_code(6)) == np.dtype('<f4')
    assert np.dtype(dm4_io.numpy_dtype_from_code(7)) == np.dtype('<f8')
    assert np.dtype(dm4_io.numpy_dtype_from_code(10)) == np.dtype('u1')


def test_dm4_dtype_mapping_rejects_unknown():
    import pytest
    with pytest.raises(RuntimeError):
        dm4_io.numpy_dtype_from_code(999)


def test_preprocess_preserves_bf_disk():
    # Regression: the old global 99th-percentile upper clip flattened the BF
    # disk whenever the disk covers less than 1% of the cube (Cu-like
    # geometry: 512x512 detector, 3 px disk), destroying the CoM signal.
    det = 512
    yy, xx = np.mgrid[0:det, 0:det]
    disk = ((yy - 255.5) ** 2 + (xx - 255.5) ** 2 <= 3 ** 2).astype(float)
    rng = np.random.RandomState(0)
    cube = rng.rand(4, 4, det, det) + disk[None, None, :, :] * 10.0

    clipped = dm4_io.preprocess(cube)
    assert clipped.max() > 9.0          # disk core must survive the default
    assert clipped.dtype == np.float32

    # Explicit opt-in keeps the legacy upper-clip behaviour available.
    legacy = dm4_io.preprocess(cube, pmax=99)
    assert legacy.max() <= np.percentile(cube, 99) + 1e-6


def test_compute_com_robust_recovers_shift():
    data = make_synthetic_datacube()
    center = (data.shape[2] / 2, data.shape[3] / 2)
    alpha = 5
    com = compute_com_robust(data, center=center, radius=alpha)
    assert com['com_y'].shape == (16, 16)
    assert com['com_x'].shape == (16, 16)
    # CoM shift should vary smoothly across the scan
    assert abs(np.mean(com['com_y'])) < 0.05
    assert abs(np.mean(com['com_x'])) < 0.05
    assert np.std(com['com_y']) > 0.1
    assert np.std(com['com_x']) > 0.1


def test_find_alpha_recovers_disk_radius():
    data = make_synthetic_datacube()
    center = (data.shape[2] / 2, data.shape[3] / 2)
    alpha, _, _ = find_alpha_from_radial(data, center)
    assert 4 <= alpha <= 7


def test_idpc_phase_from_linear_shift_is_smooth():
    data = make_synthetic_datacube()
    center = (data.shape[2] / 2, data.shape[3] / 2)
    com = compute_com_robust(data, center=center, radius=5)
    phase = idpc_reconstruct(com)
    assert phase.shape == (16, 16)
    assert np.isfinite(phase).all()
    # Phase should be a smooth low-frequency map
    grad = np.gradient(phase)
    assert np.std(grad[0]) < 0.5
    assert np.std(grad[1]) < 0.5


def test_fix_dimensions_swaps_when_needed():
    data = make_synthetic_datacube(scan=16, det=16, alpha=5)
    # Simulate the swapped layout: (det, det, scan, scan)
    swapped = data.transpose(2, 3, 0, 1)
    fixed, info = fix_dimensions(swapped, method='auto')
    assert info['swapped'] is True
    assert fixed.shape == data.shape
    # The decision must be auditable from the returned scores.
    assert info['score_det_first'] > info['score_scan_first']


def test_fix_dimensions_ambiguous_noise_keeps_layout():
    # Pure noise scores nearly identically under both interpretations;
    # the margin gate must keep the DM4-header layout instead of letting
    # a coin-flip transpose the cube.
    rng = np.random.RandomState(11)
    noise = rng.rand(12, 12, 12, 12)
    fixed, info = fix_dimensions(noise, method='auto')
    assert info['swapped'] is False
    assert fixed.shape == noise.shape


def test_virtual_images_chunked_matches_naive():
    # The chunked einsum implementation must reproduce the naive
    # broadcast-multiply reduction (it replaced 4 full-cube temporaries).
    from core.dpc_core import compute_virtual_images

    data = make_synthetic_datacube(scan=6, det=12, alpha=4)
    center = (6.0, 6.0)
    vimg = compute_virtual_images(data, center, 4, chunk_rows=2)

    ky, kx = np.meshgrid(np.arange(12), np.arange(12), indexing='ij')
    r2 = (ky - 6.0) ** 2 + (kx - 6.0) ** 2
    bf_naive = np.sum(data * (r2 <= 16.0), axis=(2, 3))
    adf_naive = np.sum(data * (np.sqrt(r2) >= 5), axis=(2, 3))
    assert np.allclose(vimg['bf'], bf_naive, atol=1e-6)
    assert np.allclose(vimg['adf'], adf_naive, atol=1e-6)


def test_template_correlate_fft_matches_direct(monkeypatch):
    # The FFT path (large kernels) must agree with the direct
    # ndimage.correlate path (small kernels) — the template is odd-sized
    # and both pad with zeros, so they are numerically equivalent.
    from core import strain_mapping

    rng = np.random.RandomState(3)
    dp = rng.rand(40, 40)
    template = strain_mapping._disk_template(5.0)
    direct = strain_mapping._template_correlate(dp, template)

    monkeypatch.setattr(strain_mapping, '_DIRECT_CORRELATION_LIMIT', 0)
    via_fft = strain_mapping._template_correlate(dp, template)
    assert np.allclose(direct, via_fft, atol=1e-7)


def test_bragg_disk_detection_survives_fft_path(monkeypatch):
    # detect_bragg_disks must find the same subpixel position whether the
    # correlation runs the direct or the FFT route.
    from core import strain_mapping

    yy, xx = np.mgrid[0:64, 0:64]
    dp = np.clip((3.0 - np.sqrt((yy - 30.3) ** 2 + (xx - 41.7) ** 2)) / 1.5,
                 0, 1) * 100.0
    center = (32.0, 32.0)

    direct = strain_mapping.detect_bragg_disks(dp, 3.0, center=center)
    monkeypatch.setattr(strain_mapping, '_DIRECT_CORRELATION_LIMIT', 0)
    via_fft = strain_mapping.detect_bragg_disks(dp, 3.0, center=center)
    assert len(direct) == len(via_fft) == 1
    assert abs(direct[0, 0] - via_fft[0, 0]) < 1e-6
    assert abs(direct[0, 1] - via_fft[0, 1]) < 1e-6


def test_ssb_reconstruct_runs_and_saves_shapes():
    data = make_synthetic_datacube(scan=16, det=24, alpha=5)
    center = (data.shape[2] / 2, data.shape[3] / 2)
    result = ssb_reconstruct(data, alpha_pixels=5, center=center,
                             verbose=False)
    assert result['phase'].shape == (16, 16)
    assert result['amplitude'].shape == (16, 16)
    assert result['cancelled'] is False


def test_ssb_recovers_synthetic_phase():
    # End-to-end accuracy lock: SSB must recover the phase of the synthetic
    # forward model in core.ssb_core (crude Gaussian-probe kinematics, so the
    # absolute correlation ceiling is limited; ~0.50 is the deterministic
    # baseline for this configuration). Guards against sign flips, mirroring
    # and gross regressions such as a reintroduced aperture wrap-around.
    from core.ssb_core import generate_4dstem_fast

    scan, det, alpha = 24, 48, 8
    datacube, obj, _ = generate_4dstem_fast((scan, scan), (det, det), alpha,
                                            defocus_rad=0.0)
    center = (det / 2.0, det / 2.0)
    result = ssb_reconstruct(datacube, alpha_pixels=alpha, center=center,
                             verbose=False)
    assert result['cancelled'] is False

    rec_phase = np.angle(result['complex_obj'])
    true_phase = np.angle(obj)

    def corr(a, b):
        a = a.ravel() - a.mean()
        b = b.ravel() - b.mean()
        return float(a @ b / np.sqrt((a @ a) * (b @ b)))

    c_direct = corr(rec_phase, true_phase)
    c_flipped = corr(rec_phase, np.flip(true_phase))
    assert c_direct > 0.3
    assert c_direct > c_flipped


def test_ssb_cancellation():
    data = make_synthetic_datacube(scan=16, det=24, alpha=5)
    center = (data.shape[2] / 2, data.shape[3] / 2)
    calls = {'n': 0}

    def stop():
        calls['n'] += 1
        return calls['n'] > 5

    result = ssb_reconstruct(data, alpha_pixels=5, center=center,
                             verbose=False, should_stop=stop)
    assert result['cancelled'] is True
    # A partial Psi_s would inverse-transform into a wrong image, so a
    # cancelled run must not return any reconstruction output.
    assert result['phase'] is None
    assert result['amplitude'] is None


def test_ssb_with_aberrations_runs():
    # Regression: the aberration branch referenced an undefined
    # `aperture_shifted` (NameError) whenever defocus_rad/Cs_rad != 0.
    data = make_synthetic_datacube(scan=8, det=16, alpha=4)
    center = (data.shape[2] / 2.0, data.shape[3] / 2.0)
    result = ssb_reconstruct(data, alpha_pixels=4, center=center,
                             defocus_rad=1.5, Cs_rad=0.3, verbose=False)
    assert result['cancelled'] is False
    assert result['phase'].shape == (8, 8)
    assert np.isfinite(result['phase']).all()
    assert np.isfinite(result['amplitude']).all()


def test_ssb_q_indices_cover_each_bin_once():
    # Regression: for even N the Nyquist bin N/2 appeared in both the
    # positive and negative frequency lists, so the Q loop processed it
    # twice with mirrored DO+ masks.
    from core.ssb_core import _generate_q_indices

    for n in (16, 15, 64):
        indices = [fft_index for fft_index, _ in
                   _generate_q_indices(n, 2 * n)]
        assert len(indices) == len(set(indices)), \
            f"duplicate FFT bins for N={n}"
        assert set(indices) == set(range(1, n)), \
            f"non-DC bins not fully covered for N={n}"


def test_ssb_do_plus_no_wrap():
    # Regression: np.roll on the aperture wrapped disk pixels around the
    # detector edges, fabricating double-overlap pixels whenever 2*alpha is
    # comparable to the detector size (under-sampled detectors such as the
    # 32x32 Au datasets). The analytic shifted apertures must match a
    # zero-padded reference exactly.
    from core.ssb_core import create_aperture_mask, double_overlap_plus

    det, center, alpha = 32, (15.5, 15.5), 10.0
    aperture = create_aperture_mask((det, det), center, alpha)

    def zero_padded_do(qy, qx):
        pad = det
        big = np.zeros((3 * det, 3 * det))
        big[pad:pad + det, pad:pad + det] = aperture
        shifted = np.roll(big, (-qy, -qx), axis=(0, 1))[pad:pad + det,
                                                       pad:pad + det]
        shifted_neg = np.roll(big, (qy, qx), axis=(0, 1))[pad:pad + det,
                                                         pad:pad + det]
        do = big[pad:pad + det, pad:pad + det] * shifted
        return (do * (1 - do * shifted_neg)).sum()

    def legacy_roll_do(qy, qx):
        return (aperture * np.roll(aperture, (-qy, -qx),
                                   axis=(0, 1))).sum()

    for Q in ((0, 14), (0, 16)):  # wrap-around regime (old code: 72 / 64 px)
        got = double_overlap_plus(aperture, center, alpha, *Q).sum()
        assert got == zero_padded_do(*Q)
        assert got < legacy_roll_do(*Q)

    # Without wrap-around the analytic mask equals the roll-based one.
    for Q in ((0, 6), (3, 5)):
        qy, qx = Q
        ap_s = np.roll(aperture, (-qy, -qx), axis=(0, 1))
        ap_sn = np.roll(aperture, (qy, qx), axis=(0, 1))
        rolled = aperture * ap_s
        rolled = rolled * (1 - rolled * ap_sn)
        assert np.array_equal(
            double_overlap_plus(aperture, center, alpha, *Q), rolled)


def test_epie_fixed_offsets_no_wrap():
    # Old code wrapped negative slices when sy < det//2. With a 6x6 scan and
    # 12x12 detector the first rows would have been corrupted.
    data = make_synthetic_datacube(scan=6, det=12, alpha=4)
    center = (6.0, 6.0)
    result = run_ptychography(data, center, alpha=4, method='epie',
                              n_iterations=2, step_size=0.3, verbose=False)
    assert result['phase'].shape == (6, 6)
    assert result['object'].shape == (6, 6)
    assert np.isfinite(result['phase']).all()
    assert result['errors'][-1] >= 0

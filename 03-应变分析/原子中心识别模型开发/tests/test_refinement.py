import numpy as np

from atom_center.refinement import refine_points


def gaussian_image(center: tuple[float, float], *, dark: bool = False) -> np.ndarray:
    ys, xs = np.mgrid[0:31, 0:31]
    x, y = center
    peak = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * 1.4**2))
    return 1.0 - peak if dark else peak


def test_bright_gaussian_refinement() -> None:
    truth = np.asarray([[15.35, 14.65]])
    refined = refine_points(
        gaussian_image(tuple(truth[0])),
        [[15.0, 15.0]],
        window_size=9,
        method="gaussian",
        polarity="bright",
    )
    assert np.linalg.norm(refined[0] - truth[0]) < 0.03


def test_auto_polarity_handles_dark_atom() -> None:
    truth = np.asarray([[14.30, 16.20]])
    refined = refine_points(
        gaussian_image(tuple(truth[0]), dark=True),
        [[14.0, 16.0]],
        window_size=9,
        method="com",
        polarity="auto",
    )
    assert np.linalg.norm(refined[0] - truth[0]) < 0.12
def test_continuous_com_has_no_half_pixel_window_jump():
    import numpy as np
    from atom_center.refinement import refine_with_diagnostics
    yy, xx = np.mgrid[:41, :41]
    raw = 100 + 500*np.exp(-((xx-20.2)**2+(yy-20.3)**2)/(2*2.**2))
    points = np.array([[20.49999,20.3],[20.50001,20.3]])
    result = refine_with_diagnostics(raw,points,method='com_continuous',polarity='bright',window_size=7,max_shift_px=2.4)
    assert np.linalg.norm(result.points[0]-result.points[1]) < .001
    assert all(d['status']=='refined' for d in result.diagnostics)
    assert np.max(np.linalg.norm(result.points-points,axis=1)) <= 2.4


def test_continuous_com_keeps_flat_and_boundary_points():
    import numpy as np
    from atom_center.refinement import refine_with_diagnostics
    points = np.array([[1.,1.],[10.,10.]])
    result=refine_with_diagnostics(np.ones((24,24)),points,method='com_continuous',polarity='bright')
    np.testing.assert_array_equal(points,result.points)
    assert [d['status'] for d in result.diagnostics] == ['image_boundary','flat']


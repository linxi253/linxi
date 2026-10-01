"""Sub-pixel refinement on unmodified intensities, with per-point diagnostics."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import warnings
import numpy as np
from numpy.typing import ArrayLike, NDArray

Polarity = Literal["bright", "dark", "auto"]
RefinementMethod = Literal["com", "com_continuous", "gaussian", "adaptive_blob"]


@dataclass(frozen=True)
class RefinementResult:
    points: NDArray[np.float64]
    diagnostics: tuple[dict[str, object], ...]


def refine_with_diagnostics(
    image: ArrayLike, points: ArrayLike, *, window_size: int = 7,
    method: RefinementMethod = "com", polarity: Polarity = "auto",
    max_shift_fraction: float = 0.8, max_shift_px: float | None = None,
) -> RefinementResult:
    """Keep candidates on failed fits. Use full raw windows, not reflected pixels."""
    array = np.asarray(image, dtype=np.float64)
    coords = np.asarray(points, dtype=np.float64)
    if coords.size == 0:
        coords = np.empty((0, 2), dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("refinement expects a finite 2-D image")
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("points must be finite with shape (N, 2)")
    if not isinstance(window_size, int) or window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")
    if method not in {"com", "com_continuous", "gaussian", "adaptive_blob"} or polarity not in {"bright", "dark", "auto"}:
        raise ValueError("invalid refinement method or polarity")
    if not 0 < max_shift_fraction <= 1:
        raise ValueError("max_shift_fraction must be within (0, 1]")
    half = window_size // 2
    limit = half * max_shift_fraction if max_shift_px is None else float(max_shift_px)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("max_shift_px must be finite and positive")
    if method == "adaptive_blob":
        if polarity != "bright":
            raise ValueError("adaptive_blob currently requires explicit bright polarity")
        from scipy.ndimage import gaussian_laplace
        scales = (1., 1.5, 2., 3., 4., 6., 8.)
        result = coords.copy()
        diagnostics = [{"status": "outside_image", "method": method, "shift_px": 0.} for _ in coords]
        valid = (coords[:,0]>=0)&(coords[:,0]<array.shape[1])&(coords[:,1]>=0)&(coords[:,1]<array.shape[0])
        indices = np.flatnonzero(valid)
        if not len(indices):
            return RefinementResult(result, tuple(diagnostics))
        pixel = np.rint(coords[valid]).astype(int)
        x = np.clip(pixel[:,0],0,array.shape[1]-1); y = np.clip(pixel[:,1],0,array.shape[0]-1)
        responses = np.stack([-gaussian_laplace(array, sigma)[y,x]*sigma*sigma for sigma in scales])
        chosen = np.argmax(responses,axis=0)
        for index, sigma in enumerate(scales):
            owned = indices[chosen==index]
            if not len(owned):
                continue
            wide = sigma >= 3.
            window = min(window_size, int(4*sigma)+1 if wide else 7)
            shift = min(limit, .8*(window//2)) if wide else min(limit, 2.4)
            sub = refine_with_diagnostics(array,coords[owned],window_size=window,
                method="gaussian" if wide else "com_continuous",polarity="bright",max_shift_px=shift)
            result[owned] = sub.points
            for original, detail in zip(owned, sub.diagnostics):
                diagnostics[original] = {**detail, "requested_method": method, "estimated_sigma_px": sigma,
                                         "window_size": window, "max_shift_px": shift}
        return RefinementResult(result, tuple(diagnostics))
    output = coords.copy()
    diagnostics: list[dict[str, object]] = []
    height, width = array.shape
    ys, xs = np.mgrid[-half:half + 1, -half:half + 1]
    for index, (x, y) in enumerate(coords):
        entry: dict[str, object] = {
            "status": "unchanged", "method": method, "polarity": polarity,
            "shift_px": 0.0, "proposed_shift_px": None, "normalized_rmse": None}
        diagnostics.append(entry)
        cx, cy = int(round(float(x))), int(round(float(y)))
        if not (0 <= x < width and 0 <= y < height):
            entry["status"] = "outside_image"
            continue
        if cx-half < 0 or cy-half < 0 or cx+half >= width or cy+half >= height:
            entry["status"] = "image_boundary"
            continue
        patch = array[cy-half:cy+half+1, cx-half:cx+half+1]
        median = float(np.median(patch))
        resolved = polarity
        if polarity == "auto":
            resolved = "bright" if patch.max()-median >= median-patch.min() else "dark"
        entry["polarity"] = resolved
        signed = patch if resolved == "bright" else -patch
        contrast = float(np.ptp(signed))
        if contrast <= np.finfo(float).eps * max(1., float(np.abs(signed).max())) * 32:
            entry["status"] = "flat"
            continue
        if method == "com_continuous":
            # Interpolate centroid estimates from adjacent integer windows.
            # Intensities in each window remain unmodified. This removes the
            # half-pixel window jump of round(x)/round(y) used by legacy COM.
            fx, fy = int(np.floor(x)), int(np.floor(y))
            ax, ay = float(x-fx), float(y-fy)
            proposals, coefficients = [], []
            for ox, oy, coefficient in ((0,0,(1-ax)*(1-ay)), (1,0,ax*(1-ay)),
                                        (0,1,(1-ax)*ay), (1,1,ax*ay)):
                if coefficient == 0:
                    continue
                px, py = fx+ox, fy+oy
                if px-half < 0 or py-half < 0 or px+half >= width or py+half >= height:
                    break
                local = array[py-half:py+half+1, px-half:px+half+1]
                signed_local = local if resolved == "bright" else -local
                weights = np.maximum(signed_local-np.percentile(signed_local, 5), 0.)
                mass = float(weights.sum())
                if mass <= 0:
                    break
                proposals.append([px+float((weights*xs).sum()/mass), py+float((weights*ys).sum()/mass)])
                coefficients.append(coefficient)
            if not coefficients or not np.isclose(sum(coefficients), 1., atol=1e-12, rtol=0.):
                entry["status"] = "incomplete_continuous_window"
                continue
            proposed = np.average(proposals, axis=0, weights=coefficients)
            dx, dy = float(proposed[0]-cx), float(proposed[1]-cy)
        elif method == "com":
            weights = np.maximum(signed-np.percentile(signed, 5), 0.)
            mass = float(weights.sum())
            if mass <= 0:
                entry["status"] = "zero_mass"
                continue
            dx, dy = float((weights*xs).sum()/mass), float((weights*ys).sum()/mass)
        else:
            from scipy.optimize import OptimizeWarning, curve_fit
            # Fit actual intensities plus background, without percentile clipping.
            target = (signed-signed.min()) / contrast

            def gaussian(coordinates, amplitude, px, py, sx, sy, background):
                xx, yy = coordinates
                return amplitude*np.exp(-.5*((xx-px)**2/sx**2+(yy-py)**2/sy**2))+background

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", OptimizeWarning)
                    fitted, _ = curve_fit(
                        gaussian, (xs.ravel(), ys.ravel()), target.ravel(),
                        p0=[1., float(x-cx), float(y-cy), 1.3, 1.3, 0.],
                        bounds=([0, -half, -half, .3, .3, -1],
                                [4, half, half, float(half), float(half), 1]),
                        maxfev=1500)
                predicted = gaussian((xs.ravel(), ys.ravel()), *fitted)
                residual = float(np.sqrt(np.mean((predicted-target.ravel())**2)))
                entry["normalized_rmse"] = residual
                if not np.isfinite(fitted).all() or residual > .25:
                    entry["status"] = "poor_fit"
                    continue
                dx, dy = float(fitted[1]), float(fitted[2])
            except (RuntimeError, ValueError, FloatingPointError, OptimizeWarning):
                entry["status"] = "fit_failed"
                continue
        proposed = np.asarray([cx+dx, cy+dy])
        shift = float(np.linalg.norm(proposed-coords[index]))
        entry["proposed_shift_px"] = shift
        if shift > limit:
            entry["status"] = "excessive_shift"
            continue
        if not (0 <= proposed[0] < width and 0 <= proposed[1] < height):
            entry["status"] = "outside_image"
            continue
        output[index] = proposed
        entry.update(status="refined", shift_px=shift)
    return RefinementResult(output, tuple(diagnostics))


def refine_points(
    image: ArrayLike, points: ArrayLike, *, window_size: int = 7,
    method: RefinementMethod = "com", polarity: Polarity = "auto",
    max_shift_fraction: float = .8, max_shift_px: float | None = None,
) -> NDArray[np.float64]:
    """Compatibility wrapper; refine_with_diagnostics also returns quality data."""
    return refine_with_diagnostics(
        image, points, window_size=window_size, method=method, polarity=polarity,
        max_shift_fraction=max_shift_fraction, max_shift_px=max_shift_px).points

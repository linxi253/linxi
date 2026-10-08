"""Finite, bounded project/settings validation before GUI state changes."""
import copy
import json
import math


class ProjectValidationError(ValueError):
    pass


MAX_POINTS = 250_000


def finite_number(value, label, *, minimum=None, maximum=None, integer=False):
    if isinstance(value, bool):
        raise ProjectValidationError(f"{label} must be a number, not bool.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ProjectValidationError(f"{label} must be numeric.") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum) or \
            (maximum is not None and result > maximum) or (integer and result != round(result)):
        raise ProjectValidationError(f"{label} is outside its supported finite range.")
    return int(result) if integer else result


def validate_project_data(payload, *, image_shape=None):
    if not isinstance(payload, dict):
        raise ProjectValidationError("Project must be a JSON object.")
    data = copy.deepcopy(payload)
    version = data.get('schema_version', 1)
    if type(version) is not int or version not in (1, 2):
        raise ProjectValidationError("Unsupported project schema version.")
    image = data.get('image')
    if version == 2:
        if not isinstance(image, dict) or not isinstance(image.get('path'), str):
            raise ProjectValidationError("Missing image identity metadata.")
        digest = image.get('sha256', '')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest.lower()):
            raise ProjectValidationError("Invalid image SHA-256.")
        finite_number(image.get('size_bytes'), 'image size', minimum=0, integer=True)
        frame = image.get('frame_index')
        if frame is not None and (type(frame) is not int or frame < 0):
            raise ProjectValidationError("Invalid image frame index.")
    points = data.get('points', [])
    if not isinstance(points, list) or len(points) > MAX_POINTS:
        raise ProjectValidationError("Invalid or oversized point list.")
    clean = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ProjectValidationError("Each point must contain exactly x and y.")
        x, y = (finite_number(v, 'point coordinate', minimum=0.) for v in point)
        if image_shape is not None and (x >= image_shape[1] or y >= image_shape[0]):
            raise ProjectValidationError("Project point lies outside the verified image.")
        clean.append([x, y])
    data['points'] = clean
    params = data.get('detect_params', {})
    if not isinstance(params, dict):
        raise ProjectValidationError("detect_params must be an object.")
    for key, default, lo, hi in [('sigma', .8, 0., 50.), ('min_dist', 8., 1e-9, 1e6), ('window', 5, 3, 101)]:
        params[key] = finite_number(params.get(key, default), key, minimum=lo, maximum=hi, integer=key == 'window')
    if params['window'] % 2 == 0 or params.get('method', 'com') not in ('com', 'gaussian'):
        raise ProjectValidationError("Detection requires an odd window and a known centroid method.")
    params.setdefault('method', 'com')
    data['detect_params'] = params
    threshold = params.get('threshold')
    if threshold is not None and threshold != '':
        threshold = finite_number(threshold, 'threshold', minimum=1e-12, maximum=1.-1e-12)
        params['threshold'] = threshold
    fallback = params.get('gaussian_fallback')
    if fallback is not None:
        if not isinstance(fallback, dict):
            raise ProjectValidationError("Invalid Gaussian refinement diagnostics.")
        n = finite_number(fallback.get('n_points'), 'n_points', minimum=0, maximum=MAX_POINTS, integer=True)
        finite_number(fallback.get('n_fallback'), 'n_fallback', minimum=0, maximum=n, integer=True)
        finite_number(fallback.get('fallback_ratio'), 'fallback_ratio', minimum=0, maximum=1)
    analysis = data.setdefault('analysis', {})
    if not isinstance(analysis, dict):
        raise ProjectValidationError("analysis must be an object.")
    analysis['equivalent_strain_coefficient'] = finite_number(
        analysis.get('equivalent_strain_coefficient', data.get('von_mises_coeff', 4./9.)),
        'equivalent_strain_coefficient', minimum=1e-12, maximum=1e6)
    analysis['lattice_max_condition'] = finite_number(analysis.get('lattice_max_condition', 30.),
                                                     'lattice_max_condition', minimum=1., maximum=1e6)
    reference = data.get('reference')
    if reference is not None:
        from .strain import validate_reference_lattice
        if not isinstance(reference, dict):
            raise ProjectValidationError("reference must be an object.")
        for key in ('origin', 'a_vec', 'b_vec'):
            values = reference.get(key)
            if not isinstance(values, (list, tuple)) or len(values) != 2:
                raise ProjectValidationError(f"Invalid reference {key}.")
            reference[key] = [finite_number(v, key) for v in values]
        validate_reference_lattice(reference['a_vec'], reference['b_vec'], max_condition=analysis['lattice_max_condition'])
        estimation = reference.get('estimation')
        if estimation is not None:
            if not isinstance(estimation, dict) or not isinstance(estimation.get('method', ''), str):
                raise ProjectValidationError("Invalid reference estimation metadata.")
            for key in ('a_point_count', 'b_point_count', 'a_anchor_count', 'b_anchor_count',
                        'a_interval_count', 'b_interval_count', 'origin_point_index'):
                if key in estimation:
                    estimation[key] = finite_number(estimation[key], key, minimum=0, maximum=MAX_POINTS, integer=True)
            for key in ('a_rms_fit_px', 'b_rms_fit_px', 'a_spacing_std_px', 'b_spacing_std_px'):
                if key in estimation:
                    estimation[key] = finite_number(estimation[key], key, minimum=0., maximum=1e6)
    preprocess = data.get('preprocess')
    if preprocess is not None:
        if not isinstance(preprocess, dict) or preprocess.get('method', 'none') not in \
                ('none', 'gaussian', 'median', 'butterworth', 'wiener', 'absf'):
            raise ProjectValidationError("Invalid preprocessing method.")
        if type(preprocess.get('use_preprocessed', False)) is not bool:
            raise ProjectValidationError("use_preprocessed must be boolean.")
        pp = preprocess.get('params', {})
        if not isinstance(pp, dict):
            raise ProjectValidationError("Preprocessing params must be an object.")
        bounds = {'gaussian_sigma': (0., 50., False), 'median_size': (1, 101, True),
                  'bw_order': (1, 20, True), 'bw_cutoff': (1e-9, 1., False),
                  'delta': (0., 1e6, False), 'cycles': (1, 1000, True), 'step': (1, 1000, True)}
        for key, value in pp.items():
            if key in bounds:
                lo, hi, integer = bounds[key]
                pp[key] = finite_number(value, key, minimum=lo, maximum=hi, integer=integer)
        if pp.get('median_size', 3) % 2 == 0:
            raise ProjectValidationError("Median window must be odd.")
    for name in ('detect_roi', 'ref_region'):
        rect = data.get(name)
        if rect is not None:
            if not isinstance(rect, (list, tuple)) or len(rect) != 4:
                raise ProjectValidationError(f"Invalid {name} rectangle.")
            rect = [finite_number(v, name, minimum=0.) for v in rect]
            if not (rect[0] < rect[2] and rect[1] < rect[3]):
                raise ProjectValidationError(f"Invalid {name} rectangle bounds.")
            if image_shape and (rect[2] > image_shape[1] or rect[3] > image_shape[0]):
                raise ProjectValidationError(f"{name} lies outside the image.")
            data[name] = rect
    polygon = data.get('detect_polygon')
    if polygon is not None:
        if not isinstance(polygon, list) or not 3 <= len(polygon) <= 10000:
            raise ProjectValidationError("Invalid polygon.")
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ProjectValidationError("Invalid polygon point.")
            x, y = (finite_number(v, 'polygon', minimum=0.) for v in point)
            if image_shape and (x > image_shape[1] or y > image_shape[0]):
                raise ProjectValidationError("Polygon lies outside the image.")
    colorbar = data.get('colorbar')
    if colorbar is None and any(key in data for key in ('colorbar_manual', 'colorbar_vmin', 'colorbar_vmax')):
        colorbar = data['colorbar'] = {'manual': data.get('colorbar_manual', False),
                                     'vmin': data.get('colorbar_vmin', 0.), 'vmax': data.get('colorbar_vmax', 1.)}
    if colorbar is not None:
        if not isinstance(colorbar, dict) or type(colorbar.get('manual', False)) is not bool:
            raise ProjectValidationError("Invalid colorbar settings.")
        for key, default in [('vmin', 0.), ('vmax', 1.)]:
            colorbar[key] = finite_number(colorbar.get(key, default), key)
        if colorbar.get('manual', False) and colorbar['vmin'] >= colorbar['vmax']:
            raise ProjectValidationError("Manual colorbar minimum must be less than maximum.")
    assignment = analysis.get('lattice_assignment')
    if assignment is not None:
        import numpy as np
        if not isinstance(assignment, dict):
            raise ProjectValidationError("Invalid lattice assignment metadata.")
        indices = np.asarray(assignment.get('lattice_indices'), dtype=float)
        if indices.shape != (len(clean), 2) or not np.isfinite(indices).all() or not np.equal(indices, np.rint(indices)).all():
            raise ProjectValidationError("Invalid integer lattice indices.")
        if len(np.unique(indices, axis=0)) != len(clean):
            raise ProjectValidationError("Lattice indices must be one-to-one.")
        if np.any(np.abs(indices) > 1e9):
            raise ProjectValidationError("Lattice indices exceed the supported range.")
        finite_number(assignment.get('initial_conflict_count', 0), 'initial_conflict_count', minimum=0, maximum=len(clean), integer=True)
        for key, value in assignment.items():
            if key in ('lattice_coords', 'residuals_lattice', 'residuals_px', 'reassigned', 'low_confidence', 'strain_quality', 'strain_invalid_reasons') and value is not None:
                expected = (len(clean), 2) if key == 'lattice_coords' else (len(clean),)
                array = np.asarray(value)
                # CST quality metadata belongs to triangles, not the atom table.
                if key.startswith('strain_') and analysis.get('algorithm_id') == 'lattice-cst-legacy':
                    continue
                if array.shape != expected:
                    raise ProjectValidationError(f"Invalid {key} length.")
                if key in ('lattice_coords', 'residuals_lattice', 'residuals_px') and not np.isfinite(array.astype(float)).all():
                    raise ProjectValidationError(f"Non-finite {key}.")
                if key in ('reassigned', 'low_confidence') and not np.isin(array, [True, False]).all():
                    raise ProjectValidationError(f"Invalid boolean {key}.")
    try:
        json.dumps(data, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ProjectValidationError("Project contains unsupported/non-finite metadata.") from exc
    return data

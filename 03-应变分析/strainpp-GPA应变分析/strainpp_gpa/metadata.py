"""Shared runtime/provenance helpers.

Single source of truth for the version string, dependency-version records,
and the JSON-normalised pixel-size form, so the GUI and the CLI exports
produce identical metadata instead of maintaining duplicate copies.
"""

import importlib.metadata

import numpy as np

__all__ = ['project_version', 'dependency_versions', 'json_pixel_size']

# Packages recorded in every exported metadata.json, absent ones become None.
_TRACKED_PACKAGES = (
    'numpy', 'scipy', 'matplotlib', 'tifffile', 'ttkbootstrap', 'ncempy',
)


def project_version() -> str:
    """Installed distribution version, falling back to ``_version.py``."""
    try:
        return importlib.metadata.version('strainpp-gpa')
    except importlib.metadata.PackageNotFoundError:
        pass
    except Exception:
        pass
    try:
        from _version import __version__
        return __version__
    except Exception:
        return '0.0.0'


def dependency_versions() -> dict:
    """Importable dependency versions for exported metadata."""
    versions = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
        except Exception:
            versions[name] = None
    return versions


def json_pixel_size(pixel_size) -> list:
    """Normalized ``[y, x]`` nm pair from a scalar or ``(y, x)`` input."""
    values = np.asarray(pixel_size, dtype=float)
    if values.ndim == 0:
        value = float(values)
        return [value, value]
    return [float(values[0]), float(values[1])]

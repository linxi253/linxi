"""
Original Strain++ colormap definitions, shared by the GUI and the batch
engine. Stop positions/colours are taken from the upstream Strain++ source
(ColorBarPlot::CreateColorMaps) and the QCustomPlot built-in gradients.
"""

from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    'COLORMAP_STOPS', 'build_cmap', 'register_all', 'get_cmap',
]

# name -> list of (position, (r, g, b) in 0..255)
COLORMAP_STOPS = {
    # JetLike_Map ("Turbo"): the default Strain++ strain-field ring.
    'turbo': [
        (0.0, (51, 27, 61)), (0.125, (77, 110, 223)),
        (0.25, (61, 185, 233)), (0.375, (68, 238, 154)),
        (0.5, (164, 250, 80)), (0.625, (235, 206, 76)),
        (0.75, (247, 129, 55)), (0.875, (206, 58, 32)),
        (1.0, (119, 21, 19)),
    ],
    # BluOr_Map: deep blue -> near-white -> orange.
    'blor': [
        (0.0, (7, 90, 254)), (0.5, (235, 255, 235)), (1.0, (255, 85, 0)),
    ],
    # QCPColorGradient::gpPolar: blue / black / red.
    'polar': [
        (0.0, (0, 0, 255)), (0.5, (0, 0, 0)), (1.0, (255, 0, 0)),
    ],
    # QCPColorGradient::gpThermal, approximated from its documentation.
    'thermal': [
        (0.0, (0, 0, 90)), (0.25, (110, 0, 140)), (0.5, (230, 60, 40)),
        (0.7, (255, 150, 0)), (0.85, (255, 230, 60)), (1.0, (255, 255, 255)),
    ],
    # QCPColorGradient::gpGrayscale.
    'greyscale': [
        (0.0, (0, 0, 0)), (1.0, (255, 255, 255)),
    ],
}

# Registered names used by the GUI dropdowns (prefixed to avoid clashing
# with matplotlib built-ins such as 'turbo').
PREFIX = '_spp_'

NAN_GREY = (0.851, 0.851, 0.851)  # #d9d9d9, distinct from any 0-value colour


def build_cmap(name: str):
    """Build (unregistered) colormap ``name`` from its stop table."""
    key = name[len(PREFIX):] if name.startswith(PREFIX) else name
    stops = COLORMAP_STOPS[key]
    return LinearSegmentedColormap.from_list(
        name,
        [(position, tuple(c / 255.0 for c in rgb)) for position, rgb in stops],
        N=256,
    )


def register_all() -> None:
    """Register every Strain++ colormap under matplotlib (idempotent)."""
    import matplotlib as mpl

    for key in COLORMAP_STOPS:
        name = PREFIX + key
        if name not in mpl.colormaps():
            mpl.colormaps.register(build_cmap(name), name=name)


def get_cmap(name: str = PREFIX + 'turbo'):
    """Return the registered colormap, registering on first use."""
    register_all()
    import matplotlib as mpl
    return mpl.colormaps[name]

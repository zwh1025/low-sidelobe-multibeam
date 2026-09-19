"""Physics kernels: grids, BP compensation phase, Taylor windows, FFT array factor."""

from .grid import angles_to_uv, uv_to_angles, fft_uv_axes, element_coords
from .bp_phase import bp_phase_plane, point_feed_geometry
from .taylor import (
    taylor_params,
    taylor_nulls,
    taylor_aperture_coeffs,
    taylor_window_1d,
    taylor_window_2d,
)
from .array_factor import (
    aperture_to_pattern_np,
    aperture_to_pattern_torch,
    pattern_from_phase_np,
)
from .metrics import evaluate_pattern, estimate_mainlobe_radius

__all__ = [
    "angles_to_uv",
    "uv_to_angles",
    "fft_uv_axes",
    "element_coords",
    "bp_phase_plane",
    "point_feed_geometry",
    "taylor_params",
    "taylor_nulls",
    "taylor_aperture_coeffs",
    "taylor_window_1d",
    "taylor_window_2d",
    "aperture_to_pattern_np",
    "aperture_to_pattern_torch",
    "pattern_from_phase_np",
    "evaluate_pattern",
    "estimate_mainlobe_radius",
]

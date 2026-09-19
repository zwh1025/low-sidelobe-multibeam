import numpy as np


def angles_to_uv(theta_deg, phi_az_deg):
    """(theta, phi_az) in degrees -> direction cosines (u, v).

    u = sin(theta) cos(phi_az), v = sin(theta) sin(phi_az); theta from +z (broadside).
    """
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
    phi = np.deg2rad(np.asarray(phi_az_deg, dtype=float))
    u = np.sin(theta) * np.cos(phi)
    v = np.sin(theta) * np.sin(phi)
    return u, v


def uv_to_angles(u, v):
    """Direction cosines -> (theta_deg, phi_az_deg in [0, 360))."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    rho = np.sqrt(u ** 2 + v ** 2)
    theta = np.rad2deg(np.arcsin(np.clip(rho, -1.0, 1.0)))
    phi = np.mod(np.rad2deg(np.arctan2(v, u)), 360.0)
    return theta, phi


def fft_uv_axes(n_pad, d_over_lambda=0.5):
    """Centered u (or v) axis of the zero-padded FFT grid.

    u_P = (P - n_pad/2) * lambda / (n_pad * d); with d = lambda/2 the axis spans [-1, 1).
    """
    return (np.arange(n_pad) - n_pad / 2) / (n_pad * d_over_lambda)


def element_coords(n, d_over_lambda=0.5):
    """Element coordinates in wavelength units, aperture centered: x_m = (m - (n-1)/2) d."""
    return (np.arange(n) - (n - 1) / 2) * d_over_lambda

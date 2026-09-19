import numpy as np

from .grid import angles_to_uv, element_coords


def bp_phase_plane(theta_deg, phi_az_deg, n=64, d_over_lambda=0.5):
    """BP compensation phase (geometry A, plane-wave aperture), shape (n, n) in [0, 2pi).

    phi_BP(m, n) = -k (x_m u0 + y_n v0) mod 2pi, coordinates in wavelength units.
    """
    u0, v0 = angles_to_uv(theta_deg, phi_az_deg)
    x = element_coords(n, d_over_lambda)
    ph = -2 * np.pi * (x[:, None] * u0 + x[None, :] * v0)
    return np.mod(ph, 2 * np.pi)


def point_feed_geometry(theta_deg, phi_az_deg, n=64, d_over_lambda=0.5,
                        f_over_D=0.8, feed_x=0.0, feed_y=0.0):
    """Geometry B (Du 2025 point-source fed reflectarray).

    Returns (bp_phase, feed_phase), each (n, n) in [0, 2pi):
      bp_phase   = (k r - k (x u0 + y v0)) mod 2pi
      feed_phase = -k r mod 2pi
    Aperture = exp(1j * (bp_phase + feed_phase)) equals the plane steering phase,
    so single-beam patterns coincide with geometry A; only raw code matrices differ.
    Feed at (feed_x, feed_y, f) in wavelength units, f = f_over_D * aperture_size.
    """
    x = element_coords(n, d_over_lambda)
    X, Y = np.meshgrid(x, x, indexing="ij")
    f = f_over_D * n * d_over_lambda
    r = np.sqrt((X - feed_x) ** 2 + (Y - feed_y) ** 2 + f ** 2)
    u0, v0 = angles_to_uv(theta_deg, phi_az_deg)
    bp = 2 * np.pi * (r - (X * u0 + Y * v0))
    feed = -2 * np.pi * r
    return np.mod(bp, 2 * np.pi), np.mod(feed, 2 * np.pi)

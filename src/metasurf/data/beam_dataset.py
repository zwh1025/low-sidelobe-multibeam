"""Beam direction sampling shared by phases 1-3 (dataset tensors come in phase 3)."""

import numpy as np


def angular_sep_deg(t1, p1, t2, p2):
    """3D angular separation between two (theta, phi_az) directions, degrees."""
    t1, p1, t2, p2 = (np.deg2rad(float(x)) for x in (t1, p1, t2, p2))
    c = np.cos(t1) * np.cos(t2) + np.sin(t1) * np.sin(t2) * np.cos(p1 - p2)
    return float(np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0))))


def sample_beams(rng, m, theta_min_deg=2.0, theta_max_deg=45.0, min_sep_deg=5.0,
                 max_tries=50000):
    """Random sequential sampling of m directions with pairwise min separation."""
    beams = []
    tries = 0
    while len(beams) < m:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(
                "sample_beams: max_tries exceeded (m={}, min_sep={})".format(m, min_sep_deg))
        th = float(rng.uniform(theta_min_deg, theta_max_deg))
        ph = float(rng.uniform(0.0, 360.0))
        if all(angular_sep_deg(th, ph, bt, bp) >= min_sep_deg for bt, bp in beams):
            beams.append((th, ph))
    return beams

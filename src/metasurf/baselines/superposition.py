"""Superposition baselines (Du et al., AWPL 2025) and continuous arg-sum variants."""

import numpy as np


def arg_sum_phase(phases, weights=None):
    """Baseline 1 (weights=None) / 2 (weights=amplitude prior): arg(sum_i w_i e^{j phi_i}).

    With a common real positive window shared by all beams, the weights factor
    out of the argument, so baseline 2 is analytically identical to baseline 1
    (verified numerically in the phase-1 runner).
    """
    z = np.exp(1j * np.asarray(phases, dtype=np.float64))
    if weights is not None:
        z = z * np.asarray(weights, dtype=np.float64)
    return np.mod(np.angle(z.sum(axis=0)), 2.0 * np.pi)


def onebit_code(phases):
    """Baseline 3, Du eq.(6): code = floor(mean_i |phi_i - pi| x 2/pi) in {0, 1}."""
    ph = np.asarray(phases, dtype=np.float64)
    m = np.mean(np.abs(ph - np.pi), axis=0) * (2.0 / np.pi)
    return np.clip(np.floor(m), 0.0, 1.0).astype(np.uint8)


def onebit_dual_eq4(phi1, phi2):
    """Du eq.(4) literal two-branch dual-beam 1-bit code (implementation spot-check).

    Branch 1 (|phi1-phi2| <= pi): floor(|(phi1+phi2)/2 - pi| x 2/pi)
    Branch 2 (|phi1-phi2| >  pi): floor(|((phi1+phi2)/2 + pi) mod 2pi - pi| x 2/pi)
    """
    phi1 = np.mod(np.asarray(phi1, dtype=np.float64), 2.0 * np.pi)
    phi2 = np.mod(np.asarray(phi2, dtype=np.float64), 2.0 * np.pi)
    s = 0.5 * (phi1 + phi2)
    d = np.abs(phi1 - phi2)
    b1 = np.abs(s - np.pi)
    b2 = np.abs(np.mod(s + np.pi, 2.0 * np.pi) - np.pi)
    val = np.where(d <= np.pi, b1, b2)
    return np.clip(np.floor(val * (2.0 / np.pi)), 0.0, 1.0).astype(np.uint8)


def code_phase(code):
    """1-bit code {0,1} -> physical phase {0, pi}."""
    return code.astype(np.float64) * np.pi

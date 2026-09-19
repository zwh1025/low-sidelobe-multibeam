import numpy as np


def taylor_params(sll_db, nbar):
    """R (main-to-sidelobe field ratio, >= 1), A = arccosh(R)/pi, sigma.

    Accepts negative dB (e.g. -25) or positive attenuation (25); magnitude is taken
    internally so arccosh always receives R >= 1.
    """
    atten = abs(float(sll_db))
    R = 10 ** (atten / 20)
    A = np.arccosh(R) / np.pi
    sigma = nbar / np.sqrt(A ** 2 + (nbar - 0.5) ** 2)
    return R, A, sigma


def taylor_nulls(sll_db, nbar):
    """Modified null positions u_n = sigma sqrt(A^2 + (n - 1/2)^2), n = 1..nbar-1."""
    _, A, sigma = taylor_params(sll_db, nbar)
    n = np.arange(1, nbar)
    return sigma * np.sqrt(A ** 2 + (n - 0.5) ** 2)


def taylor_aperture_coeffs(sll_db, nbar):
    """Aperture cosine-series coefficients F_k = S(k), k = 1..nbar-1.

    By the Taylor sampling theorem the ideal pattern equals its integer-sample
    expansion, so the aperture is the finite cosine series
      g(x) = 1 + 2 sum_k S(k) cos(2 pi k x / L).
    S(k) carries a removable sinc singularity, evaluated here in closed form:
      S(k) = (-1)^(k+1) (k/2) (1 - k^2/u_k^2) prod_{n != k} (1-k^2/u_n^2)/(1-k^2/n^2).
    """
    un = taylor_nulls(sll_db, nbar)
    ks = np.arange(1, nbar)
    F = np.zeros(len(ks))
    for i, k in enumerate(ks):
        val = ((-1) ** (k + 1)) * (k / 2.0) * (1.0 - k ** 2 / un[i] ** 2)
        for j in range(1, nbar):
            if j == k:
                continue
            val *= (1.0 - k ** 2 / un[j - 1] ** 2) / (1.0 - k ** 2 / j ** 2)
        F[i] = val
    return F


def taylor_window_1d(n=64, sll_db=-25.0, nbar=5):
    """Discrete Taylor n-bar taper sampled at n elements, normalized to (0, 1]."""
    F = taylor_aperture_coeffs(sll_db, nbar)
    m = np.arange(n) - (n - 1) / 2
    t = np.ones(n)
    for k, Fk in enumerate(F, start=1):
        t = t + 2.0 * Fk * np.cos(2 * np.pi * k * m / n)
    return t / t.max()


def taylor_window_2d(n=64, sll_db=-25.0, nbar=5):
    """Separable 2D Taylor taper A(m, n) = t(m) t(n)."""
    t = taylor_window_1d(n, sll_db, nbar)
    return np.outer(t, t)

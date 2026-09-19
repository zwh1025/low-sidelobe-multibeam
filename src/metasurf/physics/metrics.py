"""Unified pattern metrics: adaptive mainlobe masking, SLL, pointing error, gain consistency."""

import numpy as np

from .grid import angles_to_uv


def _vertex_offset(y0, y1, y2):
    """Quadratic vertex offset (in samples) from three log-magnitude samples."""
    denom = y0 - 2.0 * y1 + y2
    if denom >= 0:
        return 0.0
    return float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))


def _sep_uv_deg(u1, v1, u2, v2):
    """3D angular separation between two direction-cosine points, degrees."""
    w1 = np.sqrt(max(0.0, 1.0 - u1 * u1 - v1 * v1))
    w2 = np.sqrt(max(0.0, 1.0 - u2 * u2 - v2 * v2))
    c = u1 * u2 + v1 * v2 + w1 * w2
    return float(np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0))))


def estimate_mainlobe_radius(F_abs, U, V, u0, v0, r_min, r_max, n_rings=14):
    """First-null-based adaptive mainlobe radius around target (u0, v0).

    Ring-max profile of |F| vs distance from the target; the global profile
    minimum inside [r_min, r_max] locates the first null; radius = 1.15 x null
    distance, clipped to [r_min, r_max]. The square mainlobe null contour of
    separable apertures is covered because the radius is measured on axial
    rings and the mask disk is applied per beam.
    """
    if r_max <= r_min:
        return float(r_min)
    du = float(U[0, 1] - U[0, 0])
    half = r_max + 2.0 * du
    i_lo = int(np.clip(np.searchsorted(U[:, 0], u0 - half) - 1, 0, U.shape[0] - 2))
    i_hi = int(np.clip(np.searchsorted(U[:, 0], u0 + half) + 1, 1, U.shape[0] - 1))
    j_lo = int(np.clip(np.searchsorted(V[0, :], v0 - half) - 1, 0, V.shape[1] - 2))
    j_hi = int(np.clip(np.searchsorted(V[0, :], v0 + half) + 1, 1, V.shape[1] - 1))
    Usub = U[i_lo:i_hi, j_lo:j_hi]
    Vsub = V[i_lo:i_hi, j_lo:j_hi]
    Fsub = F_abs[i_lo:i_hi, j_lo:j_hi]
    D = np.sqrt((Usub - u0) ** 2 + (Vsub - v0) ** 2)
    edges = np.linspace(r_min, r_max, n_rings + 1)
    prof = np.full(n_rings, np.inf)
    for k in range(n_rings):
        ring = (D >= edges[k]) & (D < edges[k + 1])
        if ring.any():
            prof[k] = float(Fsub[ring].max())
    kmin = int(np.argmin(prof))
    r_null = 0.5 * (edges[kmin] + edges[kmin + 1])
    return float(np.clip(1.15 * r_null, r_min, r_max))


def evaluate_pattern(F, u_axis, v_axis, targets_uv, thetas_deg=None, r_mode="adaptive",
                     r_fixed=0.036, r_min=0.034, r_max_abs=0.043, r_sep_factor=0.55,
                     n_rings=14):
    """Metrics for one pattern against one beam task.

    targets_uv: list of (u0, v0); thetas_deg: target theta per beam (for the
    relative pointing-error convention). r_mode "adaptive" uses first-null
    radii clipped to [r_min, min(r_sep_factor x min pairwise u-separation,
    r_max_abs)]; "fixed" uses r_fixed for all beams.
    """
    F_abs = np.abs(np.asarray(F))
    U, V = np.meshgrid(u_axis, v_axis, indexing="ij")
    visible = (U ** 2 + V ** 2) <= 1.0
    du = float(u_axis[1] - u_axis[0])
    targets = [(float(u), float(v)) for (u, v) in targets_uv]
    if thetas_deg is None:
        thetas = [float(np.rad2deg(np.arcsin(min(1.0, np.hypot(u, v))))) for (u, v) in targets]
    else:
        thetas = [float(t) for t in thetas_deg]

    if len(targets) > 1:
        seps = [np.hypot(a[0] - b[0], a[1] - b[1])
                for i, a in enumerate(targets) for b in targets[i + 1:]]
        min_sep_u = float(min(seps))
    else:
        min_sep_u = 1.0
    r_max = float(min(r_sep_factor * min_sep_u, r_max_abs))

    if r_mode == "adaptive":
        radii = [estimate_mainlobe_radius(F_abs, U, V, u0, v0, r_min, r_max, n_rings)
                 for (u0, v0) in targets]
    else:
        radii = [float(r_fixed)] * len(targets)

    main = np.zeros_like(visible)
    for (u0, v0), r in zip(targets, radii):
        main |= ((U - u0) ** 2 + (V - v0) ** 2) <= r ** 2
    sl = visible & ~main
    peak_global = float(F_abs.max())
    sll_db = float(20.0 * np.log10(F_abs[sl].max() / peak_global)) if sl.any() else float("nan")

    per_beam = []
    for (u0, v0), r, theta_t in zip(targets, radii, thetas):
        disk = ((U - u0) ** 2 + (V - v0) ** 2) <= r ** 2
        idx = np.unravel_index(int(np.argmax(np.where(disk, F_abs, -np.inf))), F_abs.shape)
        iu = int(np.clip(idx[0], 1, F_abs.shape[0] - 2))
        iv = int(np.clip(idx[1], 1, F_abs.shape[1] - 2))
        logm = np.log(F_abs[iu - 1:iu + 2, iv - 1:iv + 2] + 1e-300)
        du_off = _vertex_offset(logm[0, 1], logm[1, 1], logm[2, 1])
        dv_off = _vertex_offset(logm[1, 0], logm[1, 1], logm[1, 2])
        u_pk = float(u_axis[iu] + du_off * du)
        v_pk = float(v_axis[iv] + dv_off * du)
        err = _sep_uv_deg(u_pk, v_pk, u0, v0)
        per_beam.append(dict(
            u_pk=u_pk, v_pk=v_pk, peak=float(F_abs[iu, iv]),
            err_deg=err, err_pct=100.0 * err / max(theta_t, 1e-9),
            grid_err=float(np.hypot(u_pk - u0, v_pk - v0)) / du,
        ))

    peaks = np.array([b["peak"] for b in per_beam])
    errs = np.array([b["err_deg"] for b in per_beam])
    pcts = np.array([b["err_pct"] for b in per_beam])
    rel = 20.0 * np.log10(peaks / peak_global)
    return dict(
        sll_db=sll_db,
        point_err_max_deg=float(errs.max()),
        point_err_mean_deg=float(errs.mean()),
        point_err_max_pct=float(pcts.max()),
        gain_consistency_db=float(20.0 * np.log10(peaks.max() / peaks.min())),
        mean_peak_rel_db=float(rel.mean()),
        radii=[float(r) for r in radii],
        per_beam=per_beam,
    )

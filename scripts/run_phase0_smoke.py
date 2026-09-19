"""Phase 0 smoke test: physics kernel sanity, numpy/torch parity, GPU check, figures."""

import os

os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metasurf.physics import (
    angles_to_uv,
    uv_to_angles,
    fft_uv_axes,
    bp_phase_plane,
    point_feed_geometry,
    taylor_window_1d,
    taylor_window_2d,
    aperture_to_pattern_np,
    aperture_to_pattern_torch,
)

N = 64
N_PAD = 512
D_OVER_LAMBDA = 0.5
SLL_DB = -25.0
NBAR = 5
SEED = 42
R_MASK_TAYLOR = 0.05
R_MASK_UNIFORM = 0.036

OUT_DIR = ROOT / "results" / "phase0_smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)

checks = []
infos = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print("[{}] {} | {}".format("PASS" if ok else "FAIL", name, detail))


def info(name, value):
    infos.append((name, str(value)))
    print("[INFO] {} | {}".format(name, value))


def angular_sep_deg(t1, p1, t2, p2):
    t1, p1, t2, p2 = (np.deg2rad(float(x)) for x in (t1, p1, t2, p2))
    c = np.cos(t1) * np.cos(t2) + np.sin(t1) * np.sin(t2) * np.cos(p1 - p2)
    return float(np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0))))


def peak_uv(F, u_axis):
    idx = np.unravel_index(np.argmax(np.abs(F)), F.shape)
    return float(u_axis[idx[0]]), float(u_axis[idx[1]])


def sidelobe_db(F, u_axis, v_axis, targets, r_mask):
    U, V = np.meshgrid(u_axis, v_axis, indexing="ij")
    visible = U ** 2 + V ** 2 <= 1.0
    main = np.zeros_like(visible)
    for (u0, v0) in targets:
        main = main | ((U - u0) ** 2 + (V - v0) ** 2 <= r_mask ** 2)
    sl = visible & ~main
    peak = np.abs(F).max()
    return float(20 * np.log10(np.abs(F)[sl].max() / peak))


def main():
    print("=" * 64)
    print("Phase 0 smoke test  (n={}, n_pad={}, d/lambda={}, Taylor {} dB nbar={})".format(
        N, N_PAD, D_OVER_LAMBDA, SLL_DB, NBAR))
    print("=" * 64)

    rng = np.random.default_rng(SEED)
    u_axis = fft_uv_axes(N_PAD, D_OVER_LAMBDA)
    v_axis = u_axis
    du = float(u_axis[1] - u_axis[0])
    info("FFT grid", "n_pad={}, du={:.6f}, range=[{:.3f},{:.3f}]".format(N_PAD, du, u_axis[0], u_axis[-1]))

    ok = True
    max_err = 0.0
    for _ in range(200):
        th = rng.uniform(0.0, 89.0)
        ph = rng.uniform(0.0, 360.0)
        u, v = angles_to_uv(th, ph)
        th2, ph2 = uv_to_angles(u, v)
        dth = abs(float(th2) - th)
        dph = min(abs(float(ph2) - ph), 360.0 - abs(float(ph2) - ph))
        max_err = max(max_err, dth, dph)
        if dth > 1e-8 or dph > 1e-8:
            ok = False
    check("C1 grid angle<->uv roundtrip (200 samples)", ok, "max_err={:.2e} deg".format(max_err))

    beams = []
    while len(beams) < 3:
        th = float(rng.uniform(2.0, 45.0))
        ph = float(rng.uniform(0.0, 360.0))
        if all(angular_sep_deg(th, ph, bt, bp) >= 5.0 for bt, bp in beams):
            beams.append((th, ph))
    info("beam set", "; ".join("({:.2f} deg, {:.2f} deg)".format(*b) for b in beams))

    for i, (th, ph) in enumerate(beams):
        u0, v0 = angles_to_uv(th, ph)
        phase = bp_phase_plane(th, ph, N, D_OVER_LAMBDA)
        F = aperture_to_pattern_np(np.exp(1j * phase), N_PAD)
        up, vp = peak_uv(F, u_axis)
        err = float(np.hypot(up - u0, vp - v0))
        check("C2-{} geometry A beam {} peak position".format(i + 1, i + 1),
              err <= 1.5 * du,
              "target=({:.4f},{:.4f}) peak=({:.4f},{:.4f}) err={:.5f} (tol={:.5f})".format(
                  u0, v0, up, vp, err, 1.5 * du))

    th, ph = beams[0]
    u0, v0 = angles_to_uv(th, ph)
    bp, feed = point_feed_geometry(th, ph, N, D_OVER_LAMBDA, f_over_D=0.8)
    F_b = aperture_to_pattern_np(np.exp(1j * (bp + feed)), N_PAD)
    up, vp = peak_uv(F_b, u_axis)
    err = float(np.hypot(up - u0, vp - v0))
    check("C3 geometry B point-feed beam peak", err <= 1.5 * du,
          "target=({:.4f},{:.4f}) peak=({:.4f},{:.4f}) err={:.5f}".format(u0, v0, up, vp, err))

    phase = bp_phase_plane(beams[1][0], beams[1][1], N, D_OVER_LAMBDA)
    a_np = np.exp(1j * phase)
    F_np = aperture_to_pattern_np(a_np, N_PAD)
    F_t = aperture_to_pattern_torch(torch.from_numpy(a_np).to(torch.complex64), N_PAD).numpy()
    rel = float(np.linalg.norm(np.abs(F_t) - np.abs(F_np)) / np.linalg.norm(np.abs(F_np)))
    check("C4 numpy/torch |F| parity", rel < 1e-5, "rel_l2={:.3e}".format(rel))

    w1 = taylor_window_1d(N, SLL_DB, NBAR)
    w2 = taylor_window_2d(N, SLL_DB, NBAR)
    edge_db = float(20 * np.log10(w1[0] / w1.max()))
    info("Taylor 1D edge taper", "{:.2f} dB (sll={} dB, nbar={})".format(edge_db, SLL_DB, NBAR))

    ideal_list = []
    for (th, ph) in beams:
        u0, v0 = angles_to_uv(th, ph)
        phase = bp_phase_plane(th, ph, N, D_OVER_LAMBDA)
        F_i = aperture_to_pattern_np(w2 * np.exp(1j * phase), N_PAD)
        sll = sidelobe_db(F_i, u_axis, v_axis, [(u0, v0)], R_MASK_TAYLOR)
        info("ideal amp+phase beam ({:.1f},{:.1f}) SLL".format(th, ph), "{:.2f} dB".format(sll))
        ideal_list.append(F_i)

    th, ph = beams[0]
    u0, v0 = angles_to_uv(th, ph)
    phase = bp_phase_plane(th, ph, N, D_OVER_LAMBDA)
    F_p = aperture_to_pattern_np(np.exp(1j * phase), N_PAD)
    sll_p = sidelobe_db(F_p, u_axis, v_axis, [(u0, v0)], R_MASK_UNIFORM)
    info("phase-only (amp=1) beam ({:.1f},{:.1f}) SLL".format(th, ph), "{:.2f} dB".format(sll_p))

    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        dev = torch.device("cuda")
        phi = torch.zeros(4, N, N, device=dev, dtype=torch.float32, requires_grad=True)
        amp = torch.ones(4, N, N, device=dev)
        z = torch.polar(amp, phi)
        F = aperture_to_pattern_torch(z, N_PAD)
        loss = F.abs().pow(2).mean()
        loss.backward()
        gmax = float(phi.grad.abs().max())
        gok = phi.grad is not None and torch.isfinite(phi.grad).all().item() and gmax > 0
        check("C6 GPU forward/backward", bool(gok),
              "device={}, loss={:.3f}, grad_max={:.3e}".format(
                  torch.cuda.get_device_name(0), float(loss), gmax))
    else:
        check("C6 GPU forward/backward", False, "CUDA not available")

    phases = rng.uniform(0, 2 * np.pi, (32, N, N))
    a_np32 = np.exp(1j * phases)
    t0 = time.perf_counter()
    for _ in range(10):
        aperture_to_pattern_np(a_np32, N_PAD)
    t_np = (time.perf_counter() - t0) / 10
    info("numpy AF batch-32 time", "{:.2f} ms/call".format(t_np * 1e3))
    if cuda_ok:
        a_t32 = torch.from_numpy(a_np32.astype(np.complex64)).to(dev)
        for _ in range(3):
            aperture_to_pattern_torch(a_t32, N_PAD)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            aperture_to_pattern_torch(a_t32, N_PAD)
        torch.cuda.synchronize()
        t_t = (time.perf_counter() - t0) / 20
        info("torch GPU AF batch-32 time", "{:.2f} ms/call".format(t_t * 1e3))

    try:
        import yaml  # noqa: F401
        info("pyyaml available", "yes")
    except ImportError:
        info("pyyaml available", "no (phase 1 configs will need it or switch to .py config)")

    sum_ap = np.zeros((N, N), dtype=complex)
    for (th, ph) in beams:
        sum_ap = sum_ap + w2 * np.exp(1j * bp_phase_plane(th, ph, N, D_OVER_LAMBDA))
    F_sum = aperture_to_pattern_np(sum_ap, N_PAD)
    targets = [angles_to_uv(th, ph) for th, ph in beams]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    panels = [
        (ideal_list[0], [targets[0]], "ideal amp+phase beam 1"),
        (ideal_list[1], [targets[1]], "ideal amp+phase beam 2"),
        (ideal_list[2], [targets[2]], "ideal amp+phase beam 3"),
        (F_sum, targets, "3-beam windowed arg-sum (baseline-2 preview)"),
    ]
    U, V = np.meshgrid(u_axis, v_axis, indexing="ij")
    visible = U ** 2 + V ** 2 <= 1.0
    for ax, (F, tg, title) in zip(axes.flat, panels):
        Fdb = 20 * np.log10(np.abs(F) / np.abs(F).max() + 1e-12)
        Fdb = np.where(visible, Fdb, np.nan)
        im = ax.imshow(Fdb.T, extent=[u_axis[0], u_axis[-1], v_axis[0], v_axis[-1]],
                       origin="lower", vmin=-40, vmax=0, cmap="viridis")
        for (u0, v0) in tg:
            ax.plot(u0, v0, "w+", markersize=10, markeredgewidth=1.5)
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="w", linewidth=0.8, linestyle="--"))
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, label="dB")
    fig.suptitle("Phase 0 smoke: BP+Taylor single beams and 3-beam arg-sum (64x64, d=0.5 lambda)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sanity_patterns.png", dpi=150)
    plt.close(fig)

    n_pass = sum(1 for _, ok_, _ in checks if ok_)
    lines = [
        "# Phase 0 smoke report", "",
        "date: {}".format(time.strftime("%Y-%m-%d %H:%M:%S")), "",
        "python: {} | torch: {} | cuda: {}".format(sys.version.split()[0], torch.__version__, cuda_ok), "",
        "## Checks", "",
        "| check | result | detail |", "|---|---|---|",
    ]
    for name, ok_, detail in checks:
        lines.append("| {} | {} | {} |".format(name, "PASS" if ok_ else "FAIL", detail))
    lines += ["", "## Informational", "", "| item | value |", "|---|---|"]
    for name, val in infos:
        lines.append("| {} | {} |".format(name, val))
    lines += [
        "",
        "Beam set: " + "; ".join("({:.2f} deg, {:.2f} deg)".format(*b) for b in beams),
        "",
        "Note: fixed mainlobe mask radii were used per aperture type "
        "(Taylor-designed 0.05, uniform/phase-only 0.036). A single fixed radius cannot serve "
        "both mainlobe widths (Taylor first null ~0.042 vs uniform ~0.031 in u); "
        "phase-1 metrics.py will adopt a first-null-based adaptive mask.",
        "",
    ]
    (OUT_DIR / "smoke_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("-" * 64)
    print("smoke summary: {}/{} checks passed".format(n_pass, len(checks)))
    print("outputs: {}".format(OUT_DIR))
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())

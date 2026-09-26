"""IFT (iterative Fourier transform) phase-only multi-beam synthesis baseline.

Alternating projection between the phase-only aperture set and a desired-
pattern set (plan v2.0 sec 8.8):

    init   a = arg-sum aperture (same starting point as the network)
    target F_des = FFT(sum_i A_i e^{j phi_i})     (ideal amp+phase reference)
    loop   F = FFT2(a)                            (512^2 zero-padded)
           mainlobe disks: F <- F_des (beam-anchored: ideal-shaped, peaks
                           equalized by fixed constants) -- beams cannot die
           sidelobe region: hard-threshold |F| at an annealed level
           a = iFFT2(F); blend with previous; project to unit modulus

Fully batched over tasks (task dimension), runs on NPU/GPU/CPU.
"""

import math

import numpy as np
import torch

from ..physics.grid import fft_uv_axes


def _pad(z, p):
    re = torch.nn.functional.pad(z.real, (p, p, p, p))
    im = torch.nn.functional.pad(z.imag, (p, p, p, p))
    return torch.complex(re, im)


def ift_synth(phases, tgt_uv, window2d=None, n_pad=512, d_ol=0.5, n_iter=300,
              r_disk=0.04, thr_start_db=-12.0, thr_end_db=-25.0, blend=0.5,
              shrink=0.85, return_history=False):
    """Batched IFT synthesis: point-anchored mainlobes + soft sidelobe shrink.

    phases: [B, M, n, n] BP phases (rad) per task/beam.
    tgt_uv: [B, M, 2] target (u, v) per beam; tgt_idx: [B, M, 2] grid indices.
    window2d: [n, n] amplitude window (Taylor); None -> uniform.
    Mainlobe disks are left FREE (no amplitude taper imposed); only the exact
    target grid points are anchored to the equalized ideal values; sidelobes
    are softly attenuated (x shrink per iteration, level-annealed hard cap as
    safety). Best iterate tracked by a balance-gated score.
    Returns unit-modulus aperture [B, n, n] complex (input device).
    """
    dev = phases.device
    B, M, n, _ = phases.shape
    u_axis = fft_uv_axes(n_pad, d_ol)
    U = torch.from_numpy(u_axis[:, None].astype(np.float32)).to(dev)
    V = torch.from_numpy(u_axis[None, :].astype(np.float32)).to(dev)
    visible = (U ** 2 + V ** 2) <= 1.0
    p = (n_pad - n) // 2

    def nearest_idx(uv):
        iu = torch.argmin(torch.abs(u_axis_t.view(1, -1)
                                    - uv[..., 0].unsqueeze(-1)), dim=-1)
        iv = torch.argmin(torch.abs(u_axis_t.view(1, -1)
                                    - uv[..., 1].unsqueeze(-1)), dim=-1)
        return iu, iv

    u_axis_t = torch.from_numpy(u_axis).to(dev)

    a = torch.zeros(B, n, n, dtype=torch.complex64, device=dev)
    a_des = torch.zeros(B, n, n, dtype=torch.complex64, device=dev)
    if window2d is not None and not torch.is_tensor(window2d):
        window2d = torch.from_numpy(np.asarray(window2d))
    w = None if window2d is None else window2d.to(dev, torch.float32)
    for i in range(M):
        e = torch.exp(1j * phases[:, i])
        a = a + e
        a_des = a_des + (w if w is not None else 1.0) * e
    a = a / (a.abs() + 1e-12)

    def _fwd(a_pad):
        # standard u-v convention, matches aperture_to_pattern_np
        return torch.conj(torch.fft.fftshift(
            torch.fft.fft2(torch.conj(a_pad)), dim=(-2, -1)))

    def _inv(F):
        # inverse of _fwd: a_pad = conj(ifft2(ifftshift(conj(F))))
        return torch.conj(torch.fft.ifft2(
            torch.fft.ifftshift(torch.conj(F), dim=(-2, -1)), dim=(-2, -1)))

    F_des = _fwd(_pad(a_des, p))

    disks = []
    for i in range(M):
        u0 = tgt_uv[:, i, 0].view(-1, 1, 1)
        v0 = tgt_uv[:, i, 1].view(-1, 1, 1)
        disks.append((U[None] - u0) ** 2 + (V[None] - v0) ** 2 <= r_disk ** 2)
    main = torch.stack(disks, dim=0).any(dim=0)
    sl = visible[None] & ~main

    # per-beam target grid indices (anchors equalize the CURRENT target-point
    # responses each iteration: keep phase, scale magnitudes to the mean)
    G_des = torch.stack(
        [(F_des.abs() * disks[i]).flatten(1).amax(dim=1) for i in range(M)],
        dim=1)
    Gbar = G_des.mean(dim=1).view(-1, 1, 1) + 1e-12
    bidx = torch.arange(B, device=dev)
    iu_all, iv_all = [], []
    for i in range(M):
        iu, iv = nearest_idx(tgt_uv[:, i, :])
        iu_all.append(iu)
        iv_all.append(iv)
    iu_all = torch.stack(iu_all, dim=1)                # [B, M]
    iv_all = torch.stack(iv_all, dim=1)

    best_a = a.clone()
    best_score = torch.full((B,), -1e9, device=dev)
    hist = []
    for k in range(n_iter):
        frac = k / max(1, n_iter - 1)
        thr_db = thr_start_db + 0.5 * (thr_end_db - thr_start_db) \
            * (1.0 + math.cos(math.pi * (1.0 - frac)))
        F = _fwd(_pad(a, p))
        F = torch.where(sl, F * shrink, F)
        # disk-region equalization at CURRENT scale (whole mainlobe, keeps
        # phase structure; prevents the weak beam's mainlobe body decaying)
        Gs_cur = torch.stack(
            [(F.abs() * disks[i]).flatten(1).amax(dim=1) for i in range(M)],
            dim=1)                                        # [B, M]
        Gbar_cur = Gs_cur.mean(dim=1).view(-1, 1, 1) + 1e-12
        for i in range(M):
            Gi = (F.abs() * disks[i]).flatten(1).amax(dim=1).view(-1, 1, 1)
            F = torch.where(disks[i], F * (Gbar_cur / (Gi + 1e-12)), F)
        G_cur = Gs_cur.amax(dim=1)
        thr = G_cur.view(B, 1, 1) * (10.0 ** (thr_db / 20.0))
        over = (F.abs() > thr) & sl
        F = torch.where(over, F * (thr / (F.abs() + 1e-12)), F)
        a_new = _inv(F)[:, p:p + n, p:p + n]
        a = (1.0 - blend) * a + blend * a_new
        a = a / (a.abs() + 1e-12)

        # balance-gated best-iterate tracking
        Fa = _fwd(_pad(a, p))
        Fa_abs = Fa.abs()
        sl_max = (Fa_abs * sl).flatten(1).amax(dim=1) / Gbar.view(B)
        sll = 20.0 * torch.log10(sl_max + 1e-12)
        Gs = torch.stack([(Fa_abs * disks[i]).flatten(1).amax(dim=1)
                          for i in range(M)], dim=1)
        g_hat = Gs / (Gs.amax(dim=1, keepdim=True) + 1e-12)
        cons = 20.0 * torch.log10(1.0 / g_hat.amin(dim=1).clamp(min=1e-6))
        score = sll - torch.clamp(cons - 1.5, min=0.0) * 5.0
        better = score > best_score
        best_score = torch.where(better, score, best_score)
        best_a = torch.where(better.view(-1, 1, 1), a, best_a)
        if return_history:
            hist.append((float(sll.mean()), float(cons.mean())))
    if return_history:
        return best_a, hist
    return best_a

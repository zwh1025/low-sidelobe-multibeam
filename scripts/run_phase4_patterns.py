"""Representative beam-pattern gallery across beam counts (phase 4).

For each M in --mlist: pick the median-SLL network task from the first
--n_pick test tasks (seed 819), then render
  (a) beam_patterns.png : 2D u-v patterns, rows = M, cols =
      [LCS net / per-task direct opt / arg-sum / ideal ref], targets marked,
      per-panel SLL annotated;
  (b) beam_cuts.png     : radial cuts through every beam direction
      (angle offset from beam center), all beams thin + mean bold, per method.

Usage: python scripts/run_phase4_patterns.py [--run r2e] [--mlist 2,4,8,16,32]
"""

import os

os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

import argparse
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
    angles_to_uv, fft_uv_axes, taylor_window_2d, aperture_to_pattern_np,
)
from metasurf.physics.bp_phase import bp_phase_plane
from metasurf.physics.metrics import evaluate_pattern
from metasurf.model.lcs import LCSNet
from metasurf.train.trainer import pregenerate_multi
from metasurf.train.losses import PhysicsLoss
from metasurf.eval.visualize import draw_pattern_db, pattern_db

N, N_PAD, D_OL = 64, 512, 0.5
SLL_DB, NBAR = -25.0, 5
TEST_SEED = 819
THETA_MIN, THETA_MAX, MIN_SEP = 2.0, 45.0, 5.0
OUT = ROOT / "results" / "phase4_network"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            return "npu"
    except ImportError:
        pass
    return "cpu"


def radial_cut(Fdb, u_axis, tgt):
    u0, v0 = tgt
    az = np.arctan2(v0, u0)
    t = np.linspace(-0.995, 0.995, 512)
    uu, vv = t * np.cos(az), t * np.sin(az)
    iu = np.argmin(np.abs(u_axis[:, None] - uu[None, :]), axis=0)
    iv = np.argmin(np.abs(u_axis[:, None] - vv[None, :]), axis=0)
    vals = Fdb[iu, iv]
    th0 = np.degrees(np.arcsin(min(1.0, np.hypot(u0, v0))))
    ang = np.degrees(np.arcsin(np.clip(t, -1, 1))) - th0
    return ang, vals


def direct_optimize(phases_np, tgt_uv_np, tgt_idx_np, W2, device, steps=400):
    """Per-task direct phase optimization (A3b protocol) for one task."""
    s = np.exp(1j * phases_np).sum(axis=0)
    phi0 = np.angle(s / (np.abs(s) + 1e-12))
    loss_fn = PhysicsLoss(n=N, n_pad=N_PAD, d_ol=D_OL, r_train=0.040,
                           alpha=1.0, beta=1.0, gamma=1.0, gamma2=50.0,
                           use_pat=False, delta=0.0, sll_domain="dB",
                           tau_start=1.0, tau_end=0.1, l_beam_mode="hinge",
                           beam_floor=0.95, dir2=30.0, dir_floor=0.05,
                           sll_floor_db=-35.0, window2d=W2).to(device)
    phi = torch.from_numpy(phi0.astype(np.float32)).to(device) \
        .unsqueeze(0).requires_grad_(True)
    uv = torch.tensor(tgt_uv_np, dtype=torch.float32).unsqueeze(0).to(device)
    idx = torch.tensor(tgt_idx_np, dtype=torch.int64).unsqueeze(0).to(device)
    opt = torch.optim.Adam([phi], lr=0.05)
    for k in range(steps):
        loss, _ = loss_fn(torch.exp(1j * phi), None, uv, idx, k / steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.exp(1j * phi)[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r2e")
    ap.add_argument("--ckpt", default="ckpt_best.pt")
    ap.add_argument("--mlist", default="2,4,8,16,32")
    ap.add_argument("--n_pick", type=int, default=100)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    device = pick_device()
    u_axis = fft_uv_axes(N_PAD, D_OL)
    W2 = taylor_window_2d(N, SLL_DB, NBAR)
    ck = torch.load(OUT / args.run / args.ckpt, map_location=device,
                    weights_only=False)
    cfg = ck["cfg"]
    model = LCSNet(c_per_beam=3, c_enc=cfg.get("c_enc", 64), width=cfg["width"],
                   variant=cfg["variant"], m_max=cfg.get("m_max", 8),
                   dec_norm=cfg.get("dec_norm", "bn")).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    bn_stats = None
    stats_path = OUT / args.run / "bn_stats_by_m.pt"
    if stats_path.exists():
        st = torch.load(stats_path, map_location=device, weights_only=False)
        bn_stats = st["stats_by_m"]

    m_list = [int(v) for v in args.mlist.split(",")]
    picks = {}
    t0 = time.time()
    for m in m_list:
        Xte, uvte, idxte, _, beams = pregenerate_multi(
            np.random.default_rng(TEST_SEED), [m], args.n_pick, W2, True, N,
            D_OL, N_PAD, THETA_MIN, THETA_MAX, MIN_SEP, m_max=max(m, 8))
        if bn_stats is not None:
            sd = model.state_dict()
            for k, v in bn_stats[min(m, max(bn_stats))].items():
                sd[k] = v
            model.load_state_dict(sd)
        aps = []
        with torch.no_grad():
            for s in range(0, args.n_pick, 32):
                aps.append(model(Xte[s:s + 32, :3 * m].to(device)).cpu())
        aps = torch.cat(aps).numpy()
        slls = []
        for k in range(args.n_pick):
            F = aperture_to_pattern_np(aps[k], N_PAD)
            r = evaluate_pattern(F, u_axis, u_axis,
                                 [angles_to_uv(t, p) for t, p in beams[k]])
            slls.append(r["sll_db"])
        med = int(np.argsort(slls)[len(slls) // 2])
        picks[m] = dict(beams=beams[med], ap_net=aps[med],
                        uv=uvte[med].numpy(), idx=idxte[med].numpy(),
                        sll_net=slls[med])
        print("[M={:2d}] median task {} (net SLL {:.2f} dB)".format(
            m, med, slls[med]), flush=True)

    # per-task baselines + direct optimization
    for m, d in picks.items():
        beams = d["beams"]
        phases = np.stack([bp_phase_plane(t, p, N, D_OL) for t, p in beams])
        d["ap_argsum"] = np.exp(1j * np.angle(np.exp(1j * phases).sum(axis=0)))
        d["ap_ideal"] = (W2[None] * np.exp(1j * phases)).sum(axis=0)
        d["ap_direct"] = direct_optimize(phases, d["uv"][:m], d["idx"][:m],
                                         W2, device, steps=args.steps)
        d["F"] = {}
        for name in ("net", "direct", "argsum", "ideal"):
            d["F"][name] = aperture_to_pattern_np(d["ap_" + name], N_PAD)
            r = evaluate_pattern(d["F"][name], u_axis, u_axis,
                                 [angles_to_uv(t, p) for t, p in beams],
                                 [t for t, p in beams])
            d["sll_" + name] = r["sll_db"]
        print("[M={:2d}] SLL net {:.2f} | direct {:.2f} | argsum {:.2f} | "
              "ideal {:.2f}".format(m, d["sll_net"], d["sll_direct"],
                                    d["sll_argsum"], d["sll_ideal"]),
              flush=True)

    # figure 1: 2D pattern gallery
    methods = [("net", "LCS net"), ("direct", "per-task direct opt"),
               ("argsum", "arg-sum (Du cont.)"), ("ideal", "ideal amp+phase")]
    fig, axes = plt.subplots(len(m_list), len(methods),
                             figsize=(3.1 * len(methods), 3.0 * len(m_list)))
    for r, m in enumerate(m_list):
        d = picks[m]
        tg = [angles_to_uv(t, p) for t, p in d["beams"]]
        for c, (name, lab) in enumerate(methods):
            ax = axes[r, c]
            draw_pattern_db(ax, d["F"][name], u_axis, u_axis, tg,
                            vmin=-40.0, colorbar=False)
            if r == 0:
                ax.set_title(lab, fontsize=10)
            if c == 0:
                ax.set_ylabel("M={}\nv".format(m), fontsize=10)
            ax.set_xlabel("")
            ax.text(0.03, 0.05, "SLL {:.1f} dB".format(d["sll_" + name]),
                    transform=ax.transAxes, color="w", fontsize=8,
                    va="bottom")
    fig.suptitle("Representative beam patterns (median-SLL task per M, "
                 "test seed 819)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.colorbar(axes[0, 0].images[0], ax=axes, shrink=0.6, label="dB")
    fig.savefig(OUT / "beam_patterns.png", dpi=130)
    plt.close(fig)

    # figure 2: radial cuts through every beam
    fig, axes = plt.subplots(1, len(m_list), figsize=(3.6 * len(m_list), 3.4),
                             sharey=True)
    styles = {"net": ("tab:purple", 0.9, "-"),
              "direct": ("tab:orange", 0.8, "--"),
              "argsum": ("tab:gray", 0.5, ":"),
              "ideal": ("tab:green", 0.5, ":")}
    for c, m in enumerate(m_list):
        ax = axes[c]
        d = picks[m]
        tg = [angles_to_uv(t, p) for t, p in d["beams"]]
        for name, (col, lw, ls) in styles.items():
            Fdb = pattern_db(d["F"][name])
            cuts = [radial_cut(Fdb, u_axis, t)[1] for t in tg]
            ang, _ = radial_cut(Fdb, u_axis, tg[0])
            for v in cuts:
                ax.plot(ang, v, color=col, alpha=0.18, lw=0.7, ls=ls,
                        rasterized=True)
            ax.plot(ang, np.mean(cuts, axis=0), color=col, lw=lw, ls=ls,
                    label=methods[[n for n, _ in methods].index(name)][1])
        ax.set_xlim(-60, 60)
        ax.set_ylim(-40, 2)
        ax.set_title("M = {}".format(m), fontsize=11)
        ax.set_xlabel("angle offset from beam (deg)")
        if c == 0:
            ax.set_ylabel("relative pattern (dB)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Radial cuts through each beam (thin: per beam, "
                 "bold: mean)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "beam_cuts.png", dpi=130)
    plt.close(fig)

    print("saved -> {} / {} ({:.0f} s)".format(
        OUT / "beam_patterns.png", OUT / "beam_cuts.png",
        time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

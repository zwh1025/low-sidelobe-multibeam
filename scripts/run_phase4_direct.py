"""A3b: per-task direct phase optimization (offline upper-bound reference).

Optimizes each task's aperture phase directly with the SAME differentiable
physics loss as the network (R2e config: dB-LSE + beam/dir hinges + clamp),
arg-sum init, Adam, N steps -- batched over tasks on NPU. This upper-bounds
what per-task iterative optimization achieves on our objective; the network
matching it at 0.35 ms/sample is the "learned the optimizer" claim.

Usage: python scripts/run_phase4_direct.py [--mlist 2,8,16,32] [--n 100] [--steps 400]
"""

import os

os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from metasurf.physics import (
    angles_to_uv, fft_uv_axes, taylor_window_2d, aperture_to_pattern_np,
)
from metasurf.physics.bp_phase import bp_phase_plane
from metasurf.physics.metrics import evaluate_pattern
from metasurf.data.beam_dataset import sample_beams
from metasurf.train.losses import PhysicsLoss

N, N_PAD, D_OL = 64, 512, 0.5
SLL_DB, NBAR = -25.0, 5
TEST_SEED = 819
THETA_MIN, THETA_MAX, MIN_SEP = 2.0, 45.0, 5.0
OUT = ROOT / "results" / "phase4_network" / "ift_baseline"


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


def gen_tasks(m, n_tasks):
    rng = np.random.default_rng(TEST_SEED)
    W2 = taylor_window_2d(N, SLL_DB, NBAR)
    u_axis = fft_uv_axes(N_PAD, D_OL)
    beams_all, tgt_idx, tgt_uv = [], [], []
    inits = []
    for k in range(n_tasks):
        while True:
            try:
                beams = sample_beams(rng, m, THETA_MIN, THETA_MAX, MIN_SEP)
                break
            except RuntimeError:
                continue
        beams_all.append(beams)
        ph = np.stack([bp_phase_plane(t, p, N, D_OL) for t, p in beams])
        s = np.exp(1j * ph).sum(axis=0)
        inits.append(np.angle(s / (np.abs(s) + 1e-12)))
        uv = [angles_to_uv(t, p) for t, p in beams]
        tgt_uv.append(uv)
        tgt_idx.append([[int(np.argmin(np.abs(u_axis - u))), int(np.argmin(np.abs(u_axis - v)))]
                        for (u, v) in uv])
    return (torch.from_numpy(np.stack(inits).astype(np.float32)),
            torch.tensor(np.array(tgt_uv), dtype=torch.float64),
            torch.tensor(np.array(tgt_idx), dtype=torch.int64), beams_all, W2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlist", default="2,8,16,32")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--chunk", type=int, default=25)
    args = ap.parse_args()

    device = pick_device()
    OUT.mkdir(parents=True, exist_ok=True)
    u_axis = fft_uv_axes(N_PAD, D_OL)
    m_list = [int(v) for v in args.mlist.split(",")]

    rows = []
    for m in m_list:
        phi0, tgt_uv, tgt_idx, beams_all, W2 = gen_tasks(m, args.n)
        loss_fn = PhysicsLoss(n=N, n_pad=N_PAD, d_ol=D_OL, r_train=0.040,
                              alpha=1.0, beta=1.0, gamma=1.0, gamma2=50.0,
                              use_pat=False, delta=0.0,
                              sll_domain="dB", tau_start=1.0, tau_end=0.1,
                              l_beam_mode="hinge", beam_floor=0.95,
                              dir2=30.0, dir_floor=0.05, sll_floor_db=-35.0,
                              window2d=W2).to(device)
        t0 = time.time()
        aps = []
        for s in range(0, args.n, args.chunk):
            phi = phi0[s:s + args.chunk].to(device).clone().requires_grad_(True)
            opt = torch.optim.Adam([phi], lr=args.lr)
            uvb = tgt_uv[s:s + args.chunk].to(device)
            idxb = tgt_idx[s:s + args.chunk].to(device)
            for k in range(args.steps):
                frac = k / args.steps
                ap = torch.exp(1j * phi)
                loss, _ = loss_fn(ap, None, uvb, idxb, frac)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                aps.append(torch.exp(1j * phi).detach().cpu())
        elapsed = time.time() - t0
        aps = torch.cat(aps).numpy()
        for k in range(args.n):
            F = aperture_to_pattern_np(aps[k], N_PAD)
            r = evaluate_pattern(
                F, u_axis, u_axis,
                [angles_to_uv(t, p) for t, p in beams_all[k]],
                [t for t, p in beams_all[k]])
            rows.append(dict(m=m, task=k, sll=r["sll_db"],
                             cons=r["gain_consistency_db"],
                             perr=r["point_err_max_deg"],
                             sec=elapsed / args.n))
        sll = np.array([r["sll"] for r in rows if r["m"] == m])
        cons = np.array([r["cons"] for r in rows if r["m"] == m])
        perr = np.array([r["perr"] for r in rows if r["m"] == m])
        print("[M={:2d}] direct-opt SLL {:.2f}+/-{:.2f} | cons {:.2f}/{:.2f} | "
              "perr_max {:.3f} | {:.2f} s/task".format(
                  m, sll.mean(), sll.std(), cons.mean(), cons.max(),
                  perr.max(), elapsed / args.n), flush=True)

    with open(OUT / "metrics_directopt.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["m", "task", "sll", "cons", "perr",
                                          "sec"])
        w.writeheader()
        w.writerows(rows)
    print("saved -> {}".format(OUT / "metrics_directopt.csv"), flush=True)


if __name__ == "__main__":
    main()

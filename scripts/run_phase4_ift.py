"""A3: IFT iterative-synthesis baseline over the phase-4 test tasks.

Runs batched IFT (src/metasurf/baselines/ift.py) for M in --mlist on the
first --n tasks per M of the fixed test set (seed 819, subset of the network
test set). Produces per-task metrics (adaptive masks, same protocol as the
network) + per-task wall time, as the quality upper bound / speed reference.

Usage: python scripts/run_phase4_ift.py [--mlist 2,8,16,32] [--n 100]
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
from metasurf.baselines.ift import ift_synth

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
    """Deterministic task draw matching pregenerate_multi's RNG stream."""
    rng = np.random.default_rng(TEST_SEED)
    W2 = taylor_window_2d(N, SLL_DB, NBAR)
    phases, tgt, beams_all = [], [], []
    for k in range(n_tasks):
        while True:
            try:
                beams = sample_beams(rng, m, THETA_MIN, THETA_MAX, MIN_SEP)
                break
            except RuntimeError:
                continue
        ph = np.stack([bp_phase_plane(t, p, N, D_OL) for t, p in beams])
        phases.append(ph.astype(np.float32))
        tgt.append([angles_to_uv(t, p) for t, p in beams])
        beams_all.append(beams)
    return (torch.from_numpy(np.stack(phases)),
            torch.tensor(np.array(tgt), dtype=torch.float32), beams_all, W2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlist", default="2,8,16,32")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n_iter", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=25)
    args = ap.parse_args()

    device = pick_device()
    OUT.mkdir(parents=True, exist_ok=True)
    u_axis = fft_uv_axes(N_PAD, D_OL)
    m_list = [int(v) for v in args.mlist.split(",")]

    rows = []
    for m in m_list:
        phases, tgt, beams_all, W2 = gen_tasks(m, args.n)
        t0 = time.time()
        aps = []
        with torch.no_grad():
            for s in range(0, args.n, args.chunk):
                aps.append(ift_synth(phases[s:s + args.chunk].to(device),
                                     tgt[s:s + args.chunk].to(device),
                                     window2d=torch.from_numpy(W2).to(device),
                                     n_pad=N_PAD, d_ol=D_OL,
                                     n_iter=args.n_iter).cpu())
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
        print("[M={:2d}] IFT SLL {:.2f}+/-{:.2f} | cons {:.2f}/{:.2f} | "
              "perr_max {:.3f} | {:.2f} s/task".format(
                  m, sll.mean(), sll.std(), cons.mean(), cons.max(),
                  perr.max(), elapsed / args.n), flush=True)

    with open(OUT / "metrics_ift.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["m", "task", "sll", "cons", "perr",
                                          "sec"])
        w.writeheader()
        w.writerows(rows)
    print("saved -> {}".format(OUT / "metrics_ift.csv"), flush=True)


if __name__ == "__main__":
    main()

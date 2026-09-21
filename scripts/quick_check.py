"""Quick test-set readout for a single phase-3 run (same adaptive metrics as run_phase3_eval.py).

Usage: python scripts/quick_check.py [--run main] [--ckpt ckpt_best.pt]
Prints SLL / gain consistency / pointing error on the fixed held-out test set (seed 819).
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

from metasurf.physics import (
    angles_to_uv, fft_uv_axes, taylor_window_2d, aperture_to_pattern_np,
)
from metasurf.physics.metrics import evaluate_pattern
from metasurf.model.unet import ArgSumResidualNet
from metasurf.train.trainer import pregenerate

N, N_PAD, D_OL = 64, 512, 0.5
SEED = 42
TEST_SEED = SEED + 777
N_TEST = 500
THETA_MIN, THETA_MAX, MIN_SEP = 2.0, 45.0, 5.0


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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--ckpt", default="ckpt_best.pt")
    args = ap.parse_args()

    device = pick_device()
    out = ROOT / "results" / "phase3_network"
    ck = torch.load(out / args.run / args.ckpt, map_location=device,
                    weights_only=False)
    model = ArgSumResidualNet(in_ch=6 if ck["cfg"]["with_amp"] else 4,
                              width=ck["cfg"]["width"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print("run={} {} epoch={} val_sll={:.2f} dB device={}".format(
        args.run, args.ckpt, ck["epoch"], ck["val_sll"], device), flush=True)

    W2 = taylor_window_2d(N, ck["cfg"]["sll_db"], ck["cfg"]["nbar"])
    Xte, uvte, idxte, beams_te = pregenerate(
        np.random.default_rng(TEST_SEED), N_TEST, W2, True, N, D_OL, N_PAD,
        THETA_MIN, THETA_MAX, MIN_SEP)
    Xr = Xte if ck["cfg"]["with_amp"] else Xte[:, [0, 1, 3, 4]]

    aps = []
    for s in range(0, Xr.shape[0], 100):
        aps.append(model(Xr[s:s + 100].to(device)).cpu())
    aps = torch.cat(aps).numpy()

    u_axis = fft_uv_axes(N_PAD, D_OL)
    rows = []
    t0 = time.time()
    for k in range(N_TEST):
        F = aperture_to_pattern_np(aps[k], N_PAD)
        rows.append(evaluate_pattern(
            F, u_axis, u_axis,
            [angles_to_uv(t, p) for t, p in beams_te[k]],
            [t for t, p in beams_te[k]]))
    sll = np.array([r["sll_db"] for r in rows])
    cons = np.array([r["gain_consistency_db"] for r in rows])
    perr = np.array([r["point_err_max_deg"] for r in rows])
    print("SLL       {:.2f} +/- {:.2f} dB   (arg-sum baseline -9.70, ideal -24.14)".format(
        sll.mean(), sll.std()))
    print("gaincons  mean {:.2f} / max {:.2f} dB   (target <= 1)".format(
        cons.mean(), cons.max()))
    print("pointerr  mean {:.3f} / max {:.3f} deg   (target <= 0.5)".format(
        perr.mean(), perr.max()))
    print("imbalance tasks (cons > 3 dB): {}/{}".format(int((cons > 3).sum()),
                                                        N_TEST))
    print("done in {:.0f} s".format(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

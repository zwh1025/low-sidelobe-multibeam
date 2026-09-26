"""Quick test-set readout for a single run (same adaptive metrics as full eval).

Usage: python scripts/quick_check.py [--run main] [--ckpt ckpt_best.pt]
                                     [--root results/phase3_network] [--m 2]

--root selects phase3 (ArgSumResidualNet) or phase4 (LCSNet) runs; --m picks
the test beam count (seed 819; 500 tasks for M=2, 200 otherwise).
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
from metasurf.model.lcs import LCSNet
from metasurf.train.trainer import pregenerate, pregenerate_multi

N, N_PAD, D_OL = 64, 512, 0.5
SEED = 42
TEST_SEED = SEED + 777
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


def select_channels(X, m, with_amp):
    if with_amp:
        return X[:, :3 * m]
    idx = [j for i in range(m) for j in (3 * i, 3 * i + 1)]
    return X[:, idx]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--ckpt", default="ckpt_best.pt")
    ap.add_argument("--root", default="results/phase3_network")
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--geometry", choices=["A", "B"], default=None,
                    help="override ckpt geometry (default: ckpt cfg)")
    ap.add_argument("--quantize", choices=["none", "ste1"], default=None,
                    help="override ckpt quantization head (default: ckpt cfg)")
    ap.add_argument("--bnstats", action="store_true",
                    help="load per-M BN stats (bn_stats_by_m.pt) if present "
                         "and select by test M (M>m_max uses m_max's)")
    args = ap.parse_args()

    device = pick_device()
    out = ROOT / args.root
    ck = torch.load(out / args.run / args.ckpt, map_location=device,
                    weights_only=False)
    cfg = ck["cfg"]
    is_phase4 = "variant" in cfg
    geometry = args.geometry or cfg.get("geometry", "A")
    quantize = args.quantize or cfg.get("quantize", "none")
    if is_phase4:
        model = LCSNet(c_per_beam=3 if cfg["with_amp"] else 2,
                       c_enc=cfg.get("c_enc", 64), width=cfg["width"],
                       variant=cfg["variant"],
                       m_max=cfg.get("m_max", 8),
                       dec_norm=cfg.get("dec_norm", "bn"),
                       quantize=quantize).to(device)
    else:
        model = ArgSumResidualNet(in_ch=6 if cfg["with_amp"] else 4,
                                  width=cfg["width"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    if args.bnstats and is_phase4:
        stats_path = out / args.run / "bn_stats_by_m.pt"
        if stats_path.exists():
            st = torch.load(stats_path, map_location=device, weights_only=False)
            key = min(args.m, max(st["stats_by_m"]))
            sd = model.state_dict()
            for k, v in st["stats_by_m"][key].items():
                sd[k] = v
            model.load_state_dict(sd)
            print("bnstats: loaded M={} stats (from ckpt {})".format(
                key, st.get("ckpt")), flush=True)
        else:
            print("bnstats: {} not found, using ckpt running stats".format(
                stats_path), flush=True)
    print("root={} run={} {} epoch={} val_sll={:.2f} dB device={} M={}".format(
        args.root, args.run, args.ckpt, ck["epoch"], ck["val_sll"], device,
        args.m), flush=True)

    W2 = taylor_window_2d(N, cfg["sll_db"], cfg["nbar"])
    n_test = 500 if args.m == 2 else 200
    feed = None
    if geometry == "B":
        from metasurf.physics.bp_phase import point_feed_geometry
        feed = point_feed_geometry(10.0, 0.0, N, D_OL, 0.8)[1]
    if is_phase4:
        Xte, uvte, idxte, _, beams_te = pregenerate_multi(
            np.random.default_rng(TEST_SEED), [args.m], n_test, W2, True, N,
            D_OL, N_PAD, THETA_MIN, THETA_MAX, MIN_SEP,
            m_max=max(args.m, cfg.get("m_max", 8)), geometry=geometry)
        Xr = select_channels(Xte, args.m, cfg["with_amp"])
    else:
        Xte, uvte, idxte, beams_te = pregenerate(
            np.random.default_rng(TEST_SEED), n_test, W2, True, N, D_OL,
            N_PAD, THETA_MIN, THETA_MAX, MIN_SEP)
        Xr = Xte if cfg["with_amp"] else Xte[:, [0, 1, 3, 4]]

    aps = []
    for s in range(0, Xr.shape[0], 100):
        aps.append(model(Xr[s:s + 100].to(device)).cpu())
    aps = torch.cat(aps).numpy()

    u_axis = fft_uv_axes(N_PAD, D_OL)
    rows = []
    t0 = time.time()
    for k in range(n_test):
        ap = aps[k] if feed is None else aps[k] * np.exp(1j * feed)
        F = aperture_to_pattern_np(ap, N_PAD)
        rows.append(evaluate_pattern(
            F, u_axis, u_axis,
            [angles_to_uv(t, p) for t, p in beams_te[k]],
            [t for t, p in beams_te[k]]))
    sll = np.array([r["sll_db"] for r in rows])
    cons = np.array([r["gain_consistency_db"] for r in rows])
    perr = np.array([r["point_err_max_deg"] for r in rows])
    print("SLL       {:.2f} +/- {:.2f} dB   (arg-sum baseline -9.70)".format(
        sll.mean(), sll.std()))
    print("gaincons  mean {:.2f} / max {:.2f} dB".format(cons.mean(), cons.max()))
    print("pointerr  mean {:.3f} / max {:.3f} deg".format(perr.mean(), perr.max()))
    print("imbalance tasks (cons > 3 dB): {}/{}".format(int((cons > 3).sum()),
                                                        n_test))
    if geometry == "B":
        from metasurf.baselines.superposition import onebit_code, code_phase
        base = []
        for k in range(n_test):
            phB = np.stack([point_feed_geometry(t, p, N, D_OL, 0.8)[0]
                            for t, p in beams_te[k]])
            ap = np.exp(1j * (code_phase(onebit_code(phB)) + feed))
            base.append(evaluate_pattern(
                aperture_to_pattern_np(ap, N_PAD), u_axis, u_axis,
                [angles_to_uv(t, p) for t, p in beams_te[k]],
                [t for t, p in beams_te[k]]))
        bs = np.array([r["sll_db"] for r in base])
        print("baseline  1-bit Du eq.(6)+geomB: {:.2f} +/- {:.2f} dB "
              "(net improvement {:+.2f} dB)".format(
                  bs.mean(), bs.std(), bs.mean() - sll.mean()))
    print("done in {:.0f} s".format(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

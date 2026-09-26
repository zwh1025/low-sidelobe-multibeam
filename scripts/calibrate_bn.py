"""Per-M BatchNorm calibration for bucketed multi-M models (phase 4).

Under same-M bucketed training the decoder's BN batch statistics are correct
per bucket, but a single running-stat set cannot serve all M (Fsum magnitude
scales ~sqrt(M)). This script calibrates one BN stat set per training M on
held-out calibration tasks and saves them alongside the checkpoint; at
inference the (known) input M selects the stat set (M > m_max uses m_max's).

Usage: python scripts/calibrate_bn.py --run r2b [--ckpt ckpt_last.pt] [--tag ""]
Writes: results/phase4_network/<run>/bn_stats_by_m.pt
"""

import os

os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from metasurf.physics.taylor import taylor_window_2d
from metasurf.model.lcs import LCSNet
from metasurf.train.trainer import pregenerate_multi

N, N_PAD, D_OL = 64, 512, 0.5


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


def stat_keys(sd):
    return [k for k in sd if "running" in k or "num_batches" in k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="ckpt_last.pt")
    ap.add_argument("--m_list", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--n_cal", type=int, default=128)
    ap.add_argument("--seed0", type=int, default=3000)
    args = ap.parse_args()

    device = pick_device()
    out = ROOT / "results" / "phase4_network" / args.run
    ck = torch.load(out / args.ckpt, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    m_max = cfg.get("m_max", 8)
    model = LCSNet(c_per_beam=3 if cfg["with_amp"] else 2,
                   c_enc=cfg.get("c_enc", 64), width=cfg["width"],
                   variant=cfg["variant"], m_max=m_max,
                   dec_norm=cfg.get("dec_norm", "bn"),
                   quantize=cfg.get("quantize", "none")).to(device)
    model.load_state_dict(ck["model"])
    geometry = cfg.get("geometry", "A")

    W2 = taylor_window_2d(N, cfg["sll_db"], cfg["nbar"])
    c_pb = 3 if cfg["with_amp"] else 2
    m_list = [int(v) for v in args.m_list.split(",")]
    stats_by_m = {}
    for m in m_list:
        for mod in model.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.reset_running_stats()
                mod.momentum = None
        Xc, _, _, mc, _ = pregenerate_multi(
            np.random.default_rng(args.seed0 + m), [m], args.n_cal, W2, True,
            N, D_OL, N_PAD, cfg["theta_min"], cfg["theta_max"],
            cfg.get("min_sep", 5.0), m_max=max(m, m_max), geometry=geometry)
        model.train()
        with torch.no_grad():
            for s in range(0, Xc.shape[0], 32):
                model(Xc[s:s + 32, :c_pb * m].to(device))
        stats_by_m[m] = copy.deepcopy(
            {k: v.clone() for k, v in model.state_dict().items()
             if "running" in k or "num_batches" in k})
        print("calibrated M={}".format(m), flush=True)

    path = out / "bn_stats_by_m.pt"
    torch.save(dict(stats_by_m=stats_by_m, ckpt=args.ckpt, m_list=m_list,
                    n_cal=args.n_cal), path)
    print("saved -> {}".format(path), flush=True)


if __name__ == "__main__":
    main()

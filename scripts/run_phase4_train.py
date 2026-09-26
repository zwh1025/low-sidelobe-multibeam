"""Phase 4 training entry: LCS (learned complex superposition), multi-M.

Usage:
  R1a (strict, M=2 only, v9 protocol):
      python scripts/run_phase4_train.py --run r1a_s --variant S --mtrain 2 --ntrain 4000 --nval 500
  R1b (decoder fallback):
      python scripts/run_phase4_train.py --run r1b_d --variant D --mtrain 2 --ntrain 4000 --nval 500
  R2 (mixed M 1..8, config defaults):
      python scripts/run_phase4_train.py --run r2_mix
  Smoke:
      python scripts/run_phase4_train.py --run smoke --smoke
"""

import os

os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

from metasurf.train.trainer import train_run_multi

CFG_PATH = ROOT / "configs" / "phase4.yaml"
OUT_ROOT = ROOT / "results" / "phase4_network"

RUNS = {
    "main":  dict(with_amp=True),
    "a_amp": dict(with_amp=False),
}


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


def parse_mlist(s):
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(v) for v in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="main")
    ap.add_argument("--variant", choices=["S", "D"], default=None)
    ap.add_argument("--mtrain", type=str, default=None,
                    help='e.g. "2" (R1) or "1-8" (R2) or "1,2,4"')
    ap.add_argument("--ntrain", type=int, default=None)
    ap.add_argument("--nval", type=int, default=None)
    ap.add_argument("--gamma2mode", choices=["fixed", "dual"], default=None)
    ap.add_argument("--decnorm", choices=["bn", "gn"], default=None,
                    help="decoder normalization (default: yaml dec_norm)")
    ap.add_argument("--init_from", default=None,
                    help="warm-start model weights from a ckpt (e.g. "
                         "results/phase4_network/r2b/ckpt_last.pt)")
    ap.add_argument("--tau_start", type=float, default=None)
    ap.add_argument("--tau_end", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--geometry", choices=["A", "B"], default=None,
                    help="A = plane-wave priors (default), B = Du point-feed "
                         "reflectarray priors (R4)")
    ap.add_argument("--quantize", choices=["none", "ste1"], default=None,
                    help="STE 1-bit element-phase head (R4)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    with open(CFG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["out_root"] = str(OUT_ROOT)
    cfg["n"] = cfg["array"]["n"]
    cfg["d_ol"] = cfg["array"]["d_over_lambda"]
    cfg["n_pad"] = cfg["array"]["n_pad"]
    cfg["sll_db"] = cfg["taylor"]["sll_db"]
    cfg["nbar"] = cfg["taylor"]["nbar"]
    d = cfg["data"]
    cfg.update(m_list=list(d["m_list"]), m_max=d["m_max"],
               n_train=d["n_train"], n_val=d["n_val"],
               theta_min=d["theta_min_deg"], theta_max=d["theta_max_deg"],
               min_sep=d["min_sep_deg"])
    mo = cfg["model"]
    cfg.update(variant=mo["variant"], c_enc=mo["c_enc"], width=mo["width"],
               dec_norm=mo.get("dec_norm", "bn"))
    tr = cfg["train"]
    cfg.update(batch=tr["batch"], epochs=tr["epochs"], lr=tr["lr"],
               weight_decay=tr["weight_decay"], patience=tr["patience"],
               clip=tr["clip"])
    for k in ("alpha", "beta", "gamma", "gamma2", "gamma2_mode", "dual_eta",
              "delta", "use_pat", "sll_domain", "tau_start", "tau_end",
              "l_beam_mode", "beam_floor", "stage1_frac", "warmup_frac",
              "r_train", "dir2", "dir_floor", "sll_floor_db"):
        cfg[k] = cfg["loss"][k]

    if args.run in RUNS:
        spec = RUNS[args.run]
        cfg["with_amp"] = spec["with_amp"]
    else:
        cfg["with_amp"] = d["with_amp_channel"]
    if args.variant is not None:
        cfg["variant"] = args.variant
    if args.mtrain is not None:
        cfg["m_list"] = parse_mlist(args.mtrain)
    if args.ntrain is not None:
        cfg["n_train"] = args.ntrain
    if args.nval is not None:
        cfg["n_val"] = args.nval
    if args.gamma2mode is not None:
        cfg["gamma2_mode"] = args.gamma2mode
    if args.geometry is not None:
        cfg["geometry"] = args.geometry
    if args.quantize is not None:
        cfg["quantize"] = args.quantize
    if args.decnorm is not None:
        cfg["dec_norm"] = args.decnorm
    if args.init_from is not None:
        cfg["init_from"] = str(ROOT / args.init_from
                                if not args.init_from.startswith("/") else args.init_from)
    for k in ("tau_start", "tau_end", "lr", "epochs", "patience"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.smoke:
        cfg.update(n_train=64, n_val=16, epochs=2, batch=16, patience=2,
                   m_list=[1, 8])

    cfg["seed"] = cfg["seed"]
    cfg["data_seed"] = cfg["seed"] + 50

    device = pick_device()
    print("device={} | run={} variant={} m_list={} n_train={} smoke={}".format(
        device, args.run, cfg["variant"], cfg["m_list"], cfg["n_train"],
        args.smoke), flush=True)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    train_run_multi(args.run, cfg, device=device)


if __name__ == "__main__":
    main()

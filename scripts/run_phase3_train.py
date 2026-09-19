"""Phase 3 training entry: main run + ablations (sequential; background-friendly)."""

import os

os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

from metasurf.train.trainer import train_run

CFG_PATH = ROOT / "configs" / "phase3.yaml"
OUT_ROOT = ROOT / "results" / "phase3_network"

RUNS = {
    "main":  dict(use_pat=True,  delta=None, with_amp=True),
    "b_pat": dict(use_pat=False, delta=0.0,  with_amp=True),
    "a_amp": dict(use_pat=True,  delta=None, with_amp=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, default="main,b_pat,a_amp")
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
    cfg.update(n_train=d["n_train"], n_val=d["n_val"],
               theta_min=d["theta_min_deg"], theta_max=d["theta_max_deg"],
               min_sep=d["min_sep_deg"], swap_augment=d["swap_augment"])
    tr = cfg["train"]
    cfg.update(batch=tr["batch"], epochs=tr["epochs"], lr=tr["lr"],
               weight_decay=tr["weight_decay"], patience=tr["patience"],
               clip=tr["clip"], width=tr["width"])

    if args.smoke:
        cfg.update(n_train=64, n_val=16, epochs=2, batch=16, patience=2,
                   warmup_frac=0.3)
        args.runs = "main,b_pat,a_amp"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device={} | runs={} | smoke={}".format(device, args.runs, args.smoke),
          flush=True)
    torch.backends.cudnn.benchmark = True

    for name in args.runs.split(","):
        run_cfg = dict(cfg)
        run_cfg["seed"] = cfg["seed"]
        run_cfg["data_seed"] = cfg["seed"] + 50
        spec = RUNS[name]
        run_cfg.update(spec)
        if spec["delta"] is None:
            run_cfg["delta"] = cfg["loss"]["delta"]
        else:
            run_cfg["delta"] = spec["delta"]
        run_cfg["use_pat"] = spec["use_pat"]
        run_cfg["with_amp"] = spec["with_amp"]
        for k in ("alpha", "beta", "gamma", "gamma2", "sll_domain",
                  "tau_start", "tau_end", "l_beam_mode", "beam_floor",
                  "stage1_frac", "warmup_frac", "r_train"):
            run_cfg[k] = cfg["loss"][k]
        train_run(name, run_cfg, device=device)


if __name__ == "__main__":
    main()

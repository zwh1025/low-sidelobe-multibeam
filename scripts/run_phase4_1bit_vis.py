"""1-bit visual comparison on the Du 2025 dual-beam case (geometry B).

Case: beams (10 deg, 45 deg) + (15 deg, 180 deg) -- the exact dual-beam
example of Du et al., AWPL 2025 (their Fig. 1). Renders
  beam_1bit_geomB.png : [LCS 1-bit STE / LCS continuous (same weights) /
                         Du eq.(6) 1-bit / arg-sum], patterns in geometry B
                        (feed phase included), SLL annotated.

Usage: python scripts/run_phase4_1bit_vis.py [--run r4a_1bit]
"""

import os

os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

import sys
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
from metasurf.physics.bp_phase import point_feed_geometry
from metasurf.physics.metrics import evaluate_pattern
from metasurf.baselines.superposition import arg_sum_phase, onebit_code, code_phase
from metasurf.model.lcs import LCSNet
from metasurf.eval.visualize import draw_pattern_db

N, N_PAD, D_OL, F_OVER_D = 64, 512, 0.5, 0.8
SLL_DB, NBAR = -25.0, 5
OUT = ROOT / "results" / "phase4_network"
CASE = [(10.0, 45.0), (15.0, 180.0)]


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


def main():
    device = pick_device()
    u_axis = fft_uv_axes(N_PAD, D_OL)
    W2 = taylor_window_2d(N, SLL_DB, NBAR)
    ck = torch.load(OUT / "r4a_1bit" / "ckpt_best.pt", map_location=device,
                    weights_only=False)
    cfg = ck["cfg"]
    model_q = LCSNet(c_per_beam=3, c_enc=cfg.get("c_enc", 64),
                     width=cfg["width"], variant=cfg["variant"],
                     m_max=cfg.get("m_max", 8),
                     dec_norm=cfg.get("dec_norm", "bn"),
                     quantize="ste1").to(device)
    model_q.load_state_dict(ck["model"])
    model_c = LCSNet(c_per_beam=3, c_enc=cfg.get("c_enc", 64),
                     width=cfg["width"], variant=cfg["variant"],
                     m_max=cfg.get("m_max", 8),
                     dec_norm=cfg.get("dec_norm", "bn"),
                     quantize="none").to(device)
    model_c.load_state_dict(ck["model"])
    st = torch.load(OUT / "r4a_1bit" / "bn_stats_by_m.pt", map_location=device,
                    weights_only=False)
    for mdl in (model_q, model_c):
        sd = mdl.state_dict()
        for k, v in st["stats_by_m"][2].items():
            sd[k] = v
        mdl.load_state_dict(sd)
    model_q.eval()
    model_c.eval()

    beams = CASE
    feed = point_feed_geometry(beams[0][0], beams[0][1], N, D_OL, F_OVER_D)[1]
    feed_c = np.exp(1j * feed)
    phB = np.stack([point_feed_geometry(t, p, N, D_OL, F_OVER_D)[0]
                    for t, p in beams])
    X = np.zeros((1, 6, N, N), dtype=np.float32)
    for i in range(2):
        X[0, 3 * i] = np.cos(phB[i])
        X[0, 3 * i + 1] = np.sin(phB[i])
        X[0, 3 * i + 2] = W2
    x = torch.from_numpy(X).to(device)
    targets = [angles_to_uv(t, p) for t, p in beams]

    with torch.no_grad():
        ap_q = model_q(x)[0].cpu().numpy()
        ap_c = model_c(x)[0].cpu().numpy()
    ap_du = np.exp(1j * (code_phase(onebit_code(phB)) + feed))
    ap_arg = np.exp(1j * arg_sum_phase(phB))

    panels = [("LCS 1-bit STE (ours)", ap_q * feed_c),
              ("LCS continuous (same weights)", ap_c * feed_c),
              ("Du eq.(6) 1-bit", ap_du),
              ("arg-sum continuous", ap_arg)]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6))
    zoom_c = (0.122 - 0.259) / 2.0        # centroid of the two beams (u, v)
    zoom_v = (0.122 + 0.0) / 2.0
    for c, (lab, ap) in enumerate(panels):
        F = aperture_to_pattern_np(ap, N_PAD)
        r = evaluate_pattern(F, u_axis, u_axis, targets,
                             [t for t, p in beams])
        ax = axes[c]
        draw_pattern_db(ax, F, u_axis, u_axis, targets, vmin=-40.0,
                        colorbar=(c == 3))
        ax.set_title("{}\nSLL {:.2f} dB".format(lab, r["sll_db"]), fontsize=9)
        ax.set_xlabel("")
        if c > 0:
            ax.set_ylabel("")
        axin = ax.inset_axes([0.56, 0.54, 0.42, 0.44])
        db = 20.0 * np.log10(np.abs(F) / np.abs(F).max() + 1e-12)
        U, V = np.meshgrid(u_axis, u_axis, indexing="ij")
        dbm = np.where(U ** 2 + V ** 2 <= 1.0, db, np.nan)
        axin.imshow(dbm.T, extent=[u_axis[0], u_axis[-1], u_axis[0],
                                   u_axis[-1]], origin="lower", vmin=-40.0,
                    vmax=0.0, cmap="viridis", interpolation="nearest")
        for (u0, v0) in targets:
            axin.plot(u0, v0, "w+", markersize=7, markeredgewidth=1.2)
        axin.set_xlim(zoom_c - 0.32, zoom_c + 0.32)
        axin.set_ylim(zoom_v - 0.32, zoom_v + 0.32)
        axin.set_xticks([])
        axin.set_yticks([])
        axin.set_facecolor("k")
        for s in axin.spines.values():
            s.set_color("w")
        ax.indicate_inset_zoom(axin, edgecolor="w")
    fig.suptitle("Du 2025 dual-beam case (10°,45°)+(15°,180°), geometry B "
                 "(point feed F/D=0.8), 1-bit comparison", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "beam_1bit_geomB.png", dpi=140)
    plt.close(fig)
    print("saved -> {}".format(OUT / "beam_1bit_geomB.png"), flush=True)
    for lab, ap in panels:
        F = aperture_to_pattern_np(ap, N_PAD)
        r = evaluate_pattern(F, u_axis, u_axis, targets,
                             [t for t, p in beams])
        print("{:32s} SLL {:.2f} dB  cons {:.2f}  perr {:.3f} deg".format(
            lab, r["sll_db"], r["gain_consistency_db"],
            r["point_err_max_deg"]), flush=True)


if __name__ == "__main__":
    main()

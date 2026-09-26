"""Phase 4 stage-0' self-test: unit acceptance before R1 training.

Checks:
  T1  LCS-S / LCS-D degeneracy: fresh init (g=0, Delta=0) == arg-sum aperture
  T2  RNG stream identity: pregenerate_multi([2]) == phase-3 pregenerate (v9 parity)
  T3  NPU fwd+bwd for M in {1, 3, 8} (bucketed slices, loss + backward)
  T4  bucketed_batches: full coverage, same-M batches, deterministic under seed
"""

import os

os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from metasurf.physics.taylor import taylor_window_2d
from metasurf.model.lcs import LCSNet
from metasurf.train.trainer import (pregenerate, pregenerate_multi,
                                    bucketed_batches)
from metasurf.train.losses import PhysicsLoss

N, N_PAD, D_OL = 64, 512, 0.5
W2 = taylor_window_2d(N, -25.0, 5)
DEV = "npu" if torch.npu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
try:
    import torch_npu  # noqa: F401
    _HAS_NPU = torch.npu.is_available()
except ImportError:
    _HAS_NPU = False

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print("[{}] {} {}".format("PASS" if ok else "FAIL", name, detail),
          flush=True)


def t1_degeneracy():
    rng = np.random.default_rng(7)
    for variant in ("S", "D"):
        X = np.zeros((4, 3 * 2, N, N), dtype=np.float32)
        for i in range(2):
            ph = rng.uniform(0, 2 * np.pi, (N, N))
            X[:, 3 * i] = np.cos(ph)
            X[:, 3 * i + 1] = np.sin(ph)
            X[:, 3 * i + 2] = W2
        x = torch.from_numpy(X).to(DEV)
        model = LCSNet(c_per_beam=3, variant=variant).to(DEV).eval()
        with torch.no_grad():
            ap = model(x)
            s = torch.complex(x[:, 0], x[:, 1]) + torch.complex(x[:, 3], x[:, 4])
            ref = s / (s.abs() + 1e-12)
        err = float((ap - ref).abs().max())
        check("T1 degeneracy LCS-{}".format(variant), err < 1e-6,
              "max|ap-argsum| = {:.2e}".format(err))


def t2_rng_parity():
    n = 200
    a = pregenerate(np.random.default_rng(819), n, W2, True, N, D_OL, N_PAD,
                    2.0, 45.0, 5.0)
    b = pregenerate_multi(np.random.default_rng(819), [2], n, W2, True, N,
                          D_OL, N_PAD, 2.0, 45.0, 5.0, m_max=8)
    same_beams = all(x == y for x, y in zip(a[3], b[4]))
    same_x = torch.equal(a[0][:, :6], b[0][:, :6])
    same_uv = torch.equal(a[1][:, :2], b[1][:, :2])
    check("T2 RNG stream parity (beams)", same_beams)
    check("T2 RNG stream parity (X/uv)", same_x and same_uv)


def t3_fwd_bwd():
    if not _HAS_NPU and DEV == "cpu":
        pass
    loss_fn = PhysicsLoss(n=N, n_pad=N_PAD, d_ol=D_OL, sll_domain="dB",
                          tau_start=1.0, tau_end=0.1, l_beam_mode="hinge",
                          beam_floor=0.95, gamma2=50.0, window2d=W2).to(DEV)
    rng = np.random.default_rng(11)
    for m in (1, 3, 8):
        X = np.zeros((4, 3 * m, N, N), dtype=np.float32)
        for i in range(m):
            ph = rng.uniform(0, 2 * np.pi, (N, N))
            X[:, 3 * i] = np.cos(ph)
            X[:, 3 * i + 1] = np.sin(ph)
            X[:, 3 * i + 2] = W2
        u_ax = np.linspace(-0.8, 0.8, m)
        tgt_uv = np.stack([u_ax, -u_ax], axis=-1)[None].repeat(4, 0)
        tgt_idx = np.zeros((4, m, 2), dtype=np.int64)
        x = torch.from_numpy(X).to(DEV).requires_grad_(True)
        uv = torch.from_numpy(tgt_uv).to(DEV)
        idx = torch.from_numpy(tgt_idx).to(DEV)
        model = LCSNet(c_per_beam=3, variant="D").to(DEV)
        ap = model(x)
        loss, parts = loss_fn(ap, x, uv, idx, 0.5)
        loss.backward()
        gmax = float(x.grad.abs().max())
        check("T3 fwd+bwd M={}".format(m),
              np.isfinite(float(loss)) and np.isfinite(gmax) and gmax > 0,
              "loss={:.4f} gmax={:.2e}".format(float(loss), gmax))


def t4_bucketing():
    m_arr = np.array([1, 1, 2, 2, 2, 8, 8, 8, 8])
    g = torch.Generator().manual_seed(0)
    seen = []
    for m, idx in bucketed_batches(m_arr, 2, generator=g):
        seen.extend(idx.tolist())
        assert (m_arr[idx] == m).all()
    check("T4 bucketing coverage", sorted(seen) == list(range(9)),
          "batches cover all tasks once, same-M within batch")
    gA = torch.Generator().manual_seed(0)
    gB = torch.Generator().manual_seed(0)
    same = all(torch.equal(i1, i2) and m1 == m2
               for (m1, i1), (m2, i2) in zip(bucketed_batches(m_arr, 2, gA),
                                             bucketed_batches(m_arr, 2, gB)))
    check("T4 bucketing determinism", same)


if __name__ == "__main__":
    print("device={}".format(DEV), flush=True)
    t1_degeneracy()
    t2_rng_parity()
    t3_fwd_bwd()
    t4_bucketing()
    n_pass = sum(1 for _, ok, _ in results if ok)
    print("--- selftest: {}/{} PASS ---".format(n_pass, len(results)), flush=True)
    sys.exit(0 if n_pass == len(results) else 1)

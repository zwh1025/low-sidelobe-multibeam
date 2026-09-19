# Phase 0 smoke report

date: 2026-09-18 10:20:16

python: 3.10.20 | torch: 2.5.1+cu121 | cuda: True

## Checks

| check | result | detail |
|---|---|---|
| C1 grid angle<->uv roundtrip (200 samples) | PASS | max_err=8.53e-14 deg |
| C2-1 geometry A beam 1 peak position | PASS | target=(0.2253,-0.1268) peak=(0.2266,-0.1250) err=0.00215 (tol=0.00586) |
| C2-2 geometry A beam 2 peak position | PASS | target=(0.4471,0.3727) peak=(0.4453,0.3711) err=0.00240 (tol=0.00586) |
| C2-3 geometry A beam 3 peak position | PASS | target=(0.5119,-0.4855) peak=(0.5117,-0.4844) err=0.00118 (tol=0.00586) |
| C3 geometry B point-feed beam peak | PASS | target=(0.2253,-0.1268) peak=(0.2266,-0.1250) err=0.00215 |
| C4 numpy/torch |F| parity | PASS | rel_l2=1.171e-07 |
| C6 GPU forward/backward | PASS | device=NVIDIA GeForce RTX 3050, loss=4095.999, grad_max=2.699e-07 |

## Informational

| item | value |
|---|---|
| FFT grid | n_pad=512, du=0.003906, range=[-1.000,0.996] |
| beam set | (14.98 deg, 330.64 deg); (35.60 deg, 39.81 deg); (44.87 deg, 316.51 deg) |
| Taylor 1D edge taper | -7.28 dB (sll=-25.0 dB, nbar=5) |
| ideal amp+phase beam (15.0,330.6) SLL | -25.02 dB |
| ideal amp+phase beam (35.6,39.8) SLL | -25.02 dB |
| ideal amp+phase beam (44.9,316.5) SLL | -25.03 dB |
| phase-only (amp=1) beam (15.0,330.6) SLL | -13.21 dB |
| numpy AF batch-32 time | 359.52 ms/call |
| torch GPU AF batch-32 time | 5.38 ms/call |
| pyyaml available | yes |

Beam set: (14.98 deg, 330.64 deg); (35.60 deg, 39.81 deg); (44.87 deg, 316.51 deg)

Note: fixed mainlobe mask radii were used per aperture type (Taylor-designed 0.05, uniform/phase-only 0.036). A single fixed radius cannot serve both mainlobe widths (Taylor first null ~0.042 vs uniform ~0.031 in u); phase-1 metrics.py will adopt a first-null-based adaptive mask.

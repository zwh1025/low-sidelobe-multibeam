"""Differentiable physics layer and multi-term pattern-domain loss."""

import math

import numpy as np
import torch
import torch.nn as nn

from ..physics.grid import fft_uv_axes
from ..physics.array_factor import aperture_to_pattern_torch


class PhysicsLoss(nn.Module):
    """Pattern-domain loss over the differentiable array factor.

    Output (cos, sin) is unit-normalized to enforce the phase-only constraint.
    SLL term: mean sidelobe power during warmup, then log-sum-exp soft-max of
    the sidelobe dB map with cosine temperature annealing. r_train is the fixed
    mainlobe mask radius used for training (adaptive radii are an evaluation
    concern only, see metrics.py).
    """

    def __init__(self, n=64, n_pad=512, d_ol=0.5, r_train=0.040,
                 alpha=1.0, beta=0.5, gamma=0.1, delta=0.05, use_pat=True,
                 gamma2=1.0, sll_domain="power", tau_start=0.01, tau_end=0.002,
                 l_beam_mode="square", beam_floor=0.7, stage1_frac=0.0,
                 warmup_frac=0.0, window2d=None):
        super().__init__()
        self.n, self.n_pad = n, n_pad
        self.r2 = r_train ** 2
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.use_pat = use_pat
        self.gamma2 = gamma2
        self.sll_domain = sll_domain
        self.tau_start, self.tau_end, self.warmup_frac = tau_start, tau_end, warmup_frac
        self.l_beam_mode = l_beam_mode
        self.beam_floor = beam_floor
        self.stage1_frac = stage1_frac
        u_axis = fft_uv_axes(n_pad, d_ol)
        U, V = np.meshgrid(u_axis, u_axis, indexing="ij")
        self.register_buffer("U", torch.from_numpy(U.astype(np.float32)))
        self.register_buffer("V", torch.from_numpy(V.astype(np.float32)))
        self.register_buffer("visible", torch.from_numpy((U ** 2 + V ** 2) <= 1.0))
        if window2d is not None:
            self.register_buffer("window2d",
                                 torch.from_numpy(window2d.astype(np.float32)))
        else:
            self.window2d = None

    def forward(self, ap, X, tgt_uv, tgt_idx, epoch_frac):
        B = ap.shape[0]
        F_abs = torch.abs(self._pattern(ap))
        F_norm = F_abs / (F_abs.amax(dim=(-2, -1), keepdim=True) + 1e-12)

        main = torch.zeros((B, self.n_pad, self.n_pad), dtype=torch.bool,
                           device=F_abs.device)
        for i in range(tgt_uv.shape[1]):
            u0 = tgt_uv[:, i, 0].view(-1, 1, 1)
            v0 = tgt_uv[:, i, 1].view(-1, 1, 1)
            main |= (self.U[None] - u0) ** 2 + (self.V[None] - v0) ** 2 <= self.r2
        sl = self.visible[None] & ~main

        Pb = torch.zeros(B, device=F_abs.device)
        for i in range(tgt_uv.shape[1]):
            u0 = tgt_uv[:, i, 0].view(-1, 1, 1)
            v0 = tgt_uv[:, i, 1].view(-1, 1, 1)
            disk = ((self.U[None] - u0) ** 2 + (self.V[None] - v0) ** 2) <= self.r2
            Pb = torch.maximum(Pb, (F_abs * disk).amax(dim=(-2, -1)) ** 2)
        p = F_abs ** 2 / (Pb.view(-1, 1, 1) + 1e-12)

        if epoch_frac < self.warmup_frac:
            l_sll = (p * sl).sum(dim=(-2, -1)) / sl.sum(dim=(-2, -1)).clamp(min=1)
            l_sll = l_sll.mean()
            tau = float("nan")
        else:
            prog = (epoch_frac - self.warmup_frac) / max(1e-9, 1.0 - self.warmup_frac)
            tau = self.tau_end + 0.5 * (self.tau_start - self.tau_end) \
                * (1.0 + math.cos(math.pi * min(1.0, prog)))
            if self.sll_domain == "dB":
                x = 20.0 * torch.log10(F_norm + 1e-12)
            else:
                x = p
            x = torch.where(sl, x, torch.full_like(x, -1e9))
            lse = torch.logsumexp(x / tau, dim=(-2, -1)) \
                - torch.log(sl.sum(dim=(-2, -1)).clamp(min=1).float())
            l_sll = (tau * lse).mean()

        bidx = torch.arange(B, device=F_abs.device)
        dir_terms = []
        G = []
        for i in range(tgt_uv.shape[1]):
            iu = tgt_idx[:, i, 0]
            iv = tgt_idx[:, i, 1]
            dir_terms.append(1.0 - F_norm[bidx, iu, iv])
            u0 = tgt_uv[:, i, 0].view(-1, 1, 1)
            v0 = tgt_uv[:, i, 1].view(-1, 1, 1)
            disk = ((self.U[None] - u0) ** 2 + (self.V[None] - v0) ** 2) <= self.r2
            G.append((F_abs * disk).amax(dim=(-2, -1)))
        l_dir = torch.stack(dir_terms, dim=1).sum(dim=1).mean()
        G = torch.stack(G, dim=1)
        l_gain = (G.std(dim=1) / G.mean(dim=1).clamp(min=1e-12)).mean()
        G_hat = G / G.amax(dim=1, keepdim=True).clamp(min=1e-12)
        if self.l_beam_mode == "hinge":
            l_beam = (torch.clamp(self.beam_floor - G_hat.amin(dim=1),
                                  min=0.0) ** 2).mean()
        else:
            l_beam = ((1.0 - G_hat.amin(dim=1)) ** 2).mean()

        total = self.beta * l_dir + self.gamma * l_gain + self.gamma2 * l_beam
        if epoch_frac >= self.stage1_frac:
            total = total + self.alpha * l_sll
        parts = dict(l_sll=float(l_sll), l_dir=float(l_dir), l_gain=float(l_gain),
                     tau=float(tau), l_pat=0.0, l_beam=float(l_beam))
        if self.use_pat and self.delta > 0:
            with_amp = X.shape[1] == 6
            step = 3 if with_amp else 2
            acc = torch.zeros(X.shape[0], self.n, self.n, dtype=torch.complex64,
                              device=X.device)
            for i in range(2):
                base = i * step
                A = X[:, base + 2] if with_amp else self.window2d[None]
                acc = acc + A * torch.complex(X[:, base], X[:, base + 1])
            Fi_abs = torch.abs(self._pattern(acc))
            Fi_norm = Fi_abs / (Fi_abs.amax(dim=(-2, -1), keepdim=True) + 1e-12)
            l_pat = ((F_norm - Fi_norm) ** 2).mean() / ((Fi_norm ** 2).mean() + 1e-12)
            total = total + self.delta * l_pat
            parts["l_pat"] = float(l_pat)
        return total, parts

    def _pattern(self, ap):
        return aperture_to_pattern_torch(ap, self.n_pad)

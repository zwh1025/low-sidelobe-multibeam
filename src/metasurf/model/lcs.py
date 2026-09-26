"""Learned Complex Superposition (LCS): set-symmetric generalization of arg-sum.

    phi = arg( sum_i w_i(m,n) e^{j phi_i} ),   w_i = 1 + g_i

g_i is predicted by one shared per-beam encoder; w == 1 (zero-init g) makes the
model EXACTLY the arg-sum baseline at initialization (Du's continuous method).
The sum structure is permutation-equivariant and M-agnostic by construction:
tasks in a batch must share the same M (bucketed batching, see trainer).

Variants:
  "S": aperture = normalize( sum_i w_i e^{j phi_i} )                (strict)
  "D": + zero-init U-Net decoder Delta on aggregated features       (fallback)

LCS-D relation to ArgSumResidualNet (v9): normalize(sum w_i e^{j phi_i} + Delta)
= normalize(s + Delta') with s = sum e^{j phi_i} — same solution family, but
Delta' decomposes additively per beam plus a decoder over aggregate features.
"""

import torch
import torch.nn as nn

from .unet import UNet


class BeamEncoder(nn.Module):
    """Shared conv encoder for one beam's (cos phi, sin phi[, A]) channels."""

    def __init__(self, c_in=3, c_hidden=64):
        super().__init__()
        ch = c_hidden
        self.body = nn.Sequential(
            nn.Conv2d(c_in, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.g_head = nn.Conv2d(ch, 2, 1)
        nn.init.zeros_(self.g_head.weight)
        nn.init.zeros_(self.g_head.bias)

    def features(self, x):
        return self.body(x)

    def g(self, f):
        return self.g_head(f)


class LCSNet(nn.Module):
    """Set-symmetric learned complex superposition network.

    Input: [B, c_per_beam * M, H, W] with contiguous per-beam blocks
    (cos, sin[, A]); M is inferred from the channel count, so every task in
    the batch must share the same M. Output: complex unit-modulus aperture
    [B, H, W] (same contract as ArgSumResidualNet).
    """

    def __init__(self, c_per_beam=3, c_enc=64, width=1.0, variant="D",
                 m_max=8, with_amp=None, dec_norm="bn", quantize="none"):
        super().__init__()
        if with_amp is not None:
            c_per_beam = 3 if with_amp else 2
        self.c_per_beam = c_per_beam
        self.variant = variant
        self.m_max = m_max
        self.quantize = quantize
        self.encoder = BeamEncoder(c_in=c_per_beam, c_hidden=c_enc)
        if variant == "D":
            n_stats = 5 if c_per_beam == 3 else 4
            self.decoder = UNet(in_ch=c_enc + n_stats, out_ch=2, width=width,
                                norm=dec_norm)
            nn.init.zeros_(self.decoder.outc.weight)
            nn.init.zeros_(self.decoder.outc.bias)
        elif variant != "S":
            raise ValueError("variant must be 'S' or 'D', got {!r}".format(variant))

    def forward(self, x):
        B, C, H, W = x.shape
        M = C // self.c_per_beam
        c = self.c_per_beam

        feats = self.encoder.features(x.reshape(B * M, c, H, W))
        f = feats.view(B, M, -1, H, W)
        g = self.encoder.g(feats).view(B, M, 2, H, W)

        s = torch.zeros(B, H, W, dtype=torch.complex64, device=x.device)
        z = torch.zeros(B, H, W, dtype=torch.complex64, device=x.device)
        Fsum = None
        for i in range(M):
            base = i * c
            ph = torch.complex(x[:, base], x[:, base + 1])
            w = 1.0 + torch.complex(g[:, i, 0], g[:, i, 1])
            s = s + ph
            z = z + w * ph
            if self.variant == "D":
                if Fsum is None:
                    Fsum = f[:, i].clone()
                else:
                    Fsum = Fsum + f[:, i]
        if self.variant == "D":
            sn = s / (s.abs() + 1e-12)
            stats = [sn.real.unsqueeze(1), sn.imag.unsqueeze(1),
                     (s.abs() / M).unsqueeze(1),
                     torch.full((B, 1, H, W), M / float(self.m_max),
                                device=x.device)]
            if c == 3:
                stats.append(x[:, 2::c].mean(dim=1).unsqueeze(1))
            d = self.decoder(torch.cat([Fsum] + stats, dim=1))
            z = z + torch.complex(d[:, 0], d[:, 1])
        ap = z / (z.abs() + 1e-12)
        if self.quantize == "ste1":
            # 1-bit element phase {0, pi}: unit complex -> sign(Re); straight-
            # through estimator passes gradients through the discrete step.
            aq = torch.where(ap.real >= 0,
                             torch.ones_like(ap.real),
                             -torch.ones_like(ap.real)).to(torch.complex64)
            ap = ap + (aq - ap).detach()
        elif self.quantize != "none":
            raise ValueError("quantize must be 'none' or 'ste1', got {!r}".format(
                self.quantize))
        return ap

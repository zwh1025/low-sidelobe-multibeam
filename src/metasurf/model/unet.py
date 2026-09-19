"""Lightweight U-Net: dual-beam aperture priors -> (cos phi, sin phi) phase maps."""

import torch
import torch.nn as nn


class DoubleConv(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.1, inplace=True),
        )


class Down(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(nn.MaxPool2d(2), DoubleConv(c_in, c_out))


class Up(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(c_in, c_out)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """in_ch -> out_ch at 64x64; width scales the channel budget (~7.7M params at width 1)."""

    def __init__(self, in_ch=6, out_ch=2, width=1.0):
        super().__init__()
        f = [max(8, int(round(c * width))) for c in (64, 128, 256, 512)]
        self.inc = DoubleConv(in_ch, f[0])
        self.down1 = Down(f[0], f[1])
        self.down2 = Down(f[1], f[2])
        self.down3 = Down(f[2], f[3])
        self.up1 = Up(f[3] + f[2], f[2])
        self.up2 = Up(f[2] + f[1], f[1])
        self.up3 = Up(f[1] + f[0], f[0])
        self.drop = nn.Dropout2d(0.1)
        self.outc = nn.Conv2d(f[0], out_ch, 1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.drop(self.down3(x2))
        y = self.up1(x3, x2)
        y = self.up2(y, x1)
        y = self.up3(y, x0)
        return self.outc(y)


class ArgSumResidualNet(nn.Module):
    """Physics-informed residual: unit-modulus aperture = normalize(arg-sum + Delta(x)).

    The analytic arg-sum superposition of the input BP phases (balanced beams,
    SLL ~ -9.7 dB) is the base solution; the wrapped U-Net outputs a complex
    residual so training starts inside the two-beam basin and spends its
    capacity on sidelobe reduction. Input layout: per beam (cos, sin[, amp]).
    """

    def __init__(self, in_ch=6, width=1.0, res_scale=0.3):
        super().__init__()
        self.c_per_beam = in_ch // 2
        self.unet = UNet(in_ch=in_ch, out_ch=2, width=width)
        self.res_scale = nn.Parameter(torch.tensor(float(res_scale)))

    def forward(self, x):
        s = torch.zeros(x.shape[0], x.shape[2], x.shape[3],
                        dtype=torch.complex64, device=x.device)
        for i in range(2):
            base = i * self.c_per_beam
            s = s + torch.complex(x[:, base], x[:, base + 1])
        ref = s / (torch.abs(s) + 1e-12)
        d = self.unet(x)
        delta = (d[:, 0] + 1j * d[:, 1]) * self.res_scale
        ap = ref + delta
        ap = ap / (torch.abs(ap) + 1e-12)
        return ap

import numpy as np
import torch


def _pad_amount(n, n_pad):
    if n_pad < n or (n_pad - n) % 2 != 0:
        raise ValueError("n_pad must be >= n with even difference, got n={}, n_pad={}".format(n, n_pad))
    return (n_pad - n) // 2


def _pad_torch(a, p):
    if a.is_complex():
        re = torch.nn.functional.pad(a.real, (p, p, p, p))
        im = torch.nn.functional.pad(a.imag, (p, p, p, p))
        return torch.complex(re, im)
    return torch.nn.functional.pad(a, (p, p, p, p))


def aperture_to_pattern_np(a, n_pad=512):
    """Complex aperture (n, n) -> complex pattern F on the centered u-v FFT grid.

    F(P, Q) = sum_{m,n} a[m,n] exp(+j 2 pi ((P - n_pad/2) m + (Q - n_pad/2) n) / n_pad),
    i.e. u_P = (P - n_pad/2) lambda / (n_pad d); implemented via conj(fft2(conj(.))).
    Magnitudes are independent of the corner-vs-centered element origin (unit-modulus
    prefactor), so BP phases defined with centered coordinates are consistent.
    """
    n = a.shape[-1]
    p = _pad_amount(n, n_pad)
    pad = [(0, 0)] * (a.ndim - 2) + [(p, p), (p, p)]
    a_pad = np.pad(a, pad)
    return np.conj(np.fft.fftshift(np.fft.fft2(np.conj(a_pad), axes=(-2, -1)), axes=(-2, -1)))


def _fftshift_torch(x):
    # cat/slice implementation: identical to torch.fft.fftshift for even sizes,
    # and avoids NPU (torch_npu) autograd bugs in fftshift/roll backward.
    h = x.shape[-2] // 2
    w = x.shape[-1] // 2
    x = torch.cat([x[..., h:, :], x[..., :h, :]], dim=-2)
    return torch.cat([x[..., :, w:], x[..., :, :w]], dim=-1)


def aperture_to_pattern_torch(a, n_pad=512):
    """Torch version of aperture_to_pattern_np; supports batched (B, n, n) complex input."""
    n = a.shape[-1]
    p = _pad_amount(n, n_pad)
    a_pad = _pad_torch(a, p)
    a_conj = torch.complex(a_pad.real, -a_pad.imag)
    F = _fftshift_torch(torch.fft.fft2(a_conj))
    return torch.complex(F.real, -F.imag)


def pattern_from_phase_np(phase, amp=None, n_pad=512):
    """Phase matrix (and optional amplitude) -> complex pattern."""
    amp = np.ones_like(phase) if amp is None else amp
    return aperture_to_pattern_np(amp * np.exp(1j * phase), n_pad)

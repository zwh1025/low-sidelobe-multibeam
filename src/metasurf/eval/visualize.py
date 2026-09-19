"""Pattern / phase / code visualization helpers (matplotlib Agg)."""

import numpy as np
import matplotlib.pyplot as plt


def pattern_db(F):
    a = np.abs(F)
    return 20.0 * np.log10(a / a.max() + 1e-12)


def draw_pattern_db(ax, F, u_axis, v_axis, targets=None, vmin=-40.0, title=None, colorbar=True):
    db = pattern_db(F)
    U, V = np.meshgrid(u_axis, v_axis, indexing="ij")
    db = np.where(U ** 2 + V ** 2 <= 1.0, db, np.nan)
    im = ax.imshow(db.T, extent=[u_axis[0], u_axis[-1], v_axis[0], v_axis[-1]],
                   origin="lower", vmin=vmin, vmax=0.0, cmap="viridis", interpolation="nearest")
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="w", linewidth=0.8, linestyle="--"))
    if targets:
        for (u0, v0) in targets:
            ax.plot(u0, v0, "w+", markersize=9, markeredgewidth=1.4)
    if title:
        ax.set_title(title, fontsize=9)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    if colorbar:
        plt.colorbar(im, ax=ax, label="dB")
    return im


def draw_phase(ax, phase, title=None, colorbar=True):
    im = ax.imshow(np.mod(np.asarray(phase), 2.0 * np.pi).T, origin="lower", cmap="twilight",
                   vmin=0.0, vmax=2.0 * np.pi, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=9)
    ax.set_xlabel("element index m")
    ax.set_ylabel("element index n")
    if colorbar:
        plt.colorbar(im, ax=ax)
    return im


def draw_code(ax, code, title=None, colorbar=True):
    im = ax.imshow(np.asarray(code).T, origin="lower", cmap="gray", vmin=0, vmax=1,
                   interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=9)
    ax.set_xlabel("element index m")
    ax.set_ylabel("element index n")
    if colorbar:
        plt.colorbar(im, ax=ax, ticks=[0, 1])
    return im

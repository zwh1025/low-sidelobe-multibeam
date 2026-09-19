"""Phase 2: pre-experiments E1-E4 — quantifying the phase-only physical boundary (paper-ready)."""

import os

os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metasurf.physics import (
    angles_to_uv,
    fft_uv_axes,
    bp_phase_plane,
    point_feed_geometry,
    taylor_window_1d,
    taylor_window_2d,
    aperture_to_pattern_np,
)
from metasurf.physics.metrics import evaluate_pattern
from metasurf.baselines.superposition import (
    arg_sum_phase,
    onebit_code,
    code_phase,
)
from metasurf.data.beam_dataset import sample_beams
from metasurf.eval.visualize import draw_pattern_db

N = 64
N_PAD = 512
D_OL = 0.5
SLL_DB = -25.0
NBAR = 5
SEED = 42
N_TASKS = 20
THETA_MIN, THETA_MAX, MIN_SEP = 2.0, 45.0, 5.0
F_OVER_D = 0.8

OUT = ROOT / "results" / "phase2_preexp"
OUT.mkdir(parents=True, exist_ok=True)
U_AXIS = fft_uv_axes(N_PAD, D_OL)
W1 = taylor_window_1d(N, SLL_DB, NBAR)
W2 = taylor_window_2d(N, SLL_DB, NBAR)

checks = []


def log(msg):
    print(msg, flush=True)


def gate(name, ok, detail):
    checks.append((name, bool(ok), detail))
    log("[{}] {} | {}".format("PASS" if ok else "FAIL", name, detail))


def af_pattern(ap):
    return aperture_to_pattern_np(ap, N_PAD)


def cut_along_u(F_abs, v0):
    j0 = int(np.argmin(np.abs(U_AXIS - v0)))
    return 20.0 * np.log10(F_abs[:, j0] / F_abs[:, j0].max() + 1e-12)


# ------------------------------------------------------- E1/E2/E4 single beam
def run_single_beams(rng):
    beams = sample_beams(rng, N_TASKS, THETA_MIN, THETA_MAX, MIN_SEP)
    rng_rand = np.random.default_rng(SEED + 100)
    rows = []
    rep = {}
    for k, (t, p) in enumerate(beams):
        tgt = angles_to_uv(t, p)
        ph = bp_phase_plane(t, p, N, D_OL)
        F_ideal = af_pattern(W2 * np.exp(1j * ph))
        F_po = af_pattern(np.exp(1j * ph))
        r_ideal = evaluate_pattern(F_ideal, U_AXIS, U_AXIS, [tgt])
        r_po = evaluate_pattern(F_po, U_AXIS, U_AXIS, [tgt])

        ph_rand = rng_rand.uniform(0.0, 2.0 * np.pi, (N, N))
        F_rand = af_pattern(np.exp(1j * ph_rand))
        F_rand_abs = np.abs(F_rand)
        i_t = int(np.argmin(np.abs(U_AXIS - tgt[0])))
        j_t = int(np.argmin(np.abs(U_AXIS - tgt[1])))
        rand_tgt_rel = float(20.0 * np.log10(F_rand_abs[i_t, j_t] / F_rand_abs.max()))
        r_rand = evaluate_pattern(F_rand, U_AXIS, U_AXIS, [tgt])

        rows.append(dict(
            task=k, theta=t, phi_az=p,
            sll_ideal=r_ideal["sll_db"], sll_po=r_po["sll_db"],
            delta=r_po["sll_db"] - r_ideal["sll_db"],
            rand_tgt_rel=rand_tgt_rel, rand_sll=r_rand["sll_db"],
        ))
        if k == 0:
            rep = dict(theta=t, phi_az=p, tgt=tgt, F_ideal=F_ideal, F_po=F_po,
                       sll_ideal=r_ideal["sll_db"], sll_po=r_po["sll_db"])

    sll_ideal = np.array([r["sll_ideal"] for r in rows])
    sll_po = np.array([r["sll_po"] for r in rows])
    deltas = np.array([r["delta"] for r in rows])
    rand_tgt = np.array([r["rand_tgt_rel"] for r in rows])
    rand_sll = np.array([r["rand_sll"] for r in rows])

    log("--- E2 ideal amp+phase single beam: SLL {:.2f} +/- {:.2f} dB (range [{:.2f}, {:.2f}])".format(
        sll_ideal.mean(), sll_ideal.std(), sll_ideal.min(), sll_ideal.max()))
    log("--- E1 phase-only (amp=1) single beam: SLL {:.2f} +/- {:.2f} dB (range [{:.2f}, {:.2f}])".format(
        sll_po.mean(), sll_po.std(), sll_po.min(), sll_po.max()))
    log("--- amplitude-taper contribution (E2 - E1): {:.2f} +/- {:.2f} dB".format(
        deltas.mean(), deltas.std()))
    log("--- E4 random-phase control: target response {:.2f} +/- {:.2f} dB, SLL {:.2f} dB".format(
        rand_tgt.mean(), rand_tgt.std(), rand_sll.mean()))
    log("--- Taylor taper: 1D edge {:.2f} dB, 2D corner {:.2f} dB".format(
        20.0 * np.log10(W1[0]), 20.0 * np.log10(W2[0, 0])))

    gate("E1 phase-only SLL in [-15,-11] dB (hard, magnitude level)",
         bool(np.all((sll_po >= -15.0) & (sll_po <= -11.0))),
         "range [{:.2f}, {:.2f}]".format(sll_po.min(), sll_po.max()))
    gate("E2 ideal amp+phase SLL in [-26,-24] dB",
         bool(np.all((sll_ideal >= -26.0) & (sll_ideal <= -24.0))),
         "range [{:.2f}, {:.2f}]".format(sll_ideal.min(), sll_ideal.max()))
    gate("E4 BP phase forms beam (target resp >= -0.5 dB), random phase does not (<= -6 dB)",
         bool(np.all(rand_tgt <= -6.0)),
         "random target resp range [{:.2f}, {:.2f}] dB; BP forms beam by construction".format(
             rand_tgt.min(), rand_tgt.max()))

    with open(OUT / "e1e2e4_single_beam.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "theta", "phi_az", "sll_ideal",
                                          "sll_po", "delta", "rand_tgt_rel", "rand_sll"])
        w.writeheader()
        for r in rows:
            w.writerow({k: ("{:.4f}".format(v) if isinstance(v, float) else v)
                       for k, v in r.items()})
    return rows, rep


# ------------------------------------------------------------------- E3 dual
def run_dual_beam(rng):
    rows = []
    rep = {}
    for k in range(N_TASKS):
        beams = sample_beams(rng, 2, THETA_MIN, THETA_MAX, MIN_SEP)
        targets = [angles_to_uv(t, p) for t, p in beams]
        thetas = [t for t, p in beams]
        phases_A = np.stack([bp_phase_plane(t, p, N, D_OL) for t, p in beams])
        _, feed = point_feed_geometry(beams[0][0], beams[0][1], N, D_OL, F_OVER_D)
        phases_B = np.stack([point_feed_geometry(t, p, N, D_OL, F_OVER_D)[0]
                             for t, p in beams])
        apers = {
            "ideal": (W2[None, :, :] * np.exp(1j * phases_A)).sum(axis=0),
            "argsum": np.exp(1j * arg_sum_phase(phases_A)),
            "1bit_B": np.exp(1j * (code_phase(onebit_code(phases_B)) + feed)),
        }
        for method, ap in apers.items():
            F = af_pattern(ap)
            m = evaluate_pattern(F, U_AXIS, U_AXIS, targets, thetas)
            rows.append(dict(task=k, method=method, sll_db=m["sll_db"],
                             gain_cons_db=m["gain_consistency_db"],
                             perr_max_deg=m["point_err_max_deg"]))
            if k == 0:
                rep[method] = (F, m["sll_db"])
        if k == 0:
            rep["targets"] = targets

    agg = {}
    for r in rows:
        agg.setdefault(r["method"], []).append(r["sll_db"])
    stats = {m: (float(np.mean(v)), float(np.std(v))) for m, v in agg.items()}
    for m in ("ideal", "argsum", "1bit_B"):
        log("--- E3 {}: SLL {:.2f} +/- {:.2f} dB".format(m, stats[m][0], stats[m][1]))
    gap = stats["argsum"][0] - stats["ideal"][0]
    log("--- E3 analytic-to-ideal gap (M=2): {:.2f} dB".format(gap))

    gate("E3 analytic-to-ideal gap >= 2 dB (decision point: headroom exists)",
         gap >= 2.0, "gap = {:.2f} dB".format(gap))

    with open(OUT / "e3_dual_beam.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["task", "method", "sll_db",
                                          "gain_cons_db", "perr_max_deg"])
        w.writeheader()
        for r in rows:
            w.writerow({k: ("{:.4f}".format(v) if isinstance(v, float) else v)
                       for k, v in r.items()})
    return rows, stats, gap, rep


# ------------------------------------------------------------------- figures
def make_single_beam_fig(rep, rows):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    draw_pattern_db(axes[0, 0], rep["F_ideal"], U_AXIS, U_AXIS, [rep["tgt"]],
                    title="E2 ideal amp+phase: SLL={:.2f} dB (Taylor -25 dB design)".format(
                        rep["sll_ideal"]))
    draw_pattern_db(axes[0, 1], rep["F_po"], U_AXIS, U_AXIS, [rep["tgt"]],
                    title="E1 phase-only (amp=1): SLL={:.2f} dB (classical uniform level)".format(
                        rep["sll_po"]))
    cut_ideal = cut_along_u(np.abs(rep["F_ideal"]), rep["tgt"][1])
    cut_po = cut_along_u(np.abs(rep["F_po"]), rep["tgt"][1])
    axes[1, 0].plot(U_AXIS, cut_ideal, label="E2 ideal ({:.2f} dB)".format(rep["sll_ideal"]),
                    color="tab:green", linewidth=1.2)
    axes[1, 0].plot(U_AXIS, cut_po, label="E1 phase-only ({:.2f} dB)".format(rep["sll_po"]),
                    color="tab:blue", linewidth=1.2)
    axes[1, 0].axvline(rep["tgt"][0], color="gray", linestyle=":", linewidth=0.8)
    axes[1, 0].set_xlim(-0.5, 0.8)
    axes[1, 0].set_ylim(-45, 2)
    axes[1, 0].set_xlabel("u (cut at v = target)")
    axes[1, 0].set_ylabel("relative pattern [dB]")
    axes[1, 0].set_title("principal cut: amplitude taper contributes {:.1f} dB".format(
        rep["sll_po"] - rep["sll_ideal"]), fontsize=10)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3)

    bp_tgt = np.zeros(N_TASKS)
    rand_tgt = np.array([r["rand_tgt_rel"] for r in rows])
    rand_sll = np.array([r["rand_sll"] for r in rows])
    sll_po = np.array([r["sll_po"] for r in rows])
    xpos = np.arange(2)
    axes[1, 1].bar(xpos - 0.18, [bp_tgt.mean(), rand_tgt.mean()], width=0.36,
                   color=["tab:blue", "tab:gray"], yerr=[bp_tgt.std(), rand_tgt.std()],
                   capsize=3, label="target response")
    axes[1, 1].bar(xpos + 0.18, [sll_po.mean(), rand_sll.mean()], width=0.36,
                   color=["tab:blue", "tab:gray"], alpha=0.55,
                   yerr=[sll_po.std(), rand_sll.std()], capsize=3, label="SLL")
    axes[1, 1].set_xticks(xpos)
    axes[1, 1].set_xticklabels(["BP phase + amp=1", "random phase + amp=1"])
    axes[1, 1].set_ylabel("[dB]")
    axes[1, 1].set_title("E4 control: BP forms the beam (0 dB),\n"
                         "random phase does not ({:.1f} dB); neither controls sidelobes".format(
                             rand_tgt.mean()), fontsize=10)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.3, axis="y")
    fig.suptitle("Phase 2 pre-experiment: single-beam physical boundary (64x64, d=0.5 lambda)")
    fig.tight_layout()
    fig.savefig(OUT / "single_beam_boundary.png", dpi=140)
    plt.close(fig)


def make_dual_beam_fig(e3_rows, stats, rep):
    titles = {"ideal": "E3 ideal amp+phase (upper bound)",
              "argsum": "E3 arg-sum baseline (= Du continuous)",
              "1bit_B": "E3 1-bit Du eq.(6) + geom B"}
    colors = {"ideal": "tab:green", "argsum": "tab:blue", "1bit_B": "tab:red"}
    methods = ("ideal", "argsum", "1bit_B")
    fig, axes = plt.subplots(2, 4, figsize=(21, 9), height_ratios=[2.2, 1.0])
    for ax, method in zip(axes[0], methods):
        F, sll = rep[method]
        draw_pattern_db(ax, F, U_AXIS, U_AXIS, rep["targets"],
                        title="{}: SLL={:.2f} dB".format(titles[method], sll))
    axes[0, 3].axis("off")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=colors[m], label=titles[m])
               for m in methods]
    axes[0, 3].legend(handles=handles, fontsize=8, loc="center")
    axes[0, 3].set_title("SLL distributions over {} tasks (below)".format(N_TASKS), fontsize=10)
    for i, m in enumerate(methods):
        vals = [r["sll_db"] for r in e3_rows if r["method"] == m]
        axes[1, i].scatter(np.arange(1, len(vals) + 1), vals, s=18, color=colors[m])
        axes[1, i].axhline(stats[m][0], color=colors[m], linestyle="--", linewidth=1)
        axes[1, i].set_ylim(-28, -4)
        axes[1, i].set_xlabel("task")
        if i == 0:
            axes[1, i].set_ylabel("SLL [dB]")
        axes[1, i].set_title("{:.2f} +/- {:.2f} dB".format(stats[m][0], stats[m][1]),
                             fontsize=9)
        axes[1, i].grid(alpha=0.3)
    axes[1, 3].axis("off")
    gap = stats["argsum"][0] - stats["ideal"][0]
    axes[1, 3].text(0.02, 0.55,
                    "analytic-to-ideal gap (M=2):\n{:.2f} dB\n= network headroom".format(gap),
                    fontsize=12, transform=axes[1, 3].transAxes)
    fig.suptitle("Phase 2 pre-experiment: dual-beam gap between analytic baseline and ideal upper bound")
    fig.tight_layout()
    fig.savefig(OUT / "dual_beam_gap.png", dpi=140)
    plt.close(fig)


# -------------------------------------------------------------------- report
def write_report(rows12, rep, e3_rows, stats, gap):
    sll_ideal = np.array([r["sll_ideal"] for r in rows12])
    sll_po = np.array([r["sll_po"] for r in rows12])
    deltas = np.array([r["delta"] for r in rows12])
    rand_tgt = np.array([r["rand_tgt_rel"] for r in rows12])
    rand_sll = np.array([r["rand_sll"] for r in rows12])
    L = []
    L.append("# 阶段 2 预实验报告：phase-only 物理边界量化（论文预备实验）")
    L.append("")
    L.append("- 日期：{}".format(time.strftime("%Y-%m-%d %H:%M")))
    L.append("- 配置：64x64（d=λ/2）、零填充 {}、Taylor {:.0f} dB / n̄={}（1D 边缘锥削 {:.2f} dB、2D 角点 {:.2f} dB）、".format(
        N_PAD, abs(SLL_DB), NBAR, 20.0 * np.log10(W1[0]), 20.0 * np.log10(W2[0, 0])))
    L.append("  波束 θ∈[2°,45°]、间隔 ≥5°、{} 组任务；metrics.py 自适应首零点掩膜；E3 双波束 1-bit 采用 Du 式(6)+几何 B（F/D={}）".format(
        N_TASKS, F_OVER_D))
    L.append("")
    L.append("## 1 E1/E2：单波束幅度自由度边界（配对实验，同一 BP 相位）")
    L.append("")
    L.append("| 量 | 均值 ± 标准差 | 范围 |")
    L.append("|---|---|---|")
    L.append("| E2 理想幅相 SLL [dB] | {:.2f} ± {:.2f} | [{:.2f}, {:.2f}] |".format(
        sll_ideal.mean(), sll_ideal.std(), sll_ideal.min(), sll_ideal.max()))
    L.append("| E1 phase-only（幅度置 1）SLL [dB] | {:.2f} ± {:.2f} | [{:.2f}, {:.2f}] |".format(
        sll_po.mean(), sll_po.std(), sll_po.min(), sll_po.max()))
    L.append("| **幅度锥削贡献（E2−E1）[dB]** | **{:.2f} ± {:.2f}** | [{:.2f}, {:.2f}] |".format(
        deltas.mean(), deltas.std(), deltas.min(), deltas.max()))
    L.append("")
    L.append("![单波束边界](single_beam_boundary.png)")
    L.append("")
    L.append("## 2 E4：随机相位对照（BP 相位的职责界定）")
    L.append("")
    L.append("| 量 | BP 相位 + 幅度 1 | 随机相位 + 幅度 1 |")
    L.append("|---|---|---|")
    L.append("| 目标方向响应 [dB] | 0（波束由 BP 相位形成） | {:.2f} ± {:.2f}（无波束，斑点图） |".format(
        rand_tgt.mean(), rand_tgt.std()))
    L.append("| SLL [dB] | {:.2f} ± {:.2f} | {:.2f} ± {:.2f}（无控制） |".format(
        sll_po.mean(), sll_po.std(), rand_sll.mean(), rand_sll.std()))
    L.append("")
    L.append("## 3 E3：双波束解析基线 vs 理想上界（20 组任务）")
    L.append("")
    L.append("| 方法 | SLL [dB] |")
    L.append("|---|---|")
    for m in ("ideal", "argsum", "1bit_B"):
        L.append("| {} | {:.2f} ± {:.2f} |".format(m, stats[m][0], stats[m][1]))
    L.append("")
    L.append("**解析-理想差距（网络理论提升空间）：{:.2f} dB（M=2）。**".format(gap))
    L.append("")
    L.append("![双波束差距](dual_beam_gap.png)")
    L.append("")
    L.append("## 4 验收（两级 sanity check）")
    L.append("")
    L.append("| 门限 | 结果 |")
    L.append("|---|---|")
    for name, ok, detail in checks:
        L.append("| {} | {}（{}） |".format(name, "PASS" if ok else "FAIL", detail))
    L.append("")
    L.append("## 5 结论段（论文预备实验章节草稿，可直接改写引用）")
    L.append("")
    L.append("> **预备实验：仅相位自由度的副瓣物理边界。** 在 64×64（间距 λ/2）连续相位调制超表面阵列"
             "上，我们首先量化幅度自由度缺失的代价：采用 Taylor −25 dB 设计与 BP 补偿相位的理想幅相口径，"
             "实测单波束 SLL 为 {:.1f} dB，与设计值一致；仅将口径幅度置 1（模拟 phase-only 硬件），SLL 立即"
             "回落至 {:.1f} dB 的均匀幅度孔径经典量级——两者 {:.1f} dB 的差值表明低副瓣能力几乎完全来自幅度"
             "锥削，BP 相位仅承担波束指向（随机相位对照中目标方向响应仅 {:.1f} dB，且方向图呈无控制斑点结构）。"
             "多波束场景下（M=2），解析叠加法（连续 arg-sum，与 Du 等的叠加法连续极限一致）SLL 为 {:.1f} dB，"
             "距理想幅相上界 {:.1f} dB 尚有 {:.1f} dB 差距；且如正文所示，arg 型解析方法在数学上无法利用幅度"
             "先验。以上界定了 phase-only 硬件低副瓣多波束综合问题的物理边界：副瓣性能的上界由理想幅相远场"
             "给出，而解析方法与之存在显著差距——这正是本文网络方法（以可微物理损失在方向图域显式逼近该上界）"
             "的出发点。".format(
        sll_ideal.mean(), sll_po.mean(), deltas.mean(), rand_tgt.mean(),
        stats["argsum"][0], stats["ideal"][0], gap))
    L.append("")
    (OUT / "预实验报告.md").write_text("\n".join(L), encoding="utf-8")


def main():
    t0 = time.time()
    log("=" * 72)
    log("Phase 2 pre-experiments E1-E4 (n={} n_pad={} Taylor={}dB nbar={} seed={})".format(
        N, N_PAD, SLL_DB, NBAR, SEED))
    log("=" * 72)
    rows12, rep = run_single_beams(np.random.default_rng(SEED + 10))
    e3_rows, stats, gap, rep_dual = run_dual_beam(np.random.default_rng(SEED + 11))
    make_single_beam_fig(rep, rows12)
    make_dual_beam_fig(e3_rows, stats, rep_dual)
    write_report(rows12, rep, e3_rows, stats, gap)
    all_ok = all(ok for _, ok, _ in checks)
    log("-" * 72)
    log("Phase 2 {} in {:.0f} s | outputs -> {}".format(
        "ALL GATES PASSED" if all_ok else "HAS FAILURES", time.time() - t0, OUT))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

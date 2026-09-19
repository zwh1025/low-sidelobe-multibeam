"""Phase 3 evaluation: network vs baselines on held-out test set (adaptive metrics)."""

import os

os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metasurf.physics import (
    angles_to_uv, fft_uv_axes, bp_phase_plane, point_feed_geometry,
    taylor_window_2d, aperture_to_pattern_np,
)
from metasurf.physics.metrics import evaluate_pattern
from metasurf.baselines.superposition import arg_sum_phase, onebit_code, code_phase
from metasurf.data.beam_dataset import sample_beams
from metasurf.model.unet import ArgSumResidualNet
from metasurf.train.trainer import pregenerate
from metasurf.eval.visualize import draw_pattern_db

N = 64
N_PAD = 512
D_OL = 0.5
SLL_DB = -25.0
NBAR = 5
SEED = 42
TEST_SEED = SEED + 777
N_TEST = 500
THETA_MIN, THETA_MAX, MIN_SEP = 2.0, 45.0, 5.0
F_OVER_D = 0.8
RUNS = ["main", "b_pat", "a_amp"]

OUT = ROOT / "results" / "phase3_network"
OUT.mkdir(parents=True, exist_ok=True)
U_AXIS = fft_uv_axes(N_PAD, D_OL)
W2 = taylor_window_2d(N, SLL_DB, NBAR)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def net_aperture(model, xb):
    return model(xb)


@torch.no_grad()
def net_patterns(model, X, chunk=100):
    model.eval()
    aps = []
    for s in range(0, X.shape[0], chunk):
        aps.append(net_aperture(model, X[s:s + chunk].to(DEVICE)).cpu())
    return torch.cat(aps).numpy()


def aperture_to_np_torch(ap):
    from metasurf.physics.array_factor import aperture_to_pattern_torch
    return aperture_to_pattern_torch(ap, N_PAD)


def baseline_apertures(beams):
    phases_A = np.stack([bp_phase_plane(t, p, N, D_OL) for t, p in beams])
    _, feed = point_feed_geometry(beams[0][0], beams[0][1], N, D_OL, F_OVER_D)
    phases_B = np.stack([point_feed_geometry(t, p, N, D_OL, F_OVER_D)[0]
                         for t, p in beams])
    ideal = (W2[None, :, :] * np.exp(1j * phases_A)).sum(axis=0)
    argsum = np.exp(1j * arg_sum_phase(phases_A))
    onebit = np.exp(1j * (code_phase(onebit_code(phases_B)) + feed))
    return dict(ideal=ideal, argsum=argsum, onebit=onebit)


def main():
    t0 = time.time()
    print("device={}".format(DEVICE), flush=True)
    Xte, uvte, idxte, beams_te = pregenerate(
        np.random.default_rng(TEST_SEED), N_TEST, W2, True, N, D_OL, N_PAD,
        THETA_MIN, THETA_MAX, MIN_SEP)

    net_rows = {}
    for run in RUNS:
        ck = torch.load(OUT / run / "ckpt_best.pt", map_location=DEVICE,
                        weights_only=False)
        model = ArgSumResidualNet(in_ch=6 if ck["cfg"]["with_amp"] else 4,
                                  width=ck["cfg"]["width"]).to(DEVICE)
        model.load_state_dict(ck["model"])
        model.eval()
        Xr = Xte if ck["cfg"]["with_amp"] else Xte[:, [0, 1, 3, 4]]
        aps = net_patterns(model, Xr)
        rows = []
        for k in range(N_TEST):
            F = aperture_to_pattern_np(aps[k], N_PAD)
            m = evaluate_pattern(F, U_AXIS, U_AXIS,
                                 [angles_to_uv(t, p) for t, p in beams_te[k]],
                                 [t for t, p in beams_te[k]])
            rows.append(m)
        net_rows[run] = rows
        slls = np.array([r["sll_db"] for r in rows])
        print("[{}] test SLL {:.2f} +/- {:.2f} dB".format(run, slls.mean(),
                                                         slls.std()), flush=True)

    base_rows = {m: [] for m in ("ideal", "argsum", "onebit")}
    for k in range(N_TEST):
        beams = beams_te[k]
        targets = [angles_to_uv(t, p) for t, p in beams]
        thetas = [t for t, p in beams]
        aps = baseline_apertures(beams)
        for m, ap in aps.items():
            mm = evaluate_pattern(aperture_to_pattern_np(ap, N_PAD), U_AXIS,
                                  U_AXIS, targets, thetas)
            base_rows[m].append(mm)
    for m in base_rows:
        slls = np.array([r["sll_db"] for r in base_rows[m]])
        print("[{}] test SLL {:.2f} +/- {:.2f} dB".format(m, slls.mean(),
                                                         slls.std()), flush=True)

    model = ArgSumResidualNet(in_ch=6, width=1.0).to(DEVICE)
    ck = torch.load(OUT / "main" / "ckpt_best.pt", map_location=DEVICE,
                    weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    with torch.no_grad():
        x1 = Xte[:1].to(DEVICE)
        for _ in range(10):
            net_aperture(model, x1)
        torch.cuda.synchronize()
        t_s = time.perf_counter()
        for _ in range(100):
            net_aperture(model, x1)
        torch.cuda.synchronize()
        t_gpu = (time.perf_counter() - t_s) / 100 * 1e3
    model_cpu = ArgSumResidualNet(in_ch=6, width=1.0)
    model_cpu.load_state_dict(ck["model"])
    model_cpu.eval()
    with torch.no_grad():
        x1c = Xte[:1]
        net_aperture(model_cpu, x1c)
        t_s = time.perf_counter()
        for _ in range(20):
            net_aperture(model_cpu, x1c)
        t_cpu = (time.perf_counter() - t_s) / 20 * 1e3

    n_perm = 100
    swap = torch.cat([torch.arange(3, 6), torch.arange(0, 3)])
    d_sll, l2 = [], []
    with torch.no_grad():
        for k in range(n_perm):
            xb = Xte[k:k + 1].to(DEVICE)
            a1 = net_aperture(model, xb)
            a2 = net_aperture(model, xb[:, swap])
            F1 = torch.abs(aperture_to_np_torch(a1))[0].cpu().numpy()
            F2 = torch.abs(aperture_to_np_torch(a2))[0].cpu().numpy()
            n1 = F1 / F1.max()
            n2 = F2 / F2.max()
            tgt = [angles_to_uv(t, p) for t, p in beams_te[k]]
            s1 = evaluate_pattern(F1, U_AXIS, U_AXIS, tgt)["sll_db"]
            s2 = evaluate_pattern(F2, U_AXIS, U_AXIS, tgt)["sll_db"]
            d_sll.append(s2 - s1)
            l2.append(float(np.linalg.norm(n1 - n2) / np.linalg.norm(n1)))
    d_sll = np.array(d_sll)
    l2 = np.array(l2)
    print("[perm] |dSLL| mean {:.3f} max {:.3f} dB | rel-L2 mean {:.2e} max {:.2e}".format(
        np.abs(d_sll).mean(), np.abs(d_sll).max(), l2.mean(), l2.max()), flush=True)

    all_methods = [("ideal", base_rows["ideal"], "ideal amp+phase (ref)"),
                   ("argsum", base_rows["argsum"], "baseline 1=2 arg-sum"),
                   ("onebit", base_rows["onebit"], "1-bit Du eq.(6)+geom B"),
                   ("main", net_rows["main"], "network (main)"),
                   ("b_pat", net_rows["b_pat"], "ablation: no L_pat"),
                   ("a_amp", net_rows["a_amp"], "ablation: no amp channel")]
    with open(OUT / "metrics_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "task", "sll_db", "perr_max_deg", "gain_cons_db"])
        for name, rows, _ in all_methods:
            for k, r in enumerate(rows):
                w.writerow([name, k, "{:.4f}".format(r["sll_db"]),
                            "{:.4f}".format(r["point_err_max_deg"]),
                            "{:.4f}".format(r["gain_consistency_db"])])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    names = [n for n, _, _ in all_methods]
    data = [np.array([r["sll_db"] for r in rows]) for _, rows, _ in all_methods]
    bp = ax.boxplot(data, labels=names, showfliers=True, patch_artist=True,
                    medianprops=dict(color="k"))
    for patch, c in zip(bp["boxes"], ["tab:green", "tab:blue", "tab:red",
                                      "tab:purple", "tab:orange", "tab:cyan"]):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.axhline(np.mean(data[0]), color="tab:green", linestyle=":", linewidth=1)
    ax.set_ylabel("SLL [dB]")
    ax.set_title("Test-set SLL distributions ({} dual-beam tasks, adaptive masks)".format(N_TEST))
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=12, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "sll_compare.png", dpi=140)
    plt.close(fig)

    sll_main = np.array([r["sll_db"] for r in net_rows["main"]])
    order = np.argsort(sll_main)
    med = order[len(order) // 2]
    worst = order[0]
    best = order[-1]
    beams = beams_te[med]
    targets = [angles_to_uv(t, p) for t, p in beams]
    aps = baseline_apertures(beams)
    ap_net = net_patterns(model, Xte[med:med + 1])[0]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    draw_pattern_db(axes[0, 0], aperture_to_pattern_np(aps["ideal"], N_PAD),
                    U_AXIS, U_AXIS, targets,
                    title="ideal amp+phase (upper bound)")
    draw_pattern_db(axes[0, 1], aperture_to_pattern_np(aps["argsum"], N_PAD),
                    U_AXIS, U_AXIS, targets, title="arg-sum baseline")
    draw_pattern_db(axes[0, 2], aperture_to_pattern_np(aps["onebit"], N_PAD),
                    U_AXIS, U_AXIS, targets, title="1-bit Du eq.(6)+geom B")
    Fm = aperture_to_pattern_np(ap_net, N_PAD)
    draw_pattern_db(axes[1, 0], Fm, U_AXIS, U_AXIS, targets,
                    title="network (median task)")
    Fw = aperture_to_pattern_np(net_patterns(model, Xte[worst:worst + 1])[0], N_PAD)
    draw_pattern_db(axes[1, 1], Fw, U_AXIS, U_AXIS,
                    [angles_to_uv(t, p) for t, p in beams_te[worst]],
                    title="network (worst task)")
    Fb = aperture_to_pattern_np(net_patterns(model, Xte[best:best + 1])[0], N_PAD)
    draw_pattern_db(axes[1, 2], Fb, U_AXIS, U_AXIS,
                    [angles_to_uv(t, p) for t, p in beams_te[best]],
                    title="network (best task)")
    fig.suptitle("Phase 3: network vs baselines (representative test tasks)")
    fig.tight_layout()
    fig.savefig(OUT / "representative.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for run, c in zip(RUNS, ["tab:purple", "tab:orange", "tab:cyan"]):
        h = json.load(open(OUT / run / "history.json", encoding="utf-8"))
        ep = [e["epoch"] for e in h["history"]]
        v = [e["val_sll"] for e in h["history"]]
        ax.plot(ep, v, label="{} (best {:.2f} dB)".format(run, h["best_val"]),
                color=c)
    ax.set_xlabel("epoch")
    ax.set_ylabel("val SLL [dB] (fixed mask)")
    ax.set_title("training curves")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=140)
    plt.close(fig)

    def stat(rows, key):
        a = np.array([r[key] for r in rows])
        return a.mean(), a.std()

    L = []
    L.append("# 阶段 3 报告：双波束网络最小验证（方向一核心）")
    L.append("")
    L.append("- 日期：{}".format(time.strftime("%Y-%m-%d %H:%M")))
    L.append("- 测试集：{} 组留出双波束任务（独立种子 {}）；指标：自适应首零点掩膜 + 亚网格插值（与基线完全同口径）".format(
        N_TEST, TEST_SEED))
    L.append("- 训练：U-Net（6ch 输入 / cos-sin 输出、幅度归一化），数据 4000/500，AdamW + 温度退火 LSE-SLL + 指向/增益/理想匹配项，交换增强 p=0.5")
    L.append("")
    L.append("## 1 测试集 SLL（主结果）")
    L.append("")
    L.append("| 方法 | SLL [dB] | 指向 max [°] | 一致性 [dB] |")
    L.append("|---|---|---|---|")
    for name, rows, label in all_methods:
        sm, ss = stat(rows, "sll_db")
        pm, _ = stat(rows, "point_err_max_deg")
        cm, _ = stat(rows, "gain_consistency_db")
        L.append("| {} | {:.2f} ± {:.2f} | {:.3f} | {:.2f} |".format(label, sm, ss, pm, cm))
    L.append("")
    L.append("![SLL 对比](sll_compare.png)")
    L.append("![代表性任务](representative.png)")
    L.append("")
    L.append("## 2 方向一可行性判据（§1.3）")
    L.append("")
    sm_arg = np.mean([r["sll_db"] for r in base_rows["argsum"]])
    sm_1b = np.mean([r["sll_db"] for r in base_rows["onebit"]])
    sm_id = np.mean([r["sll_db"] for r in base_rows["ideal"]])
    sm_main = np.mean([r["sll_db"] for r in net_rows["main"]])
    perr_main = max(r["point_err_max_deg"] for r in net_rows["main"])
    cons_main = max(r["gain_consistency_db"] for r in net_rows["main"])
    L.append("| 判据 | 阈值 | 实测 | 判定 |")
    L.append("|---|---|---|---|")
    imp = sm_arg - sm_main
    L.append("| 主判据：vs arg-sum 基线改善 | ≥2 强 / 1–2 弱 / <1 回退 | {:.2f} dB | {} |".format(
        imp, "强信号" if imp >= 2 else ("弱信号" if imp >= 1 else "回退分析")))
    L.append("| 辅助判据：vs 1-bit Du | 参考 | {:.2f} dB | — |".format(sm_1b - sm_main))
    L.append("| 与理想上界差距 | 参照 | {:.2f} dB | — |".format(sm_id - sm_main))
    L.append("| 指向误差 | ≤0.5° | {:.3f}° | {} |".format(
        perr_main, "PASS" if perr_main <= 0.5 else "FAIL"))
    L.append("| 增益一致性 | ≤1 dB（尽力） | {:.2f} dB | {} |".format(
        cons_main, "PASS" if cons_main <= 1.0 else "未达（如实报告）"))
    L.append("| 推理耗时 | GPU ≤5 ms | GPU {:.2f} ms / CPU {:.1f} ms | {} |".format(
        t_gpu, t_cpu, "PASS" if t_gpu <= 5 else "FAIL"))
    L.append("")
    L.append("## 3 消融与排列一致性")
    L.append("")
    L.append("| 变体 | SLL [dB] | 与 main 差 |")
    L.append("|---|---|---|")
    for run in RUNS:
        sm, ss = stat(net_rows[run], "sll_db")
        L.append("| {} | {:.2f} ± {:.2f} | {:+.2f} dB |".format(run, sm, ss, sm - sm_main))
    L.append("")
    L.append("排列一致性（{} 对）：|ΔSLL| 均值 {:.3f} dB / 最大 {:.3f} dB；|F| 归一化 L2 均值 {:.2e}。".format(
        n_perm, np.abs(d_sll).mean(), np.abs(d_sll).max(), l2.mean()))
    L.append("")
    L.append("## 4 训练曲线")
    L.append("")
    L.append("![训练曲线](training_curves.png)")
    L.append("")
    L.append("## 5 结论")
    L.append("")
    L.append("（见检查点 3 汇报）")
    (OUT / "方向一验证报告.md").write_text("\n".join(L), encoding="utf-8")
    print("eval done in {:.0f} s -> {}".format(time.time() - t0, OUT), flush=True)


if __name__ == "__main__":
    main()

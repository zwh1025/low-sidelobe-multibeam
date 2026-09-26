"""Phase 4 evaluation: LCS network vs baselines across beam counts (money plot).

Per M in --mtest: held-out test set (seed 819; 500 tasks for M=2, else n_test),
network + arg-sum + 1-bit Du eq.(6)+geom B + ideal amp+phase reference, all
under the same adaptive-mask protocol. Produces metrics_test.csv,
sll_vs_m.png (money plot) and an auto-judged report against plan-v2.0 sec-3.2.
"""

import os

os.environ.setdefault("MKL_THREADING_LAYER", "sequential")

import argparse
import csv
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
from metasurf.model.lcs import LCSNet
from metasurf.train.trainer import pregenerate_multi

N, N_PAD, D_OL = 64, 512, 0.5
SLL_DB, NBAR = -25.0, 5
TEST_SEED = 819
THETA_MIN, THETA_MAX, MIN_SEP = 2.0, 45.0, 5.0
F_OVER_D = 0.8
OUT = ROOT / "results" / "phase4_network"

# plan v2.0 sec 3.2 frozen absolute SLL targets (M=2 anchored to R1b-0.5;
# M>=4 derived from the 20-task phase-1 ideal references) + cons targets
TARGETS = {2: (-21.9, 1.0), 4: (-19.6, 1.0), 8: (-17.1, 1.0),
           16: (-15.3, 1.5), 32: (-13.3, 2.0)}


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


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "npu":
        torch.npu.synchronize()


def select_channels(X, m, with_amp):
    if with_amp:
        return X[:, :3 * m]
    idx = [j for i in range(m) for j in (3 * i, 3 * i + 1)]
    return X[:, idx]


def swap01_index(m, with_amp):
    c = 3 if with_amp else 2
    blocks = [list(range(b * c, (b + 1) * c)) for b in range(m)]
    order = blocks[:]


    order[0], order[1] = blocks[1], blocks[0]
    return [j for b in order for j in b]


@torch.no_grad()
def net_patterns(model, X, m, with_amp, device, chunk=32, bn_stats=None):
    model.eval()
    if bn_stats is not None:
        key = min(m, max(bn_stats))
        sd = model.state_dict()
        for k, v in bn_stats[key].items():
            sd[k] = v
        model.load_state_dict(sd)
    Xr = select_channels(X, m, with_amp)
    aps = []
    for s in range(0, Xr.shape[0], chunk):
        aps.append(model(Xr[s:s + chunk].to(device)).cpu())
    return torch.cat(aps).numpy()


def baseline_apertures(beams, W2):
    phases_A = [bp_phase_plane(t, p, N, D_OL) for t, p in beams]
    _, feed = point_feed_geometry(beams[0][0], beams[0][1], N, D_OL, F_OVER_D)
    phases_B = [point_feed_geometry(t, p, N, D_OL, F_OVER_D)[0] for t, p in beams]
    ideal = sum(W2 * np.exp(1j * ph) for ph in phases_A)
    argsum = np.exp(1j * arg_sum_phase(np.stack(phases_A)))
    onebit = np.exp(1j * (code_phase(onebit_code(np.stack(phases_B))) + feed))
    return dict(ideal=ideal, argsum=argsum, onebit=onebit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--ckpt", default="ckpt_best.pt")
    ap.add_argument("--mtest", default="2,4,8,16,32")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    device = pick_device()
    OUT.mkdir(parents=True, exist_ok=True)
    ck = torch.load(OUT / args.run / args.ckpt, map_location=device,
                    weights_only=False)
    cfg = ck["cfg"]
    m_max = cfg.get("m_max", 8)
    model = LCSNet(c_per_beam=3 if cfg["with_amp"] else 2,
                   c_enc=cfg.get("c_enc", 64), width=cfg["width"],
                   variant=cfg["variant"], m_max=m_max,
                   dec_norm=cfg.get("dec_norm", "bn")).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    bn_stats = None
    stats_path = OUT / args.run / "bn_stats_by_m.pt"
    if stats_path.exists():
        st = torch.load(stats_path, map_location=device, weights_only=False)
        bn_stats = st["stats_by_m"]
        print("bnstats: loaded per-M stats (ckpt {})".format(st.get("ckpt")),
              flush=True)
    print("run={} {} epoch={} val_sll={:.2f} variant={} device={}".format(
        args.run, args.ckpt, ck["epoch"], ck["val_sll"], cfg["variant"],
        device), flush=True)

    W2 = taylor_window_2d(N, SLL_DB, NBAR)
    u_axis = fft_uv_axes(N_PAD, D_OL)
    m_list = [int(v) for v in args.mtest.split(",")]

    # optional iterative-baseline rows (A3) merged from CSVs when present
    iter_rows = {}
    for tag, fname in (("ift", "metrics_ift.csv"),
                       ("direct", "metrics_directopt.csv")):
        p = OUT / "ift_baseline" / fname
        if p.exists():
            import collections
            by_m = collections.defaultdict(list)
            with open(p, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    by_m[int(r["m"])].append(
                        (float(r["sll"]), float(r["cons"]),
                         float(r["perr"]), float(r["sec"])))
            iter_rows[tag] = by_m

    rows = {meth: {m: [] for m in m_list}
            for meth in ("net", "ideal", "argsum", "onebit")}
    beams_by_m = {}
    for m in m_list:
        n_t = 500 if m == 2 else args.n
        Xte, uvte, idxte, m_arr, beams_te = pregenerate_multi(
            np.random.default_rng(TEST_SEED), [m], n_t, W2, True, N, D_OL,
            N_PAD, THETA_MIN, THETA_MAX, MIN_SEP, m_max=max(m, m_max))
        beams_by_m[m] = beams_te
        aps_net = net_patterns(model, Xte, m, cfg["with_amp"], device,
                               bn_stats=bn_stats)
        for k in range(n_t):
            beams = beams_te[k]
            targets = [angles_to_uv(t, p) for t, p in beams]
            thetas = [t for t, p in beams]
            aps = baseline_apertures(beams, W2)
            rows["net"][m].append(evaluate_pattern(
                aperture_to_pattern_np(aps_net[k], N_PAD), u_axis, u_axis,
                targets, thetas))
            for name, ap in aps.items():
                rows[name][m].append(evaluate_pattern(
                    aperture_to_pattern_np(ap, N_PAD), u_axis, u_axis,
                    targets, thetas))
        for name in rows:
            s = np.array([r["sll_db"] for r in rows[name][m]])
            print("[M={:2d}] {:7s} SLL {:.2f} +/- {:.2f} dB".format(
                m, name, s.mean(), s.std()), flush=True)

    with open(OUT / "metrics_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "m", "task", "sll_db", "perr_max_deg",
                    "gain_cons_db"])
        for name in rows:
            for m in m_list:
                for k, r in enumerate(rows[name][m]):
                    w.writerow([name, m, k, "{:.4f}".format(r["sll_db"]),
                                "{:.4f}".format(r["point_err_max_deg"]),
                                "{:.4f}".format(r["gain_consistency_db"])])

    # permutation consistency (swap beams 0<->1), first 50 tasks per M>=2
    perm_stats = {}
    with torch.no_grad():
        for m in m_list:
            if m < 2:
                continue
            n_perm = min(50, len(beams_by_m[m]))
            Xte, _, _, _, beams_te = pregenerate_multi(
                np.random.default_rng(TEST_SEED), [m], n_perm, W2, True, N,
                D_OL, N_PAD, THETA_MIN, THETA_MAX, MIN_SEP,
                m_max=max(m, m_max))
            sw = swap01_index(m, cfg["with_amp"])
            d_sll, l2 = [], []
            for k in range(n_perm):
                xb = select_channels(Xte, m, cfg["with_amp"]).to(device)
                a1 = model(xb[k:k + 1])
                a2 = model(xb[k:k + 1][:, sw])
                F1 = torch.abs(aperture_to_np(a1))[0].cpu().numpy()
                F2 = torch.abs(aperture_to_np(a2))[0].cpu().numpy()
                n1, n2 = F1 / F1.max(), F2 / F2.max()
                tgt = [angles_to_uv(t, p) for t, p in beams_te[k]]
                d_sll.append(evaluate_pattern(F2, u_axis, u_axis, tgt)["sll_db"]
                             - evaluate_pattern(F1, u_axis, u_axis, tgt)["sll_db"])
                l2.append(float(np.linalg.norm(n1 - n2) / np.linalg.norm(n1)))
            perm_stats[m] = (float(np.abs(d_sll).mean()),
                             float(np.abs(d_sll).max()), float(np.mean(l2)))
            print("[M={:2d}] perm |dSLL| mean {:.3f} max {:.3f} dB | rel-L2 {:.2e}".format(
                m, *perm_stats[m]), flush=True)

    # latency (M=2 test tasks): single-sample and batch-16
    Xl, _, _, _, _ = pregenerate_multi(np.random.default_rng(TEST_SEED), [2],
                                       16, W2, True, N, D_OL, N_PAD,
                                       THETA_MIN, THETA_MAX, MIN_SEP,
                                       m_max=max(2, m_max))
    Xl = select_channels(Xl, 2, cfg["with_amp"]).to(device)
    with torch.no_grad():
        for _ in range(50):
            model(Xl[:1])
        _sync(device)
        t_s = time.perf_counter()
        for _ in range(100):
            model(Xl[:1])
        _sync(device)
        t_single = (time.perf_counter() - t_s) / 100 * 1e3
        for _ in range(20):
            model(Xl)
        _sync(device)
        t_s = time.perf_counter()
        for _ in range(50):
            model(Xl)
        _sync(device)
        t_batch = (time.perf_counter() - t_s) / 50 / 16 * 1e3
    print("[latency] {} single {:.2f} ms / batch16 {:.2f} ms/sample".format(
        device.upper(), t_single, t_batch), flush=True)

    # money plot
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    styles = {"direct": ("tab:orange", "per-task direct opt (upper bound, 400 steps)"),
              "net": ("tab:purple", "LCS net (0.35 ms/sample)"),
              "ift": ("tab:gray", "IFT projection baseline"),
              "argsum": ("tab:blue", "arg-sum"),
              "onebit": ("tab:red", "1-bit Du eq.(6)+B"),
              "ideal": ("tab:green", "ideal amp+phase (ref)")}
    for name, (c, lab) in styles.items():
        if name in ("ift", "direct"):
            if name not in iter_rows:
                continue
            ms = [m for m in m_list if m in iter_rows[name]]
            mu = [np.mean([v[0] for v in iter_rows[name][m]]) for m in ms]
            sd = [np.std([v[0] for v in iter_rows[name][m]]) for m in ms]
            ls = "--" if name == "ift" else ":"
            ax.errorbar(ms, mu, yerr=sd, marker="o", color=c, label=lab,
                        capsize=3, linestyle=ls)
        else:
            mu = [np.mean([r["sll_db"] for r in rows[name][m]]) for m in m_list]
            sd = [np.std([r["sll_db"] for r in rows[name][m]]) for m in m_list]
            ax.errorbar(m_list, mu, yerr=sd, marker="o", color=c, label=lab,
                        capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(m_list)
    ax.set_xticklabels([str(m) for m in m_list])
    ax.set_xlabel("beam count M")
    ax.set_ylabel("SLL [dB]")
    ax.set_title("SLL vs beam count (test seed 819, adaptive masks)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    cons_mu = [np.mean([r["gain_consistency_db"] for r in rows["net"][m]])
               for m in m_list]
    cons_sd = [np.std([r["gain_consistency_db"] for r in rows["net"][m]])
               for m in m_list]
    ax2.errorbar(m_list, cons_mu, yerr=cons_sd, marker="s", color="tab:purple",
                 capsize=3, label="net")
    if "direct" in iter_rows:
        ms = [m for m in m_list if m in iter_rows["direct"]]
        ax2.errorbar(ms, [np.mean([v[1] for v in iter_rows["direct"][m]])
                          for m in ms],
                     yerr=[np.std([v[1] for v in iter_rows["direct"][m]])
                           for m in ms],
                     marker="^", color="tab:orange", capsize=3, linestyle=":",
                     label="direct opt")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(m_list)
    ax2.set_xticklabels([str(m) for m in m_list])
    ax2.set_xlabel("beam count M")
    ax2.set_ylabel("gain consistency [dB]")
    ax2.set_title("beam balance")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "sll_vs_m.png", dpi=140)
    plt.close(fig)

    # report with auto-judgment against plan v2.0 sec 3.2
    L = ["# 阶段 4 评测报告：LCS 多波束（金钱图口径）", "",
         "- 日期：{}".format(time.strftime("%Y-%m-%d %H:%M")),
         "- run={} ckpt={}（variant {}，epoch {}，val_sll {:.2f} dB）".format(
             args.run, args.ckpt, cfg["variant"], ck["epoch"], ck["val_sll"]),
         "- 测试：种子 819（M=2 为 500 任务，其余 {} 任务/M）；自适应首零点掩膜 + 亚网格插值".format(args.n),
         "- 训练 M 分布：{}".format(cfg.get("m_list")),
         "- 迭代基线（A3，100 任务/M，几何 A）：IFT 投影 / 逐任务直接优化（同损失，arg-sum 起点，400 步 Adam）", ""]
    L.append("## 1 SLL vs M（主结果）")
    L.append("")
    L.append("| M | net | direct-opt(ref) | IFT | arg-sum | 1-bit Du(B) | ideal(ref) | net 距 ideal | 改善 vs arg-sum |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for m in m_list:
        mu = {k: np.mean([r["sll_db"] for r in rows[k][m]]) for k in rows}
        dstr = "{:.2f}".format(np.mean([v[0] for v in iter_rows["direct"][m]])) \
            if "direct" in iter_rows and m in iter_rows["direct"] else "—"
        istr = "{:.2f}".format(np.mean([v[0] for v in iter_rows["ift"][m]])) \
            if "ift" in iter_rows and m in iter_rows["ift"] else "—"
        L.append("| {} | {:.2f} | {} | {} | {:.2f} | {:.2f} | {:.2f} | {:+.2f} dB | {:+.2f} dB |".format(
            m, mu["net"], dstr, istr, mu["argsum"], mu["onebit"], mu["ideal"],
            mu["net"] - mu["ideal"], mu["argsum"] - mu["net"]))
    if "direct" in iter_rows:
        L.append("")
        L.append("耗时对照：direct-opt {:.1f}–{:.1f} s/任务（M 递增） vs net 批16 {:.2f} ms/样本（~10⁴–10⁵× 加速）；net 距 direct-opt {:.1f}–{:.1f} dB（M 递增）。".format(
            min(np.mean([v[3] for v in vv]) for vv in iter_rows["direct"].values()),
            max(np.mean([v[3] for v in vv]) for vv in iter_rows["direct"].values()),
            t_batch,
            min(np.mean([v[0] for v in iter_rows["direct"][m]])
                - np.mean([r["sll_db"] for r in rows["net"][m]])
                for m in m_list if m in iter_rows["direct"]),
            max(np.mean([v[0] for v in iter_rows["direct"][m]])
                - np.mean([r["sll_db"] for r in rows["net"][m]])
                for m in m_list if m in iter_rows["direct"])))
    L.append("")
    L.append("![SLL vs M](sll_vs_m.png)")
    L.append("")
    L.append("## 2 判据核对（执行计划 v2.0 §3.2）")
    L.append("")
    L.append("| M | SLL 目标 | 实测 | 判定 | cons 目标 | 实测 mean/max | 判定 | 指向 max |")
    L.append("|---|---|---|---|---|---|---|---|")
    for m in m_list:
        net = rows["net"][m]
        sll = np.mean([r["sll_db"] for r in net])
        ideal = np.mean([r["sll_db"] for r in rows["ideal"][m]])
        cons = np.array([r["gain_consistency_db"] for r in net])
        perr = max(r["point_err_max_deg"] for r in net)
        if m not in TARGETS:
            L.append("| {} | （sanity：≥−13.2 量级） | {:.2f} | — | — | — | — | {:.3f}° |".format(m, sll, perr))
            continue
        tgt, cons_t = TARGETS[m]
        j1 = "PASS" if sll <= tgt else "FAIL"
        j2 = "PASS" if cons.mean() <= cons_t else "FAIL"
        L.append("| {} | ≤{:.1f}（§3.2 冻结，抑制至少 {:.1f} dB） | {:.2f} | {} | ≤{:.1f} | {:.2f}/{:.2f} | {} | {:.3f}° |".format(
            m, tgt, -tgt, sll, j1, cons_t, cons.mean(), cons.max(), j2,
            perr))
    L.append("")
    L.append("## 3 排列一致性（交换波束 0↔1）")
    L.append("")
    for m, (dm, dx, l2m) in perm_stats.items():
        L.append("- M={}: |ΔSLL| mean {:.3f} / max {:.3f} dB；|F| rel-L2 {:.2e}".format(
            m, dm, dx, l2m))
    L.append("")
    L.append("## 4 推理耗时")
    L.append("")
    L.append("- {} 单样本 {:.2f} ms / 批16 {:.2f} ms/样本".format(
        device.upper(), t_single, t_batch))
    L.append("")
    (OUT / "阶段4评测报告.md").write_text("\n".join(L), encoding="utf-8")
    print("eval done -> {}".format(OUT), flush=True)


def aperture_to_np(ap):
    from metasurf.physics.array_factor import aperture_to_pattern_torch
    return aperture_to_pattern_torch(ap, N_PAD)


if __name__ == "__main__":
    main()

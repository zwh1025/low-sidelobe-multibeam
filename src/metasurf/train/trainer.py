"""Training pipeline: dataset pregeneration, fast validation, training loop."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..physics.grid import angles_to_uv, fft_uv_axes
from ..physics.bp_phase import bp_phase_plane
from ..physics.array_factor import aperture_to_pattern_torch
from ..data.beam_dataset import sample_beams
from .losses import PhysicsLoss


def pregenerate(rng, n_tasks, window2d, with_amp, n=64, d_ol=0.5, n_pad=512,
                theta_min=2.0, theta_max=45.0, min_sep=5.0):
    """Pre-generate task tensors (held in RAM): inputs, target uv, target grid indices."""
    from ..physics.grid import angles_to_uv as a2uv
    c_per_beam = 3 if with_amp else 2
    X = np.zeros((n_tasks, c_per_beam * 2, n, n), dtype=np.float32)
    tgt_uv = np.zeros((n_tasks, 2, 2), dtype=np.float64)
    tgt_idx = np.zeros((n_tasks, 2, 2), dtype=np.int64)
    beams_all = []
    u_axis = fft_uv_axes(n_pad, d_ol)
    for k in range(n_tasks):
        while True:
            try:
                beams = sample_beams(rng, 2, theta_min, theta_max, min_sep)
                break
            except RuntimeError:
                continue
        beams_all.append(beams)
        for i, (t, p) in enumerate(beams):
            ph = bp_phase_plane(t, p, n, d_ol)
            base = i * c_per_beam
            X[k, base] = np.cos(ph)
            X[k, base + 1] = np.sin(ph)
            if with_amp:
                X[k, base + 2] = window2d
        uv = [a2uv(t, p) for t, p in beams]
        tgt_uv[k] = uv
        for i, (u0, v0) in enumerate(uv):
            tgt_idx[k, i, 0] = int(np.argmin(np.abs(u_axis - u0)))
            tgt_idx[k, i, 1] = int(np.argmin(np.abs(u_axis - v0)))
    return (torch.from_numpy(X), torch.from_numpy(tgt_uv),
            torch.from_numpy(tgt_idx), beams_all)


@torch.no_grad()
def fast_val_sll(model, Xv, uvv, phys, device, chunk=100, r=0.040):
    """Fast fixed-mask SLL on GPU (early-stopping metric; final eval uses metrics.py)."""
    model.eval()
    vals = []
    for s in range(0, Xv.shape[0], chunk):
        xb = Xv[s:s + chunk].to(device)
        uvb = uvv[s:s + chunk].to(device)
        ap = model(xb)
        F_abs = torch.abs(phys._pattern(ap))
        Fn = F_abs / (F_abs.amax(dim=(-2, -1), keepdim=True) + 1e-12)
        B = xb.shape[0]
        main = torch.zeros((B, phys.n_pad, phys.n_pad), dtype=torch.bool,
                           device=device)
        for i in range(uvb.shape[1]):
            u0 = uvb[:, i, 0].view(-1, 1, 1)
            v0 = uvb[:, i, 1].view(-1, 1, 1)
            main |= (phys.U[None] - u0) ** 2 + (phys.V[None] - v0) ** 2 <= r ** 2
        sl = phys.visible[None] & ~main
        sl_max = (Fn * sl).flatten(1).amax(dim=1)
        sll = 20.0 * torch.log10(sl_max + 1e-12)
        vals.append(sll.cpu())
    model.train()
    return float(torch.cat(vals).mean())


def train_run(run_name, cfg, device="cuda"):
    """Train one run; returns history. Saves ckpt_best.pt + history.json."""
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    g = torch.Generator().manual_seed(cfg["seed"] + 1)
    t_start = time.time()

    out = Path(cfg["out_root"]) / run_name
    out.mkdir(parents=True, exist_ok=True)

    from ..physics.taylor import taylor_window_2d
    W2 = taylor_window_2d(cfg["n"], cfg["sll_db"], cfg["nbar"])

    Xtr, uvtr, idxtr, _ = pregenerate(
        np.random.default_rng(cfg["data_seed"]), cfg["n_train"], W2,
        cfg["with_amp"], cfg["n"], cfg["d_ol"], cfg["n_pad"],
        cfg["theta_min"], cfg["theta_max"], cfg["min_sep"])
    Xva, uvva, idxva, _ = pregenerate(
        np.random.default_rng(cfg["data_seed"] + 1), cfg["n_val"], W2,
        cfg["with_amp"], cfg["n"], cfg["d_ol"], cfg["n_pad"],
        cfg["theta_min"], cfg["theta_max"], cfg["min_sep"])

    from ..model.unet import ArgSumResidualNet
    model = ArgSumResidualNet(in_ch=Xtr.shape[1], width=cfg["width"]).to(device)
    loss_fn = PhysicsLoss(n=cfg["n"], n_pad=cfg["n_pad"], d_ol=cfg["d_ol"],
                          r_train=cfg["r_train"], alpha=cfg["alpha"],
                          beta=cfg["beta"], gamma=cfg["gamma"],
                          delta=cfg["delta"], use_pat=cfg["use_pat"],
                          gamma2=cfg["gamma2"], sll_domain=cfg["sll_domain"],
                          tau_start=cfg["tau_start"], tau_end=cfg["tau_end"],
                          l_beam_mode=cfg["l_beam_mode"],
                          beam_floor=cfg["beam_floor"],
                          stage1_frac=cfg["stage1_frac"],
                          warmup_frac=cfg["warmup_frac"], window2d=W2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    half = Xtr.shape[1] // 2
    swap_idx = torch.cat([torch.arange(half, 2 * half), torch.arange(0, half)])
    N = Xtr.shape[0]
    B = cfg["batch"]
    history = []
    best_val = float("inf")
    patience = 0
    for epoch in range(cfg["epochs"]):
        t_ep = time.time()
        model.train()
        perm = torch.randperm(N, generator=g)
        ep_loss = 0.0
        nb = 0
        for s in range(0, N, B):
            idx = perm[s:s + B]
            xb = Xtr[idx].to(device, non_blocking=True)
            if cfg["swap_augment"]:
                sw = torch.rand(xb.shape[0], generator=g) < 0.5
                xb[sw] = xb[sw][:, swap_idx]
            uvb = uvtr[idx].to(device)
            idxb = idxtr[idx].to(device)
            pred = model(xb)
            frac = (epoch + s / N) / cfg["epochs"]
            loss, parts = loss_fn(pred, xb, uvb, idxb, frac)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            ep_loss += float(loss)
            nb += 1
        sched.step()
        val_sll = fast_val_sll(model, Xva, uvva, loss_fn, device,
                               r=cfg["r_train"])
        history.append(dict(epoch=epoch, train_loss=ep_loss / nb,
                            val_sll=val_sll,
                            **{k: v for k, v in parts.items()}))
        if val_sll < best_val - 1e-4:
            best_val = val_sll
            patience = 0
            torch.save(dict(model=model.state_dict(), epoch=epoch,
                            val_sll=val_sll, cfg=cfg), out / "ckpt_best.pt")
        elif (epoch / cfg["epochs"]) >= cfg["warmup_frac"]:
            patience += 1
        if epoch % 10 == 0 or patience >= cfg["patience"]:
            rs = float(model.res_scale) if hasattr(model, "res_scale") else float("nan")
            print("[{}] epoch {:3d} loss {:.4f} (sll {:.4f} dir {:.4f} gain {:.4f} pat {:.4f} beam {:.4f} tau {:.4f})"
                  " val_sll {:.2f} dB best {:.2f} rs {:.3f} | {:.1f}s".format(
                      run_name, epoch, ep_loss / nb, parts["l_sll"], parts["l_dir"],
                      parts["l_gain"], parts["l_pat"], parts["l_beam"], parts["tau"],
                      val_sll, best_val, rs, time.time() - t_ep), flush=True)
        if patience >= cfg["patience"]:
            print("[{}] early stop at epoch {}".format(run_name, epoch), flush=True)
            break
    torch.save(dict(model=model.state_dict(), epoch=epoch, val_sll=best_val,
                    cfg=cfg), out / "ckpt_last.pt")
    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump(dict(history=history, best_val=best_val,
                       elapsed_s=time.time() - t_start), f, indent=1)
    print("[{}] done: best val SLL {:.2f} dB, {:.0f} s".format(
        run_name, best_val, time.time() - t_start), flush=True)
    return history


# ---------------------------------------------------------------------------
# Phase 4: multi-M (learned complex superposition) pipeline.
# Storage is padded to m_max beams; batches are bucketed so that all tasks in
# one batch share the same M (slicing to c_per_beam*M before forward/loss).
# ---------------------------------------------------------------------------

def pregenerate_multi(rng, m_list, n_tasks, window2d, with_amp, n=64, d_ol=0.5,
                      n_pad=512, theta_min=2.0, theta_max=45.0, min_sep=5.0,
                      m_max=None, geometry="A", f_over_d=0.8):
    """Multi-M task pregeneration; per-task M drawn uniformly from m_list.

    With len(m_list) == 1 the RNG stream is identical to phase-3 pregenerate
    (v9 / R1 comparability). geometry "B" (Du point-source reflectarray) uses
    point-feed BP phases as network inputs (feed position fixed, f_over_D).
    Returns (X, tgt_uv, tgt_idx, m_arr, beams_all) with X padded to
    c_per_beam * m_max channels.
    """
    from ..physics.grid import angles_to_uv as a2uv
    from ..physics.bp_phase import point_feed_geometry
    if m_max is None:
        m_max = max(m_list)
    c_pb = 3 if with_amp else 2
    X = np.zeros((n_tasks, c_pb * m_max, n, n), dtype=np.float32)
    tgt_uv = np.zeros((n_tasks, m_max, 2), dtype=np.float64)
    tgt_idx = np.zeros((n_tasks, m_max, 2), dtype=np.int64)
    m_arr = np.zeros(n_tasks, dtype=np.int64)
    beams_all = []
    u_axis = fft_uv_axes(n_pad, d_ol)
    single = len(m_list) == 1
    for k in range(n_tasks):
        m = m_list[0] if single else m_list[int(rng.integers(0, len(m_list)))]
        while True:
            try:
                beams = sample_beams(rng, m, theta_min, theta_max, min_sep)
                break
            except RuntimeError:
                continue
        m_arr[k] = m
        beams_all.append(beams)
        for i, (t, p) in enumerate(beams):
            if geometry == "B":
                ph, _ = point_feed_geometry(t, p, n, d_ol, f_over_d)
            else:
                ph = bp_phase_plane(t, p, n, d_ol)
            base = i * c_pb
            X[k, base] = np.cos(ph)
            X[k, base + 1] = np.sin(ph)
            if with_amp:
                X[k, base + 2] = window2d
            uv = a2uv(t, p)
            tgt_uv[k, i] = uv
            tgt_idx[k, i, 0] = int(np.argmin(np.abs(u_axis - uv[0])))
            tgt_idx[k, i, 1] = int(np.argmin(np.abs(u_axis - uv[1])))
    return (torch.from_numpy(X), torch.from_numpy(tgt_uv),
            torch.from_numpy(tgt_idx), m_arr, beams_all)


def bucketed_batches(m_arr, batch_size, generator=None):
    """Yield (M, LongTensor indices); all tasks in one batch share the same M."""
    m_t = torch.as_tensor(np.asarray(m_arr), dtype=torch.long)
    out = []
    for m in torch.unique(m_t).tolist():
        idx = torch.nonzero(m_t == m).flatten()
        if generator is not None:
            idx = idx[torch.randperm(idx.numel(), generator=generator)]
        out.append((int(m), idx))
    if generator is not None and len(out) > 1:
        order = torch.randperm(len(out), generator=generator).tolist()
        out = [out[i] for i in order]
    for m, idx in out:
        for s in range(0, idx.numel(), batch_size):
            yield m, idx[s:s + batch_size]


@torch.no_grad()
def fast_val_sll_multi(model, Xv, uvv, m_arr, with_amp, phys, device, batch=32,
                       r=0.040):
    """Fixed-mask SLL over mixed-M val tasks (early stopping metric).

    Uses train-mode BN (per-batch statistics, correct per same-M bucket) with
    dropout disabled: single averaged running stats are wrong under bucketed
    multi-M training (see plan v2.0 sec 8.4), so eval-mode val would pollute
    the early-stopping signal. Deployment uses per-M calibrated stats instead
    (scripts/calibrate_bn.py).
    """
    drop, p_backup = getattr(model, "decoder", None), None
    if drop is not None and hasattr(drop, "drop"):
        p_backup = drop.drop.p
        drop.drop.p = 0.0
    model.train()
    m_arr = np.asarray(m_arr)
    c_pb = 3 if with_amp else 2
    vals = []
    for m in np.unique(m_arr):
        sel = np.nonzero(m_arr == m)[0]
        Xs = Xv[sel][:, :c_pb * m]
        uvs = uvv[sel][:, :m]
        for s in range(0, Xs.shape[0], batch):
            xb = Xs[s:s + batch].to(device)
            uvb = uvs[s:s + batch].to(device)
            ap = model(xb)
            F_abs = torch.abs(phys._pattern(ap))
            Fn = F_abs / (F_abs.amax(dim=(-2, -1), keepdim=True) + 1e-12)
            B = xb.shape[0]
            main = torch.zeros((B, phys.n_pad, phys.n_pad), dtype=torch.bool,
                               device=device)
            for i in range(m):
                u0 = uvb[:, i, 0].view(-1, 1, 1)
                v0 = uvb[:, i, 1].view(-1, 1, 1)
                main |= (phys.U[None] - u0) ** 2 + (phys.V[None] - v0) ** 2 <= r ** 2
            sl = phys.visible[None] & ~main
            sl_max = (Fn * sl).flatten(1).amax(dim=1)
            vals.append((20.0 * torch.log10(sl_max + 1e-12)).cpu())
    if p_backup is not None:
        drop.drop.p = p_backup
    return float(torch.cat(vals).mean())


def train_run_multi(run_name, cfg, device="cuda"):
    """Phase-4 training: LCSNet over bucketed multi-M batches.

    Extra cfg keys vs train_run: m_list, m_max, variant, c_enc, gamma2_mode
    ("fixed" | "dual" Lagrangian ascent on the hinge violation).
    """
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    g = torch.Generator().manual_seed(cfg["seed"] + 1)
    t_start = time.time()

    out = Path(cfg["out_root"]) / run_name
    out.mkdir(parents=True, exist_ok=True)

    from ..physics.taylor import taylor_window_2d
    from ..physics.bp_phase import point_feed_geometry
    W2 = taylor_window_2d(cfg["n"], cfg["sll_db"], cfg["nbar"])
    m_list = list(cfg["m_list"])
    m_max = cfg.get("m_max", max(m_list))
    c_pb = 3 if cfg["with_amp"] else 2
    geometry = cfg.get("geometry", "A")
    feed_np = None
    if geometry == "B":
        feed_np = point_feed_geometry(10.0, 0.0, cfg["n"], cfg["d_ol"],
                                      cfg.get("f_over_d", 0.8))[1]

    Xtr, uvtr, idxtr, mtr, _ = pregenerate_multi(
        np.random.default_rng(cfg["data_seed"]), m_list, cfg["n_train"], W2,
        cfg["with_amp"], cfg["n"], cfg["d_ol"], cfg["n_pad"],
        cfg["theta_min"], cfg["theta_max"], cfg["min_sep"], m_max=m_max,
        geometry=geometry, f_over_d=cfg.get("f_over_d", 0.8))
    Xva, uvva, idxva, mva, _ = pregenerate_multi(
        np.random.default_rng(cfg["data_seed"] + 1), m_list, cfg["n_val"], W2,
        cfg["with_amp"], cfg["n"], cfg["d_ol"], cfg["n_pad"],
        cfg["theta_min"], cfg["theta_max"], cfg["min_sep"], m_max=m_max,
        geometry=geometry, f_over_d=cfg.get("f_over_d", 0.8))

    from ..model.lcs import LCSNet
    model = LCSNet(c_per_beam=c_pb, c_enc=cfg.get("c_enc", 64),
                   width=cfg["width"], variant=cfg["variant"],
                   m_max=m_max, dec_norm=cfg.get("dec_norm", "bn"),
                   quantize=cfg.get("quantize", "none")).to(device)
    if cfg.get("init_from"):
        sd = torch.load(cfg["init_from"], map_location=device,
                        weights_only=False)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        print("[{}] warm start from {}".format(run_name, cfg["init_from"]),
              flush=True)
    loss_fn = PhysicsLoss(n=cfg["n"], n_pad=cfg["n_pad"], d_ol=cfg["d_ol"],
                          r_train=cfg["r_train"], alpha=cfg["alpha"],
                          beta=cfg["beta"], gamma=cfg["gamma"],
                          delta=cfg["delta"], use_pat=cfg["use_pat"],
                          gamma2=cfg["gamma2"], sll_domain=cfg["sll_domain"],
                          tau_start=cfg["tau_start"], tau_end=cfg["tau_end"],
                          l_beam_mode=cfg["l_beam_mode"],
                          beam_floor=cfg["beam_floor"],
                          stage1_frac=cfg["stage1_frac"],
                          warmup_frac=cfg["warmup_frac"], window2d=W2,
                          dir2=cfg.get("dir2", 0.0),
                          dir_floor=cfg.get("dir_floor", 0.05),
                          sll_floor_db=cfg.get("sll_floor_db"),
                          feed_phase=feed_np).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    gamma2_mode = cfg.get("gamma2_mode", "fixed")
    gamma2_eff = float(cfg["gamma2"])
    N = Xtr.shape[0]
    B = cfg["batch"]
    history = []
    best_val = float("inf")
    patience = 0
    for epoch in range(cfg["epochs"]):
        t_ep = time.time()
        model.train()
        ep_loss = 0.0
        nb = 0
        ep_parts = {}
        for m, idx in bucketed_batches(mtr, B, generator=g):
            xb = Xtr[idx][:, :c_pb * m].to(device, non_blocking=True)
            uvb = uvtr[idx][:, :m].to(device)
            idxb = idxtr[idx][:, :m].to(device)
            pred = model(xb)
            frac = (epoch + nb / max(1, N // B)) / cfg["epochs"]
            loss, parts = loss_fn(pred, xb, uvb, idxb, frac)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            opt.step()
            ep_loss += float(loss)
            nb += 1
            for k, v in parts.items():
                ep_parts[k] = ep_parts.get(k, 0.0) + v
        for k in ep_parts:
            ep_parts[k] /= max(1, nb)
        sched.step()
        if gamma2_mode == "dual":
            gamma2_eff = min(200.0, gamma2_eff + cfg.get("dual_eta", 50.0)
                             * max(0.0, ep_parts.get("l_beam", 0.0)))
            loss_fn.gamma2 = gamma2_eff
        val_sll = fast_val_sll_multi(model, Xva, uvva, mva, cfg["with_amp"],
                                     loss_fn, device, batch=B, r=cfg["r_train"])
        history.append(dict(epoch=epoch, train_loss=ep_loss / nb,
                            val_sll=val_sll, gamma2=gamma2_eff,
                            **{k: v for k, v in ep_parts.items()}))
        if val_sll < best_val - 1e-4:
            best_val = val_sll
            patience = 0
            torch.save(dict(model=model.state_dict(), epoch=epoch,
                            val_sll=val_sll, cfg=cfg), out / "ckpt_best.pt")
        elif (epoch / cfg["epochs"]) >= cfg["warmup_frac"]:
            patience += 1
        if epoch % 10 == 0 or patience >= cfg["patience"]:
            print("[{}] epoch {:3d} loss {:.4f} (sll {:.4f} dir {:.4f} dirh {:.4f} gain {:.4f} pat {:.4f} beam {:.4f} tau {:.4f})"
                  " val_sll {:.2f} dB best {:.2f} g2 {:.1f} | {:.1f}s".format(
                      run_name, epoch, ep_loss / nb, ep_parts.get("l_sll", float("nan")),
                      ep_parts.get("l_dir", float("nan")),
                      ep_parts.get("l_dirh", float("nan")),
                      ep_parts.get("l_gain", float("nan")),
                      ep_parts.get("l_pat", float("nan")), ep_parts.get("l_beam", float("nan")),
                      ep_parts.get("tau", float("nan")), val_sll, best_val,
                      gamma2_eff, time.time() - t_ep), flush=True)
        if patience >= cfg["patience"]:
            print("[{}] early stop at epoch {}".format(run_name, epoch), flush=True)
            break
    torch.save(dict(model=model.state_dict(), epoch=epoch, val_sll=best_val,
                    cfg=cfg), out / "ckpt_last.pt")
    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump(dict(history=history, best_val=best_val,
                       elapsed_s=time.time() - t_start), f, indent=1)
    print("[{}] done: best val SLL {:.2f} dB, {:.0f} s".format(
        run_name, best_val, time.time() - t_start), flush=True)
    return history

# 阶段 0′ 自检报告（2026-09-21，NPU）

命令：python scripts/run_phase4_selftest.py（9/9 PASS）+ 端到端冒烟 run_phase4_train --smoke（M∈{1,8}）+ run_phase4_eval --mtest 2,8 --n 16（已验证后清理）

```
device=npu
[PASS] T1 degeneracy LCS-S max|ap-argsum| = 0.00e+00
[PASS] T1 degeneracy LCS-D max|ap-argsum| = 0.00e+00
[PASS] T2 RNG stream parity (beams) 
[PASS] T2 RNG stream parity (X/uv) 
  x = torch.where(sl, x, torch.full_like(x, -1e9))
[PASS] T3 fwd+bwd M=1 loss=-3.7194 gmax=1.56e-02
[PASS] T3 fwd+bwd M=3 loss=-1.7668 gmax=5.44e-01
[PASS] T3 fwd+bwd M=8 loss=4.0000 gmax=9.34e-01
[PASS] T4 bucketing coverage batches cover all tasks once, same-M within batch
[PASS] T4 bucketing determinism 
--- selftest: 9/9 PASS ---
```

# v6 归档说明（2026-09-19 服务器交接核查）

## 核查结论

`main_v6_power/`、`b_pat_v6_power/`、`a_amp_v6_power/` 三个目录中**不含 v6 真实训练产物**：

1. **ckpt（*.pt）从未进入本仓库**——`.gitignore` 排除 `*.pt`，GitHub 上传与服务器克隆均不含；
2. **history.json 在本地打包前已被 `--smoke` 覆盖**——git 证据：首次提交 d5127e0 中 `main/history.json` 即为 2-epoch / 4 s 冒烟记录（best_val −9.79）；同批受影响的还有归档目录 `*_v2_imbalance`、`*_v3_dbbasin`（同为 2-epoch 冒烟记录）；`*_v1_degenerate`、`*_v4_warmupkill`、`*_v5_softtau` 的 history.json 为真实记录；
3. 2026-09-19 13:37 服务器侧 NPU 冒烟曾在原 `main/b_pat/a_amp` 写入 ckpt，归档时已删除（属今日冒烟产物，非 v6）。

## v6 权威记录（不受影响）

- 逐 epoch 训练日志：`train_v6.log`（含 loss 分量与 val_sll）
- 评测口径：`方向一验证报告_v6_power.md`、`metrics_test_v6_power.csv`（测试集 500 任务，自适应掩膜）
- 结论：v6 残差塌缩（Δ≈0），网络 ≡ arg-sum 基线（−9.70 dB），详见执行计划 §14.4

## 若需在服务器复评 v1–v6 的 ckpt

需从本地机（`D:\学习\单波束到多波束\results\phase3_network\`）重新上传对应 `*_vN_*/ckpt_best.pt`。

# v9 ckpt 恢复说明（2026-09-21 阶段 0′ 事故记录）

## 事故

2026-09-21 16:30 前后，为回归验证 `losses.py` M 通用化修改，在服务器上执行了
`python scripts/run_phase3_train.py --smoke --runs main`。该脚本的 `--smoke` 模式
**硬编码改跑全部三个 run**（`run_phase3_train.py:56`：`args.runs = "main,b_pat,a_amp"`），
且执行前未按 §5.3 归档规范先行改名保护——导致：

- `main/`（v9 收官模型）、`b_pat/`、`a_amp/`（v9 配置消融）的
  `ckpt_best.pt` / `ckpt_last.pt` 被 2-epoch 冒烟模型**覆盖**；
- 随后清理时 `main/` 整目录被删除重建。

## 已恢复（服务器侧）

- 三个 run 的 `history.json` 已从 git（提交 `1b6c574`）恢复，为 v9 真实记录
  （main −22.44 / b_pat −22.26 / a_amp −21.35，各 150 epoch）；
- 冒烟垃圾 ckpt 已删除，三个目录现仅含 history.json；
- 训练日志（`train_v7_route2/v8_floor09/v9_floor095/v9_ablations.log`）、
  评测产物（`metrics_test.csv`、`方向一验证报告.md`、3 图）**未受影响**。

## 待恢复（需从用户备份）

`*.pt` 被 `.gitignore` 排除、未入仓库。请从 2026-09-21 之前的整体备份
（用户在 GitHub 推送前已备份整个文件夹）中恢复以下三个文件：

```
<备份>/results/phase3_network/main/ckpt_best.pt     # v9 收官模型（论文主结果）
<备份>/results/phase3_network/b_pat/ckpt_best.pt    # 消融：去理想匹配项
<备份>/results/phase3_network/a_amp/ckpt_best.pt    # 消融：去幅度通道
```

（`ckpt_last.pt` 可选；`main_v7_route2/`、`main_v8_floor09/` 归档完好未受影响。）

## 教训（已列入阶段 4 约定）

1. 任何写入 `results/<phase>/<run>/` 的脚本（含 `--smoke`）执行前必须确认目标目录
   已归档或为空——phase4 入口已按此约定设计（`--run` 显式命名，不硬编码 run 列表）；
2. 回归验证优先使用不落盘的单元测试（`run_phase4_selftest.py` 模式），
   仅在必要时才跑入口冒烟。

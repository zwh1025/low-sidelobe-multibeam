# 单波束到低副瓣多波束编码 —— 方向一：幅度-相位双通道输入的连续相位多波束综合网络

以连续相位调制超表面（64×64，单元幅度恒定，仅相位自由度）为对象，研究**"输入低副瓣单波束幅相先验（BP 补偿相位 + Taylor 幅度窗）→ 输出低副瓣多波束连续相位编码"**的神经网络映射方法。基线为 Du et al., AWPL 2025 的叠加法（SLL ≈ −9.7 dB，无副瓣控制）；目标是在毫秒级推理下逼近理想幅相参照（≈ −24 dB）。

**主文档：[`执行计划.md`](执行计划.md)**（方案、判据、逐阶段实测记录 §14——所有结论以该文档为准）；**阶段 4（多波束泛化 + 1-bit + 迭代基线）：[`执行计划v2.0.md`](执行计划v2.0.md)**；**项目完整方案说明（letter 骨架，参照 Du 论文组织）：[`方案计划书.md`](方案计划书.md)**。

## 当前结果速览（详见 执行计划.md §14 / 执行计划v2.0.md §8）

| 阶段 | 内容 | 状态 | 关键数字 |
|---|---|---|---|
| 0 | 工程脚手架 + 物理内核 | ✅ 完成 | Taylor 理想幅相 SLL −25.02 dB（设计值命中）；numpy/torch 一致性 1.2e-7 |
| 1 | 三连检 + 叠加法基线 + Du 2025 复现 | ✅ 完成 | 基线 arg-sum SLL −9.7 dB；Du 对标全部量级一致；两项新发现（基线②≡①、式(6)几何依赖性） |
| 2 | 预实验（物理边界） | ✅ 完成 | 幅度锥削贡献 11.6 dB；解析-理想差距 14.4 dB（情况 A） |
| 3 | 双波束网络最小验证 | ✅ **闭环（检查点 3 达成）** | v9（dB-LSE+铰链 floor 0.95）：SLL −22.42 dB（**+12.72 dB** vs 基线，双束平衡 0.62/1.03 dB 前提下）+ 指向 0.111° + 推理批16 0.35 ms/样本——**强信号成立**；v7/v8/v9 为 Pareto 前沿（§14.4(7)） |
| 4 | 多波束泛化 + 1-bit + 迭代基线 | ✅ **闭环（CP6 达成，实验全部完成）** | R2e 单网络 M=2–32：**+3.6~+12.2 dB** vs arg-sum（M=2 −21.88，训内全判据达标）；1-bit STE vs Du 式(6)：**+5.23 dB**；逐任务优化上界对照：距 1.5–3.2 dB、快 10⁴–10⁵×；排列一致性精确为零（§8.0–8.9） |

## 目录结构

```
├─ 执行计划.md            # 总方案 + 逐检查点实测记录（§14，有图有真相）
├─ 交接文档.md            # ★ 服务器训练交接（当前状态、路线决策、操作手册）
├─ README.md              # 本文件
├─ requirements.txt
├─ configs/               # phase1/2/3.yaml（阶段 3 超参与路线开关）
├─ src/metasurf/
│  ├─ physics/            # grid / bp_phase / taylor / array_factor(可微FFT) / metrics(自适应掩膜)
│  ├─ baselines/          # 叠加法基线①②（arg-sum）与③（Du 式(5)/(6) 1-bit）
│  ├─ data/               # 波束采样与任务张量生成
│  ├─ model/              # UNet + ArgSumResidualNet（物理信息残差）
│  ├─ train/              # 可微物理损失（dB/功率域开关、束项、课程开关）+ 训练器
│  └─ eval/               # 绘图助手
├─ scripts/               # 各阶段入口（见下）
└─ results/               # phase0_smoke / phase1_baseline / phase2_preexp / phase3_network
                          # （phase3_network 内含 v1–v6 全部 ckpt、日志与归档 *_vN_*）
```

## 环境与运行

```bash
pip install -r requirements.txt   # torch 需带 CUDA
# 所有脚本均为相对路径，任意工作目录可跑；GPU 显存需求 < 4 GB
```

| 阶段 | 命令 | 产物 |
|---|---|---|
| 0 冒烟 | `python scripts/run_phase0_smoke.py` | results/phase0_smoke/ |
| 1 基线复现 | `python scripts/run_phase1_baseline.py` | results/phase1_baseline/（metrics.csv + 3 图 + 复现核对报告.md） |
| 2 预实验 | `python scripts/run_phase2_preexp.py` | results/phase2_preexp/（E1–E4 + 预实验报告.md） |
| 3 训练 | `python scripts/run_phase3_train.py --runs main,b_pat,a_amp` | results/phase3_network/<run>/（ckpt + history.json） |
| 3 评测 | `python scripts/run_phase3_eval.py` | metrics_test.csv + 3 图 + 方向一验证报告.md |

本地参考环境：`D:\ProgramFiles\Anaconda\envs\antenna_ai\python.exe`（Python 3.10 / torch 2.5.1+cu121 / RTX 3050 8GB；单 run 约 50 min）。服务器环境（2026-09-19 起）：昇腾 910B3（torch 2.7.1+cpu / torch_npu 2.7.1.post4，device 自动选择 cuda→npu→cpu；单 run 150 epoch ≈ 5 h；`array_factor.py` 已含 NPU fftshift/conj 兼容修复）。**服务器训练前必读 [`交接文档.md`](交接文档.md)**——含路线决策、配置开关与验收判据。

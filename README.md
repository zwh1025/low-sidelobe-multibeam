# 单波束到低副瓣多波束编码 —— 方向一：幅度-相位双通道输入的连续相位多波束综合网络

以连续相位调制超表面（64×64，单元幅度恒定，仅相位自由度）为对象，研究**"输入低副瓣单波束幅相先验（BP 补偿相位 + Taylor 幅度窗）→ 输出低副瓣多波束连续相位编码"**的神经网络映射方法。基线为 Du et al., AWPL 2025 的叠加法（SLL ≈ −9.7 dB，无副瓣控制）；目标是在毫秒级推理下逼近理想幅相参照（≈ −24 dB）。

**主文档：[`执行计划.md`](执行计划.md)**（方案、判据、逐阶段实测记录 §14——所有结论以该文档为准）。

## 当前结果速览（详见 执行计划.md §14）

| 阶段 | 内容 | 状态 | 关键数字 |
|---|---|---|---|
| 0 | 工程脚手架 + 物理内核 | ✅ 完成 | Taylor 理想幅相 SLL −25.02 dB（设计值命中）；numpy/torch 一致性 1.2e-7 |
| 1 | 三连检 + 叠加法基线 + Du 2025 复现 | ✅ 完成 | 基线 arg-sum SLL −9.7 dB；Du 对标全部量级一致；两项新发现（基线②≡①、式(6)几何依赖性） |
| 2 | 预实验（物理边界） | ✅ 完成 | 幅度锥削贡献 11.6 dB；解析-理想差距 14.4 dB（情况 A） |
| 3 | 双波束网络最小验证 | ⚠️ 执行完毕、优化未闭环 | SLL 强信号已证（−25~−26 dB，+15~16 dB）但伴随弱束退化；"平衡+低副瓣"联合优化为核心难点，路线待决策（§14.4(5)） |

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

本地参考环境：`D:\ProgramFiles\Anaconda\envs\antenna_ai\python.exe`（Python 3.10 / torch 2.5.1+cu121 / RTX 3050 8GB；单 run 约 50 min）。**服务器训练前必读 [`交接文档.md`](交接文档.md)**——含路线决策、配置开关与验收判据。

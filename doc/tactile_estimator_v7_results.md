# v7 结果：数据扩展、容量复测、双频通道、前瞻接触

> 口径（全文不变）：主口径 = **contact>0.5 且 phase>2.5（支撑相）**，指标 = consumer.vector_mag.r2（**未标定**，单位 mg，|F|/mg）；stairs_interp = 10.5/13.5 cm，stairs_legacy = 16.5/18 cm，stairs_extrap = 16.5/18/19.5 cm，rough = rough_s1/s2。除特别标注外都是 **3 seed 均值**，单 seed 数字一律标注。设计见 doc/tactile_estimator_v7_waves.md（rev4）。

## 1. Wave A：数据（无 gate，用户要求无条件扩量）

| 项 | 数值 |
| --- | --- |
| 网格 | 10 策略 × 12 条件（flat + 4.5/6/9/12/15 cm + 6 种 rough）= 120 格 |
| 采集 | 每格 2 回合 → 摔落/不走审计 → 格子级剔除（保留 fall==0 且成功回合==要求回合） |
| 补量 | 对保留格 3 回合重采，新增 2 格出现 1 摔 → 再剔 |
| 最终 | **98 格 / 294 回合 / 154 581 帧**（v6 A3_qty 22 格 / 105 083 帧） |
| 清洗后 | train_hi195_rough_a2 因标签穿透尖峰被隔离（§3）→ **97 格 / 152 783 帧 / 291 回合** |
| 覆盖 | 每条件 6–10 个策略；每策略 5–12 个条件；rough 覆盖 seed 0/3/4/5/6/7（振幅 2–5 cm） |
| 留出 | 7 个 case 与 v6 逐位相同（sha 比对 max abs diff = 0.0） |

两个网格事实：**15 cm 是高度特定步态策略的难点**（asym9 / crawl21-23 / ab_6 在 15 cm 摔或走不动）；**rough 振幅 ≥4 cm 对平面策略是硬墙**（l0v2 平地就摔、l1all 在 rough 上 700 步只推进 <0.2 m）。

## 2. Wave B：容量阶梯（3 seed）

| 臂 | 参数 | interp | legacy | extrap | rough | 训练墙钟 |
| --- | --- | --- | --- | --- | --- | --- |
| C1 1×128 | 78 492 | 0.3542 | 0.4191 | 0.3904 | 0.2643 | 240–298 s |
| C2 2×128 | 177 564 | 0.5420 | 0.5477 | 0.5205 | 0.3538 | 493–617 s |
| **C3 3×128** | **276 636** | **0.6070** | 0.5123 | 0.5170 | 0.3111 | 933–1200 s |
| C4 6×128 | 573 852 | 0.4830 | 0.4790 | 0.4624 | 0.2688 | 1937–1966 s |

判据（阈值 = max(0.05, MDE 0.1278)，MDE 由逐 case 跨 seed sd 推得）：

| id | 判据 | 数值 | 结论 |
| --- | --- | --- | --- |
| B-P1 | C4−C3 interp ≥ +0.128 | **−0.124** | FAIL |
| B-P2 | C4 contact F1 最大降 ≤ 0.01 | 0.0026 | PASS |
| B-P3 | C4−C3 rough ≥ −0.05 | −0.0423 | PASS |
| B-P4 | \|C1−C3\| 是否 <0.02（容量饱和？） | −0.2528 | RECORDED |

**结论：C3（共享 3×128）仍是容量最优**；6 层明确负收益（train loss 也高，属优化困难而非过拟合）。数据扩到 154 k 帧后最优从 1×128 上移到 3×128，但 6×128 掉头向下。

### 2.1 数据扩展的真实收益（同架构横向比）

| 数据 | interp | legacy | rough |
| --- | --- | --- | --- |
| v6 S3_A3_qty（22 格 / 105 083 帧） | 0.5801 | 0.6416 | 0.3131 |
| v7 C3（98 格 / 154 581 帧） | 0.6070 | **0.5123** | 0.3111 |
| 差 | +0.0269 | **−0.1293** | −0.0020 |

interp 的 +0.027 < MDE 0.088 ⇒ **不算有效提升**；legacy（16.5/18 cm）掉 0.129（>2×MDE）⇒ **策略宽度扩展牺牲了旧高度**。故本报告不声称「数据更多就更好」；数据按用户要求保留（覆盖更宽、可作后续不变性实验底座），但结论必须带这条代价。

## 3. 数据病理与清洗

- Wave C 首轮 B3（`--w-forcenorm 0.5 --bifreq-k 5 --debounce-contact`，3 seed）**发散**：train loss 50 632、val RMSE 46.9 N、contact F1 0.807、第 19 轮早停（best 11）。
- 根因：`train_hi195_rough_a2.npz` 的 force_world 含 **|F| = 66 911 N**（frame 888，Z = −65 725 N）级仿真穿透尖峰；该文件 1798 帧里 **416 帧 force_norm > 5 mg（23%）**。F̄ 头的接触掩码 MSE 分母是帧数（不是标签能量），被尖峰主导 ⇒ 梯度爆炸。全库扫描：98 个训练文件里只有它 >10 帧超标，其余 1–7 帧（真实单帧瞬态）。
- 处理：①文件隔离到 `.tmp/tactile_est_v7/data_quarantine/`；②`EstimatorConfig.label_clip_mg` 默认 **10 mg（1492 N）**，训练侧对 force_world / forcenorm / delta / peak 限幅，**评估与标定用 `dataclasses.replace(label_clip_mg=0)` 读原始标签**；③重训干净基线。
- 干净基线 C3c（3 seed）：interp **0.6115**、legacy **0.5773**、extrap **0.5734**、rough **0.3247**；相对 C3 = +0.0045 / +0.065 / +0.056 / +0.014 ⇒ 容量排序不变。

## 4. Wave C：双频幅值通道（时移冗余）

设计：接触通道保持 50 Hz 瞬时；幅值通道改用**因果 5 帧均值**标签（0.1 s 带宽），训练侧接触标签同时 3 帧 ≥2 帧去抖。

| 臂 | 做法 | head lag0 | head lag+1 | head lag+2 | vecmag lag0 | contact F1 |
| --- | --- | --- | --- | --- | --- | --- |
| C3c_s0 | 基线（无低频通道） | — | — | — | 0.6041 | 0.9816 |
| B3_s0 | **同训**（主干+头一起） | **0.7023** | 0.6679 | — | 0.4662 | **0.8307** |
| B3f_s0 | **冻结主干 + 线性头**（15 ep） | 0.6212 | 0.6429 | 0.5454 | 0.6041 | 0.9816 |
| B3f2_s0 | 冻结主干 + **MLP64 头**（40 ep） | 0.6187 | **0.6801** | 0.6102 | 0.6041 | 0.9816 |

| id | 判据 | 数值 | 结论 |
| --- | --- | --- | --- |
| C-P1 | 中心 R² ≥0.70 | 0.6212（冻结最优） | **FAIL**（同训 0.7023 达标但被 C-P3 否掉） |
| C-P2 | lag+1 ≥0.45 | 0.6801 | **PASS**（基线 0.2268 ⇒ 3×） |
| C-P3 | contact F1 降 ≤0.01 | 冻结 0.0000 / 同训 −0.1509 | **FAIL**（同训臂） |
| C-P4 | label_std_ratio ≥0.70 | 0.7969 | **PASS** |

**结论（PARTIAL）**：时移冗余达成，且必须「冻结主干 + 单独头」——lag+1 由 0.227 → 0.680，contact F1 逐位不变（0.9816）。同训能把中心精度顶到 0.702，但接触通道崩到 0.831，与前几轮「共用主干互相拖累」一致。中心 0.619 < 预注册 0.70 说明：**因果均值的低带宽目标在冻结特征上达不到瞬时精度**；加头容量（linear 0.621 vs MLP64 0.619）与多训 25 轮都没换来收益。

## 5. Wave D：前瞻接触头

冻结主干 + `--lookahead k --w-lookahead 0.5`，20 ep，1 seed：

| 臂 | AUC(k=1) | AUC(k=2) | AUC(k=3) | AUC(k=4) |
| --- | --- | --- | --- | --- |
| LA2_s0 | 0.9038 | **0.8836** | — | — |
| LA4_s0 | 0.9033 | 0.8837 | 0.8682 | 0.8567 |

判据 D-P1 AUC(k=2) ≥ 0.93 ⇒ **NOT CONFIRMED**（0.884）。对照 v6 线性探针（窗口 mean/std 130 维 + ridge）同组留出 AUC(2 帧) = 0.900 —— **加 GRU 前瞻头没有超过线性探针**，说明「未来 40–160 ms 该足是否触地」的信息量在 65 维本体感觉 + 20 帧窗口下大约就到 0.88–0.90。

## 6. Wave E（物理结构：ΣFz + 分配 + CoP）——未开始

设计已在 doc/tactile_estimator_v7_waves.md rev4 固定（D0–D4；主判据 D3：R² ≥ D0−0.02、slope ≥0.85、四足排序 ρ ≥0.85、argmax ≥0.93、contact F1 ≥0.90、ΣFz 中位相对误差 ≤5%、摩擦锥违反 ≤2%）。实现点：force 头拆成「总竖直力 + 四足分配 + 每足切向」，用 ΣFz≈mg 与摩擦锥做结构损失。本轮未执行，成本估 ≈1.5 h（2 臂×2 seed）。

## 7. 复现

```powershell
# 训练（干净数据，C3c 三 seed）
.venv\Scripts\python.exe -X utf8 scripts\train_tactile_estimator.py `--data-dir` .tmp\tactile_est_v7\arms\data `--out` .tmp\tactile_est_v7\ckpts\C3c_s0.pt `--features` 65 `--layers` 3 `--epochs` 40 `--lr-gamma` 0.98 `--early-stop` 8 `--seed` 0
# 双频（冻结主干 + MLP 头）
... --init-from .tmp\tactile_est_v7\ckpts\C3c_s0.pt --freeze-trunk --w-forcenorm 0.5 --bifreq-k 5 --debounce-contact --head-hidden 64 --epochs 40
# 前瞻
... --freeze-trunk --lookahead 2 --w-lookahead 0.5 --epochs 20
# 判据
PYTHONPATH=. .venv\Scripts\python.exe -X utf8 .tmp\tactile_est_v7\tools\bifreq_lag.py --checkpoint <ckpt> --lags 0,1,2
PYTHONPATH=. .venv\Scripts\python.exe -X utf8 .tmp\tactile_est_v7\tools\lookahead_auc.py --checkpoint <ckpt>
PYTHONPATH=. .venv\Scripts\python.exe -X utf8 .tmp\tactile_est_v7\tools\rollup_CD.py
```

证据文件：`.tmp/tactile_est_v7/{rollup_B.json, rollup_C3c.json, rollup_CD.json, bifreq_C3c_B3_s0.json, bifreq_B3f_B3f2.json, lookahead_auc.json, force_outlier_scan.json, l2_evidence.json}`；run 级 trace 在 `.tmp/tactile_est_v7/runs/`。

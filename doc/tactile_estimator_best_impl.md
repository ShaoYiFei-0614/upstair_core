# 触觉估计器「当前最优实现」速查（v7 收官，2026-10-23）

本文只描述**目前最好的一组配置**（v7 Wave B 的 `C3c`，3 个 seed）及其能力边界；
历史版本为什么被淘汰见 §11。所有数字都是同一口径下的实测，不是目标值。

结论先行：**65 维输入（53 维 actor 观测 + 12 维归一化关节力矩）、20 帧窗口 @ 50 Hz、
3 层 × 128 hidden 的共享 GRU 取最后隐状态、4 个线性头（力 12 / 接触 4 / 面积 4 / CoP 8）、
97 个 npz / 291 回合 / 15.28 万帧、40 epoch、训练侧 ±10 mg 标签限幅**。
留出 7 个地形 case 上，支撑相 `|F|/mg` 的 R² 为 0.32–0.68（3 seed 均值：插值 0.611、
外推 0.573、旧高度 0.577、rough 0.325）。

---

## 1. 配置速查

| 项目 | 取值 | 代码位置 |
| --- | --- | --- |
| checkpoint | `.tmp/tactile_est_v7/ckpts/C3c_s{0,1,2}.pt` | — |
| 输入维度 | 65 = actor 观测 53 + 关节力矩 12 | `scripts/collect_tactile_data.py` 采集时拼接 |
| 窗口 | 20 帧（0.4 s @ 50 Hz），stride 训练 2 / 评估 2 | `EstimatorConfig.window` |
| 主干 | `nn.GRU(65, 128, 3, batch_first=True)`，取 `h_n[-1]` | `robotdog/tactile_estimator.py:224` |
| 头 | force 12 / contact 4 / area 4 / cop 8，均为 `Linear(128, d)` | 同上 |
| 参数量 | 276 636 | `ARCH_PARAM_TARGETS["S3"]` |
| 力参考 | `force_ref_n = 149.17 N`（Go2 15.206 kg × 9.81） | `DEFAULT_FORCE_REF_N` |
| 力矩参考 | `4 × (23.7, 23.7, 45.43) N·m`，顺序 FL/FR/RL/RR × hip/thigh/calf | `TORQUE_REF_NM` |
| 优化 | Adam lr 1e-3、wd 1e-5、batch 256、40 epoch、`lr_gamma 0.98`、`early_stop 8`、梯度裁剪 5.0 | `train_estimator` |
| 标签限幅 | `label_clip_mg = 10`（训练侧 ±10 倍体重，评估置 0 用原值） | `make_dataset_windows` / `_eval_cfg` |
| 数据 | 97 个 `train_*.npz`，291 回合，152 783 帧 | `.tmp/tactile_est_v7/arms/data` |
| 样本 | stride 2 ⇒ 66 645 train / 7 173 val 窗口（s0） | ckpt `meta` |
| 推理成本 | 0.196 ms / 窗口（CPU 单线程，实测 batch 2000） | — |

消费端可选头（`w_forcenorm / w_delta / w_peak / w_magrel / w_fbar_rel / physics_weight /
lookahead_contact / w_lookahead / bifreq_k`）在 C3c 里**全部关闭**——v2/v4/v6/v7 的消融
显示这些头没有一个能稳定超过这个基线（见 §11）。

## 2. 输入

- **53 维 actor 观测**：按 policy config 的 `cfg.obs.actor_index` 从 `env._observation()`
  切片，并硬校验 = 53。部分历史策略带 `privileged_foot_contact`（critic 特权块）会把
  观测撑到 57 维，直接当特征等于把**足端接触真值**漏进输入（v5 已修复，现为硬断言）。
- **12 维关节力矩**：`env.data.actuator_force / TORQUE_REF_NM`。归一化尺度来自
  `go2.xml` 的 `ctrlrange`（`actuator_forcerange` 全 0，不可用）；12 维单源定义在
  `TORQUE_REF_NM`，采集与推理必须同源。
- **窗口 20 帧**（0.4 s，50 Hz 控制周期）。`window_starts` 从 `window-1` 起按 stride 取终点，
  并强制包含最后一帧（保证每回合尾部也有样本）。
- **标准化**：逐维 z-score，均值/方差只从训练集算、随权重存档；标准差下限 1e-6。
  除此之外没有手工特征、没有外部传感器。

## 3. 输出（4 个头）

| 头 | 维度 | 物理含义 | 训练目标 | 推理后处理 |
| --- | --- | --- | --- | --- |
| `force` | 4 足 × 3 轴 = 12 | **世界系**足底合力（N） | `force_world / 149.17` | 乘回 `force_ref_n`，reshape `(-1,4,3)` |
| `contact` | 4 | 接触概率 | `(force_n > 1e-6)` 去抖后的 0/1 | sigmoid，阈值 0.5 |
| `area` | 4 | 归一化接触面积 Ā | `area_norm` | 直出；只在接触足上有意义 |
| `cop` | 4 × 2 = 8 | 归一化压力中心 | `cop_norm` | reshape `(-1,4,2)`；只在接触足上有意义 |

`force` 的 `r2_all`（含无接触帧）在 0.83–0.91；`area` R² 0.26–0.57、`cop` R² 0.88–0.93
（逐组见 §8）。

## 4. 内部逻辑（数据流）

```
npz(97) → make_dataset_windows(stride=2)
        → 训练侧限幅 label_clip_mg=10、接触去抖、可选消费端标签
        → (x - mean) / std
        → GRU(3×128) → h_n[-1]  (B,128)
        → 4 个 Linear 头
        → 损失（训练）/ 指标（评估）
```

- 主干是**共享**的（`arch="shared"`）；`arch="indep"`（每任务独立 GRU）实测更差
  （v6 `I2x64`、v5 `I*" 各臂），所以没有采用。
- 序列聚合只取 `h_n[-1]`：没有 attention、没有 mean-pooling、没有双向（因果约束）。
- 推理无状态：每个控制步重算 20 帧窗口。0.2 ms 的成本允许这样做；若上真机需要
  省算力，可改成保留隐状态、每帧只喂新帧，但**必须重新验证**（训练时是滑窗）。

## 5. 训练数据来源与构造

采集脚本 `scripts/collect_tactile_data.py`（固定策略、`deterministic=True` 的确定性
rollout，`env.set_command_now(vx, hold=True)`），标签来自仿真 `mj_contactForce`
（`robotdog/tactile.py` 的 4×5 taxel 阵列扩散到 4 足）。

**10 个策略**（v7 全量，全部成功、0 摔落）：

| 策略 | 文件数 | 说明 |
| --- | --- | --- |
| `ab_15_ctl` / `lr15_7e5` / `lr15_9e5` | 11 / 10 / 12 | 15 cm 基线及其 lr 变体 |
| `asym15_ladder` | 10 | 非对称楼梯 |
| `crawl24` | 11 | 低姿爬行 |
| `hi165` / `hi195` | 12 / 9 | 16.5 / 19.5 cm 高度特化 |
| `l0_v2` / `l1_st_v2` | 5 / 12 | 课程早期检查点 |
| `l1_all` | 5 | 混合地形课程 |

**地形/指令条件**（文件名前缀 + meta 几何键双重编码，留出按此守卫）：

| 类 | 取值 | 文件数 |
| --- | --- | --- |
| flat | 平地行进（tread 0.50） | 8 |
| stair | 4.5 / 6 / 9 / 12 / 15 cm（踏步深 0.30 m） | 9 / 10 / 9 / 8 / 10 |
| rough | 3 / 4 / 5 cm（a/s 两种 seed，tread 0.50） | 7 / 8 / 28 |
| vx | 0.35 / 0.42 / 0.45 m/s | — |

规模：**97 npz / 291 回合 / 152 783 帧**，每帧 18 个数组（`features` 65 维、
`force_world`、`force_foot`、`contact`、`area_norm`、`cop_norm`、`phase`、`episode`、
`force_norm`、`delta_force_norm`、`impact_peak`、`torque` 及 6 个 `priv_*` 诊断列）。
`meta.privileged=True` 只表示**额外存了 `priv_*` 诊断列**，特征本身仍是 53+12。

- 样本数：stride 2 ⇒ **66 645 train / 7 173 val 窗口**（`C3c_s0`）。按**回合**切 10% val
  （`split_seed = seed`，`C3c_s0` 有 29 个 val 回合），同一回合不会同时出现在 train/val。
- 数据清洗：`train_hi195_rough_a2` 的 `force_world` 有 |F| ≈ 66 911 N 的穿透尖峰
  （416/1798 帧 > 5 mg），已**隔离**；训练侧再统一 `label_clip_mg=10` 兜底。
- 留出 7 个 case（**完全不参与训练**）：`test_stair10_5 / 13_5`（插值）、
  `test_stair16_5 / 18 / 19_5`（外推）、`test_rough_s1 / s2`（不同 seed 的 5 cm rough）。

## 6. 训练配方

```powershell
.venv\Scripts\python.exe -X utf8 scripts/train_tactile_estimator.py `
  --data-dir .tmp/tactile_est_v7/arms/data --out .tmp/tactile_est_v7/ckpts/C3c_s0.pt `
  --features 65 --layers 3 --epochs 40 --lr-gamma 0.98 --early-stop 8 --seed 0
```

| 超参 | 值 |
| --- | --- |
| window / hidden / layers | 20 / 128 / 3 |
| batch / epochs | 256 / 40 |
| lr / weight_decay | 1e-3 / 1e-5（每 epoch ×0.98） |
| early_stop | 8（s0 未触发，best_epoch 39；s1 36；s2 32） |
| clip_grad_norm | 5.0 |
| val_frac | 0.1（按回合） |
| 训练成本 | CPU 约 1007–1112 s / seed |

**损失**（`multi_task_loss`，训练/验证同口径）：

```
total = 1.0 · MSE(force/ref)            # 全帧、4×3
      + 0.2 · BCE(contact_logits, y)    # y 为去抖后的接触标签
      + 0.05 · MSE(area, y_a | contact)
      + 0.05 · MSE(cop, y_c | contact)
```

接触/面积/CoP 三项只在接触足上计权（`mask_c = y > 0.5`）。`w_stance = 0`：没有做支撑相
加权（v4 实测支撑相加权只把幅值往均值拉，收益不显著）。best-val 选权重；val_loss 非有限
直接报错（不静默保存最后一轮）。

## 7. 评估与复现口径

```powershell
.venv\Scripts\python.exe -X utf8 scripts/eval_tactile_estimator.py `
  --checkpoint .tmp/tactile_est_v7/ckpts/C3c_s0.pt `
  --data-dir .tmp/tactile_est_v7/arms/holdout `
  --out .tmp/tactile_est_v7/reports/report_C3c_s0.md --stride 2 --latency
```

- **主口径**：`stance = contact > 0.5 & phase > 2.5`（稳态支撑相），指标
  `vector_mag = |F|/mg` 与 `vector_z = F_z/mg` 的 R²、slope、mean_ratio、p95_ratio。
  主口径固定用 `vector_mag`（所有 checkpoint 都有这个量，可跨版本比，避免「换候选集
  就不可比」）。
- **次口径**：接触帧（`contact>0.5`）的力 R²、全帧三维力 `r2_all`、接触 F1/acc、
  `area/cop` R²、支撑相 RMSE（BW 与 N）。
- 训练侧 `label_clip_mg` 在评估时强制置 0（`_eval_cfg`），否则 R² 是跟被截断的标签比。
- 置信区间：按**回合整群** bootstrap 2000 次（不按帧，帧间强相关会低估方差）。
- 接触时延：`--stride 1` 全帧，逐回合统计 `onset_recall / prev_prob_mean / next_prob_mean`。

## 8. 准确度（C3c，3 seed 0/1/2，支撑相口径）

### 8.1 分组（3 seed 均值 ± 标准差）

| 组 | 地形 | `vector_mag` R² | `vector_z` R² | mag slope | mag mean_ratio |
| --- | --- | --- | --- | --- | --- |
| interp | 10.5 / 13.5 cm | **0.6115 ± 0.0231** | 0.7098 | 0.834 | 1.091 |
| legacy | 16.5 / 18 cm | 0.5773 ± 0.0297 | 0.6351 | 0.811 | 1.096 |
| extrap | 19.5 cm | 0.5734 ± 0.0408 | 0.6334 | 0.806 | 1.094 |
| rough | 5 cm（s1/s2） | 0.3247 ± 0.0291 | 0.4852 | 0.576 | 1.115 |
| all7 | 全部 | 0.5132 ± 0.0209 | 0.6129 | 0.749 | 1.099 |

### 8.2 单 case（`C3c_s0`）

| case | mag R² | z R² | slope | mean_ratio | contact F1 | n_stance |
| --- | --- | --- | --- | --- | --- | --- |
| stair 10.5 | +0.547 | +0.646 | 0.784 | 1.078 | 0.979 | 736 |
| stair 13.5 | +0.676 | +0.773 | 0.884 | 1.103 | 0.982 | 698 |
| stair 16.5 | +0.475 | +0.529 | 0.717 | 1.097 | 0.975 | 500 |
| stair 18 | +0.679 | +0.742 | 0.904 | 1.096 | 0.978 | 630 |
| stair 19.5 | +0.566 | +0.630 | 0.798 | 1.090 | 0.952 | 606 |
| rough s1 | +0.336 | +0.563 | 0.691 | 1.121 | 0.993 | 553 |
| rough s2 | +0.313 | +0.407 | 0.462 | 1.109 | 0.995 | 567 |

bootstrap 95% CI（`C3c_s0`，mag）：10.5 [0.487,0.597]、13.5 [0.606,0.728]、
16.5 [0.303,0.662]、18 [0.692,0.751]、19.5 [0.602,0.716]、rough_s1 [0.252,0.441]、
rough_s2 [0.224,0.508]。

### 8.3 其他口径（3 seed 均值）

- **接触帧**（`contact>0.5`）`vector_mag` R²：interp 0.692 / legacy 0.755 / extrap 0.663 /
  rough 0.855 / all7 0.726。
- **全帧三维力** `r2_all`：0.847 / 0.881 / 0.825 / 0.908 / 0.855（分轴 x 0.62–0.77、
  y 0.80–0.86、z 0.84–0.94），`rmse_all` ≈ 0.117 BW。
- **支撑相三维力 RMSE**：interp 0.097 BW（MAE 10.1 N、p95 28.7 N）、legacy 0.102、
  extrap 0.109、rough 0.146；接触帧 RMSE 0.140–0.200 BW。
- **接触检测**：all7 F1 **0.9788**（P 0.9776 / R 0.9800 / acc 0.9838），逐组 0.968–0.994。
- **面积 / CoP**：area R² 0.534 / 0.568 / 0.551 / 0.260（all7 0.463）；
  cop R² 0.883 / 0.908 / 0.897 / 0.928（all7 0.902）。
- **接触时延**（s0）：`onset_recall` 0.944–0.997、`onset_prob_mean` 0.937–0.995、
  `prev_prob_mean` 0.010–0.069、`next_prob_mean` 0.667–0.742。

## 9. 能力边界（使用前必读）

1. **离线开环**：估计器没有接入 actor；`env` 里的 `obs.tactile` 仍是仿真真值。
   闭环收益/风险**未测**。
2. **只有世界系**：数据集没存 `foot_rot`，窗口化时 `force_foot` 被丢弃，所以**足体系力
   无法评估也预测不了**；需要足体系（踝关节控制/足底摩擦）必须重新采数据。
3. **主口径只覆盖 17–28% 的接触帧**：楼梯上大部分接触帧是 LANDING/SWING 抖动，
   稳态支撑帧少。接触帧 R²（0.66–0.85）比主口径更高，但含冲击尖峰，别混着用。
4. **rough 是明显弱项**：5 cm rough 的 mag R² 只有 0.31、slope 0.46（幅值仍被压），
   且 5 cm 对平面策略本身是硬墙（采集阶段就有策略过不去）。
5. **策略漂移是最大风险**：换一个没见过的策略，同一地形上的 F̄ R² 可以从 +0.48 掉到
   −0.48（v5 的 15 cm 五策略扫描）；输入分布几乎不变，变的是「观测→力」的条件映射。
   现有 10 策略覆盖是能用的前提，**换策略必须重测**。
6. **没有前瞻**：`prev_prob_mean` 0.01–0.07（基本为 0），`next_prob_mean` 0.67–0.74
   ⇒ 接触是**当帧识别**，事件上有约 1 帧（20 ms）晚到的风险。v7 Wave D 的专用前瞻头
   AUC 0.884 < 0.93，未达标。
7. **旧高度组（16.5/18 cm）没有随数据扩容提升**：v6 最好臂 legacy 0.642 → v7 `C3`
   0.512；`C3c` 0.577 介于两者。跨波比较受 MDE 限制，不要声称「数据更多就更好」。
8. **幅值轻微过冲**：`mean_ratio` 1.08–1.12（v1 的 0.79 收缩已被修掉），不是标定问题，
   是分布上界被冲击帧拉高。
9. **15 cm 是高度特定策略的难点**，不是估计器难点；估计器只反映策略本身的接触分布。
10. **发布状态**：v7 gate L0–L4 PASS、**L5 SUSPECT**、`release = go_with_caveats`、
    `l5_review_required = true`（pending）。

## 10. 如果要部署

- 每控制步（50 Hz）重算 20 帧窗口：单核 0.2 ms，实时可行；不要用 stride>1 的相邻帧
  复用去省算力（训练是滑窗，复用等于改输入分布）。
- 接触直接用当帧概率（阈值 0.5）；**不要**用「提前量」做预测控制。
- 力只用世界系三轴；需要足底摩擦/踝力矩时另训（要补 `foot_rot` 到数据集）。
- 输入必须严格 53 维 actor 观测 + 12 维力矩、同一 `TORQUE_REF_NM`；策略更换后
  先跑 §7 的评估再决定是否上线。
- 归一化统计随 checkpoint 走，不要在部署侧重算。

## 11. 历史版本为什么被淘汰

| 版本 | 尝试 | 结论 |
| --- | --- | --- |
| v1 | 53 维 + 4 头基线 | 支撑相 F̄ 系统性低估（slope 0.65–0.72），数据覆盖太窄 |
| v2 | 专用 F̄/Δ/峰 头 + 物理项 + 1 帧前瞻 | 单头能把 slope 0.78→0.84，但与共用主干互相拖累，接触 F1 掉 |
| v3 | 加 12 维关节力矩 | ΔR² ≈ +0.003（12 列与 53 列 `max|r|` 0.89–0.96，高度共线） |
| v4 | 支撑相加权 / 两阶段解耦 / 数据多样性 / 限幅逆标定 | 只有「数据多样性」确认有效；全局仿射标定救不了 slope |
| v5 | 新高度覆盖 + 解耦族 + 窗口 40 | 深度/解耦无效；窗口 40 无显著增益 |
| v6 | 容量阶梯 + 类型/数量扩容 | **3×128 最优**（C2 0.542、C3 0.607、C4 0.483，过参数化反而差） |
| v7 | 10 策略全量数据 + 尖峰隔离 + `label_clip_mg=10` | 定稿 `C3c`：interp 0.612 / legacy 0.577 / extrap 0.573 / rough 0.325 |

一句话记忆：**唯一被反复确认的杠杆是「训练数据的策略 × 地形覆盖」，容量停在 3×128，
消费端解码头和标定都是局部修补。**

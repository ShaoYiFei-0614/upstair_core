# 足底触觉估计器（Tactile Estimator）

从**本体感觉**（53 维观测：关节角/角速度、机身姿态/角速度、上一步动作、机身系线速度、三轴指令）估计上一轮加入的足底触觉量。
盲走时代的观测里没有足底信息；现在把「足底有什么」当作**可估计量**，而不是传感器输入。

## 1. 估计什么、为什么

| 量 | 参考目标 | 本文结论 |
| --- | --- | --- |
| 三维接触力（世界系，4 足 × 3 轴） | `tactile.force_vec`（N） | **可估计**：留出地形 RMSE 24–30 N（0.16–0.20 体重）、R² 0.56–0.81；但**支撑相幅值系统性欠估**（斜率 0.78、均值比 0.90），详见 §6 |
| 支撑相 F̄ 幅值（消费端口径） | `force_norm` = Σ|f|/mg | **专用 F̄ 头**可到 stairs R² 0.62、斜率 0.83、均值比 1.04；但共用主干多任务会把 contact 拉低（§6.2） |
| 接触检测（4 足是否触地） | `tactile.force_n > 1e-6`（存在任意接触；`taxel_force_min=1.0 N` 只用于面积 active taxel 计数） | **可估计**：准确率 0.92–0.98、F1 0.90–0.97 |
| 接触面积（归一化） | `tactile.area_norm` | **仅平地（R² 0.73）与规则 15cm 楼梯（0.53）勉强可用**；2–5cm rough 训练分布内只有 0.09–0.30、留出 ≈0（+0.03/+0.04），楼梯留出为负（−0.96/−0.70）⇒ **整体不可用于控制/奖励** |
| 压力中心 CoP | `tactile.cop_norm`（4 足 × 2） | 数值指标好看（R² 0.80–0.93）**但没有信息量**：Go2 球足点接触 ⇒ 目标近似常数（见 §5） |
| 接触相位 | `tactile.phase` | 未作为监督目标（相位机是确定性规则，接触检测准了就能推） |

## 2. 数据

采集脚本 `scripts/collect_tactile_data.py`：跑确定性策略、每步存一行样本（特征 = `env._observation()`，标签 = 触觉真值）。

| split | 地形 | 策略 | 回合 | vx |
| --- | --- | --- | --- | --- |
| train | 平地 | `runs/ab_f_ctl/checkpoints/flat_final.zip` | 16 | 0.45 |
| train | 15cm 楼梯 | `runs/crawl24/checkpoints/stair15_final.zip` | 16 | 0.45 |
| train | rough 2–5cm, seed 0 | flat policy / stair15 policy | 10 + 10 | 0.35 |
| test | **16.5cm 楼梯** | `runs/hi165/.../stair16_5_final.zip` | 8 | 0.42 |
| test | **18cm 楼梯** | `runs/hi18/.../stair18_final.zip` | 8 | 0.42 |
| test | **rough seed 1** | crawl15 policy | 8 | 0.35 |
| test | **rough seed 2** | crawl15 policy | 8 | 0.35 |

共 ~49k 步（约 110 s）。粗糙地形见 `doc/terrain.md` 的 `RoughSpec`：2–5cm 均匀高度场、格子 12cm、13 列（走廊 1.56m 宽）、x 两端 3 格 taper 到 0（**首/末列高度严格 0**，出口与区域外连续）。

> **本页所有数字均由当前代码重跑生成**（2026-10-20 修复 `window_starts` 含末帧、`loss_a` 去多余 /4、rough taper 之后）：
> 采集 `.tmp/_collect3.log` → 训练 `.tmp/_train_final2.log` → 评估 `.tmp/_eval_final4.log`；
> 表内数值直接取自 `.tmp/tactile_estimator/report_final.json`（分轴来自 `report_final.axes.txt`），val 取自 `estimator_final.json`。

## 3. 模型

`robotdog/tactile_estimator.py`：**单层 GRU（hidden 128）+ 四个基础头**（力 12 / 接触 4 / 面积 4 / CoP 8），窗口 20 步、stride 2。
另有四个**可选头**（`forcenorm` / `delta` / `peak` / `lookahead`，见 §6）：由对应损失权重决定是否创建，
权重全 0 时网络结构与历史权重逐键一致。
力的标签按体重 `force_ref_n = 149.17 N` 归一化后做 MSE；接触用 BCE；面积与 CoP 只在接触足上算损失（训练与验证同口径）。

训练：`scripts/train_tactile_estimator.py`（Adam 1e-3、batch 256、40 epochs、按**回合**切 10% 验证集）。
评估：`scripts/eval_tactile_estimator.py`，对每个 npz 输出 md/json + `.axes.txt`（分轴表），含三维力、接触、面积、CoP 与「仅接触足」的力误差。足体系不单独报告（见 §5.5）。

## 4. 结果（最终交付 `estimator_final.pt`：window 20 / hidden 128 / 40 epochs，当前代码重训）

验证集（训练地形内，按回合切 10%）：力 RMSE **18.9 N（0.127 体重）**、MAE **9.1 N**、接触 acc
**0.987** / F1 **0.979**；val_area R² 0.41、val_cop R² 0.95（`.tmp/tactile_estimator/estimator_final.json`）。

> 历史对照（**不是**交付模型）：20.7 N/0.139 BW 是 `rough.py` 改版前 collection-1 的 `estimator.pt`；
> 24.8 N/0.166 BW 是「数据重采后、但训练侧修复前」的上一版。两者均已被本轮重跑覆盖。

**留出地形**（`--stride 2`，数值取 `report_final.json`）：

| 地形 | 力 RMSE | 力 MAE | p95 | R² | 接触 acc | F1 | 面积 R² | CoP R² |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **16.5cm 楼梯** | 29.8 N (0.200 BW) | 18.7 N | 58.7 N | 0.571 | 0.921 | 0.897 | **−0.96** | 0.82 |
| **18cm 楼梯** | 27.9 N (0.187 BW) | 18.1 N | 55.9 N | 0.559 | 0.931 | 0.908 | **−0.70** | 0.80 |
| **rough seed 1** | 25.3 N (0.169 BW) | 11.7 N | 40.0 N | 0.793 | 0.980 | 0.967 | +0.03 | 0.92 |
| **rough seed 2** | 24.0 N (0.161 BW) | 11.8 N | 40.7 N | 0.807 | 0.983 | 0.971 | +0.04 | 0.93 |

训练地形 in-sample：平地 14.9 N（R² 0.919）、15cm 楼梯 18.2 N（R² 0.835）、rough seed0 21.2–21.5 N（R² 0.845–0.853）。

**分轴**（世界系，留出集）：

| 地形 | x RMSE / R² | y RMSE / R² | z RMSE / R² |
| --- | --- | --- | --- |
| 16.5cm | 25.5 N / 0.41 | 16.2 N / 0.48 | 41.9 N / 0.53 |
| 18cm | 24.5 N / 0.26 | 14.9 N / 0.54 | 38.9 N / 0.53 |
| rough s1 | 24.6 N / 0.52 | 14.7 N / 0.74 | 33.1 N / 0.82 |
| rough s2 | 22.5 N / 0.56 | 14.6 N / 0.73 | 31.8 N / 0.83 |

**只算接触足**（更接近「足在支撑时力估得多准」）：16.5cm RMSE 38.1 / MAE 23.1 / p95 74.5 N；18cm 34.1 / 21.7 / 69.0；rough s1 42.4 / 21.6 / 71.1；rough s2 39.6 / 21.9 / 73.5。
误差主要来自**落地冲击尖峰**（单帧 |F| 到 1.8 kN）：p95 远小于 max，仅接触足的 RMSE 比全体大——踩地瞬间的力本质上由「即将发生的接触」决定，本体感觉有一步以上的滞后。

**消融**（window 30 / hidden 192 / 80 epochs；用的是 collection-1 数据 ⇒ 与最终模型数据版本不同，且未在修复后重跑，只能作方向参考）：训练地形更好（平地 12.3 N、15cm 14.6 N），但留出 16.5cm 反而 29.8 N、接触 acc 0.934 ⇒ **更大窗口/网络没有更好泛化**，最终交付用 window 20 / hidden 128（按留出 16.5cm 选型，存在选择偏差，结论仅方向可信）。

## 5. 明确估计不了 / 不该用的量

1. **接触面积**：只在平地（R² 0.73）与规则 15cm 楼梯（0.53）勉强可用；2–5cm rough 训练分布内 0.09–0.30、留出 ≈0（+0.03/+0.04），楼梯留出为负（16.5cm −0.96、18cm −0.70）。原因：面积由「哪些 taxel 被压到」决定，而 Go2 足端是球面点接触，楼梯棱边与踏面接合方式在观测里不可见（同一条关节轨迹可能对应不同贴合形态），且训练集里楼梯种类不足。**结论：面积整体不要拿去做控制/奖励，除非上真机传感器或补更多地形数据。**
2. **CoP 的绝对数值**：R² 高只是因为目标几乎恒定（球足点接触 ⇒ 站立 cop_norm≈(−0.73, ±0.25)、margin 0.275），模型在回归常数。要用的应该是**相对变化**（如 `cop_jump` 或与足体坐标系运动的一致性），不要用绝对 CoP 判断足底压力分布。
3. **水平分量**：**切向（x）最弱**（R² 0.26–0.56）；y 在 rough 上可达 0.73–0.74、楼梯 0.48–0.54；z 0.53–0.83。切向力依赖摩擦与滑动状态，本体感觉里没有直接对应量；如需精确切向力，应在观测里加关节力矩/电流（真机侧有）或足底传感器。
4. **落地冲击峰值**：误差最大处；若用于落地检测，建议用「接触概率 + 一阶差分」而非绝对力阈值。
5. **足体系三维力不单独报告**：模型输出是世界系；数据集 npz **存了** `force_foot` 标签，
   但窗口化（`make_dataset_windows`）时被丢弃，且未保存每步足体旋转矩阵 `foot_rot` ——
   拿世界系预测去比足体系标签会得到一个「看起来也是力误差」的误导数字。要报足体系得在
   采集时存 `foot_rot` 并在评估时按足系旋转预测（评估脚本已显式不报，见其 docstring）。

## 6. 消费端对齐改造与幅度收缩复核（v2，2026-10-22）

§4 的口径有个盲点：它只看**全量窗口的三轴 R²**。下游真正要用的量是「支撑相（`phase>2.5`）接触足上的
F̄ = |F|/mg 幅度」——用这个口径复测交付模型（c0），支撑相回归斜率只有 **0.78**（stairs）、均值比 **0.90**，
即**系统性幅值收缩**（模型把力往均值方向拉）。R² 0.57 看着能用，但「脚踩上去到底多大力」被低估约 10%，
楼梯上更明显。

本轮实验（`experiment_id=2026-10-22-tactile-est-v2`，Trace `.tmp/trace/tactile_est_v2.jsonl`）做两件事：
把消费端量**显式变成训练目标**，然后用预注册阈值检验「幅值收缩主因是训练目标错位」这个假设。

### 6.1 新增的东西（当前代码已生效，默认全关 ⇒ 旧行为逐位不变）

- **标签侧**：`collect_tactile_data.py` 现在落 `force_norm`（世界系 Σ|f| / mg 的传感器真值）、
  `delta_force_norm`（回合内一阶差分）、`impact_peak`（5 帧因果滚动 max）。旧 npz 没有这三列，
  `consumer_label_arrays` 会退化成 proxy（`|F|/mg` 的回合内差分/滚动 max）并在评估报告里标
  `label_source=proxy`——**proxy 与传感器真值差 ~7%（0.9311）**，所以新旧数据不能混着比。
- **头**：`TactileGRU` 增加可选的 `forcenorm_head`(4) / `delta_head`(4) / `peak_head`(4) /
  `lookahead_head`(4×k)。对应权重为 0 时这些头**不创建**，历史 `state_dict` 逐键不变（旧权重可直接 load）。
- **损失**：新增项都是**接触掩码 MSE**（非接触足不参与，避免把「0 力」学成「有力」）：
  `w_forcenorm` / `w_delta` / `w_peak`；`w_magrel`（`relative_magnitude_penalty`，罚 |F| 的**相对**幅值偏差，
  专门打回归到均值）；`physics_weight`（ΣF_z ≈ mg 的 huber，物理先验）；`w_fbar_rel`（F̄ 的相对幅值偏差）；
  `w_lookahead`（接触前瞻 BCE，标签 `contact_next` 取窗口终点后第 k 帧，**不跨回合**）。
- **指标**：`stance_force_metrics`（stance 帧 + 接触掩码上的 r2 / slope / mean_ratio / p95_ratio）、
  `per_foot_constant_baseline`（每足常数预测，用来判断 R² 是不是被常数拟合撑起来的）、
  `contact_latency`（触地起始帧召回 + 前后帧概率）。
- **口径**：`eval_tactile_estimator.py` 主口径**固定 `vector_mag`**（|F|/mg，所有 checkpoint 都有、可跨版本比）；
  `head` 单列。报告里的 `consumer_best` 只是参考且带 `candidate` 字段——**候选数不同的两次实验不要拿它比**。
- **CLI**：`train_tactile_estimator.py` 默认值全部来自 `EstimatorConfig()`（原来硬编码一份、config 里一份，
  改一处另一处不跟），新增 `--w-fbar-rel`；测试用 `python -m unittest tests.test_tactile_estimator`
  （venv 里没有 pytest）。

### 6.2 结果（40 epochs / window 20 / hidden 128 / stride 2；8 个 npz 全部重采为 sensor 标签）

> ⚠️ 本节数字用的是「**只掩 contact**」的旧口径，混进了落地/离地帧；同一个 c0 权重复算成支撑相口径后 stairs R² 从 +0.565/+0.425 掉到 **−0.032/+0.272**（见 §8.1）。**不要把本节与 §8 直接比较。**

训练地形 = flat / 15cm / rough-s0；留出 = **16.5cm / 18cm 楼梯** + rough-s1 / s2。
「stairs 平均」= 16.5cm 与 18cm 的均值。臂：c0 旧损失（对照）、c1 全套消费端项、c2 c1+magrel 1.0、
c3 只加 F̄ 头（`w_forcenorm=0.5`）、c4 c3+magrel 1.0。

| 臂 | stairs 3 轴 R²（Δ vs c0） | stairs slope | stairs 均值比 | **F̄ 头 R² / 斜率 / 均值比** | contact F1 Δ | onset recall Δ |
| --- | --- | --- | --- | --- | --- | --- |
| c0 旧损失（对照） | 0.495 | 0.779 | 0.898 | —（无此头） | — | — |
| c1 全套 | 0.588（**+0.093**） | 0.674 | 0.915 | **0.596 / 0.851 / 0.971** | −0.033 / −0.036 | −0.045 / −0.078 |
| c2 全套+magrel | 0.595（**+0.100**） | 0.647 | 0.921 | 0.655 / 0.811 / 0.912 | −0.049 / −0.041 | −0.041 / −0.052 |
| c3 仅 F̄ 头 | 0.481（−0.014） | 0.777 | 0.876 | **0.625 / 0.834 / 1.041** | −0.011 / −0.045 | −0.044 / −0.083 |
| c4 仅 F̄ 头+magrel | 0.447（−0.048） | 0.713 | 0.879 | 0.680 / 0.757 / 0.942 | −0.055 / −0.071 | −0.040 / −0.088 |

**预注册 3 项阈值**（task_book）：stairs 平均 3 轴 F̄ R² 相对 c0 ≥ +0.10、contact F1 下降 ≤ 0.01、
16.5cm 与 18cm 各自 ≥ c0 ⇒ **NOT CONFIRMED**：c1 差 0.007，c2 恰好压线 0.1000 但 F1 掉 0.04–0.05，
c3/c4 的 3 轴通道没有提升。**「训练目标错位是幅值收缩主因」只对了一半**：

1. **专用 F̄ 头确实把幅值收缩修好了**：c3 的 F̄ 头在 stairs 上 R² 0.625、斜率 0.834、均值比 **1.041**
   （从 0.90 的欠估变成轻微过估），而 3 轴通道几乎不动（ΔR² −0.014）。说明问题不在「GRU 学不动 3 轴力的
   绝对值」，而在「用 3 轴 MSE 训练出来的解码器不面向 F̄ 这个量」。
2. **代价在共用主干**：一旦把 delta/peak/lookahead/magrel 全压上去（c1/c2），contact F1 掉 3–8 点、
   触地起始帧召回掉 4–8 点，3 轴 R² 的增益也被抵消一部分。`magrel` 单独作用在 |F| 上反而让 F̄ 头
   （c2/c4）均值比掉回 0.91–0.94，和直觉相反——它把「相对幅值」压平，救不了 F̄。
3. **物理项 + 前瞻项的收益被接触精度吃掉**：lookahead 头能给出触地前概率，但代价是主 contact 头变差。

⇒ 正确方向不是「在共享主干上堆消费端损失」，而是**消费端量与 3 轴解码器分开**（冻结主干 / 小头分训 /
蒸馏），再加**后验斜率-均值标定**。这条已写进优化方向。

### 6.3 复核后的能力边界

| 量 | 现状 | 能不能用 |
| --- | --- | --- |
| 触地/离地（块状） | F1 0.886–0.971、acc 0.92–0.98；**对损失项很敏感**，任何多任务项都可能拉低 3–8 点 | ✅ 可用；但只用它做相位机/奖励的软门控，阈值 0.5 附近留余量 |
| 3 轴力（世界系） | 全量 R² 0.51–0.81；**支撑相幅值斜率 0.73–0.79、均值比 0.87–1.02** | ⚠️ 只可用于「有力/没力、相对大小」，**不能直接当幅值阈值/奖励量** |
| 支撑相 F̄（专用头） | stairs R² 0.62、斜率 0.78–0.89、均值比 0.93–1.04；rough R² 0.81–0.84 | ⚠️ 目前最接近可用；需标定 + 与 3 轴分开训练 |
| 接触面积 | stairs 留出 R² < 0，rough ≈ +0.03~+0.13 | ❌ 不可用 |
| CoP 绝对值 | R² 高但目标近常数（球足点接触） | ❌ 别用绝对值；只用相对变化 |
| 切向力 | x 轴 R² 0.26–0.56 | ❌ 需要关节力矩/电流或真机足底传感器 |
| 落地冲击峰值 | 误差最大处（单帧 |F| 到 1.8 kN） | ⚠️ 用「接触概率 + 一阶差分」，不要用绝对力阈值 |
| 足体系三维力 | 未报告（窗口化丢 `force_foot`、未存 `foot_rot`） | ❌ 要报得先补 `foot_rot` 落盘 |

## 7. 复现

```powershell
$env:PYTHONPATH="."
.venv\Scripts\python.exe -X utf8 scripts\collect_tactile_data.py --out .tmp\tactile_estimator\data
.venv\Scripts\python.exe -X utf8 scripts\train_tactile_estimator.py --out .tmp\tactile_estimator\estimator_final.pt --window 20 --hidden 128 --epochs 40
.venv\Scripts\python.exe -X utf8 scripts\eval_tactile_estimator.py --checkpoint .tmp\tactile_estimator\estimator_final.pt --out .tmp\tactile_estimator\report_final.md
```

v2（消费端对齐 / 新头 / 新标签；默认全 0 权重时等价旧行为）：

```powershell
$env:PYTHONPATH="."
# 重采（落 force_norm / delta_force_norm / impact_peak）
.venv\Scripts\python.exe -X utf8 scripts\collect_tactile_data.py --out .tmp\tactile_est_v2\data
# c0 对照：旧损失
.venv\Scripts\python.exe -X utf8 scripts\train_tactile_estimator.py --data-dir .tmp\tactile_est_v2\data --out .tmp\tactile_est_v2\c0.pt --epochs 40 --stride-train 2
# c3 只加 F̄ 头（本轮最接近可用的一臂）
.venv\Scripts\python.exe -X utf8 scripts\train_tactile_estimator.py --data-dir .tmp\tactile_est_v2\data --out .tmp\tactile_est_v2\c3.pt --epochs 40 --stride-train 2 --w-forcenorm 0.5
.venv\Scripts\python.exe -X utf8 scripts\eval_tactile_estimator.py --checkpoint .tmp\tactile_est_v2\c3.pt --data-dir .tmp\tactile_est_v2\data --out .tmp\tactile_est_v2\report_c3.md --stride 2 --latency
```

产物（全部在 `.tmp/tactile_estimator/`）：`report_final.md` / `.json`（逐地形指标）、`report_final.axes.txt`（分轴表，由 eval 脚本产出）、`estimator_final.pt` / `.json`（val 指标）、`data/*.npz` 与 `data/collection_summary.json`、`estimator_v2.pt`（消融）。

## 8. 关节力矩观测 + 三条优化方向的实测（v3，2026-10-22）

`experiment_id=2026-10-22-tactile-est-v3`，Trace `.tmp/trace/tactile_est_v3.jsonl`，run 级 Trace
`.tmp/tactile_est_v3/runs/<arm>_s0/trace.jsonl`，汇总 `.tmp/tactile_est_v3/trace_summary.json` /
`trace_report.md`，代码 diff `.tmp/change_tactile_est_v3.diff`。

### 8.1 先修口径：§6.2 的「stairs R² 0.495 / 斜率 0.78」其实混进了落地帧

§6.2 的消费端口径是「**只掩 contact**」，而支撑相帧（`phase>2.5`）只占接触帧的 **22%**：
16.5cm 楼梯 2980 帧里 SWING 6599 / PRELANDING 725 / LANDING 3584 / STANCE 1012 个足-帧
（接触在楼梯上高频抖动，每帧重新触地都会重置 5 帧 LANDING 窗口），rough-s1 更低（17%）。
把同一个 c0 权重按「contact & phase>2.5」重算：

| 口径 | stair16.5 R² / 斜率 / 均值比 | stair18 R² / 斜率 / 均值比 |
| --- | --- | --- |
| 旧口径（只掩 contact） | +0.565 / 0.785 / 0.893 | +0.425 / 0.772 / 0.903 |
| 支撑相（contact & phase>2.5） | **−0.032 / 0.510 / 0.861** | **+0.272 / 0.735 / 0.892** |

⇒ §6.2 那份「R² 0.5 左右」里有很大一块来自近零力的触地/离地帧。**§6.2 与 §8 的数字不可直接比**，
跨版本比较必须同口径。本轮起主口径固定为「contact & phase>2.5」，同时保留 `consumer_contact`
（旧口径）在报告里作对照——评测脚本会同时打印两套。

### 8.2 新增的东西

- **力矩特征**：`collect_tactile_data.py` 每步在 `env.step()` 之后读 `env.data.actuator_force`（与
  `robotdog/reward.py:293` 的 torque 奖励同源），按 `TORQUE_REF_NM = 4×(23.7, 23.7, 45.43)` 归一化后拼成
  12 维，观测从 53 列变 **65 列**。力矩上限取自 `models/unitree_go2/go2.xml` 的 motor ctrlrange
  （Go2 的 `actuator_forcerange` 全 0，不能用来归一化）；`meta` 里写 `n_features=65`、`feature_names`、
  `torque_ref_nm`、`torque_source`。实测 `torque_norm` 的最大绝对值正好 1.0 ⇒ 归一化尺度与 MuJoCo 内部
  力限幅一致。
- **共线性**：12 个力矩列与 53 个本体感觉列的 max|Pearson r| = 0.955（flat）/ 0.940 / 0.913 / 0.894
  （rough-s0-flatpol / rough-s0-stairpol / 15cm），6377 帧取样。
- **CLI**：`train_tactile_estimator.py` 增 `--features`（0=全宽）、`--init-from`、`--freeze-trunk`；
  `eval_tactile_estimator.py` 增 `--features`、`--calibration`（加载时校验 ckpt sha256 / features / stride，
  不一致直接退出）、`--allow-missing-cases`、`REQUIRED_CASES` 齐全性检查。
- **新脚本** `scripts/calibrate_tactile_estimator.py`：只读训练地形的 **val 回合**（`split_episodes`，
  json 记 `val_episode_ids`），拟合域 = 报告域 = `contact & phase>2.5`，留出地形名直接拒跑；
  `fit_affine(method='ls'|'inverse')` 两种方向都算（`ls`=最小二乘 `t≈a·y`，`inverse`=让标定后斜率恒 1）。
- **库层**：`torque_features` / `slice_features` / `init_model_from_checkpoint`（只允许缺 `*_head.bias/weight`，
  否则报错）/ `freeze_except_forcenorm` / `apply_affine` / `fit_affine` / `stance_force_metrics(phase=...)`；
  测试 `python -m unittest tests.test_tactile_estimator` → **61 tests OK**。

### 8.3 四臂结果（65 列数据，40 epochs / window 20 / hidden 128 / stride 2，单 seed 0）

臂：`c0` 53 列旧四项损失（对照）、`c0T` 65 列旧四项损失（力矩对照）、`t2` 65 列 + 同期训 F̄ 头
（`w_forcenorm=0.5, w_fbar_rel=0.5`）、`t3` = `c0T` 初始化 + **冻结主干**只训 F̄ 头。

| 臂 | stair16.5 支撑相 vm R²/斜率/均值比 | stair18 同上 | F̄ 头 R²/斜率/均值比（16.5 / 18） | contact F1（16.5 / 18） | onset recall |
| --- | --- | --- | --- | --- | --- |
| c0（53 列） | −0.032 / 0.510 / 0.861 | +0.272 / 0.735 / 0.892 | — | 0.897 / 0.908 | 0.864 / 0.882 |
| c0T（65 列） | −0.144 / 0.433 / 0.825 | −0.051 / 0.622 / 0.822 | — | 0.911 / 0.873 | 0.856 / 0.835 |
| t2（同期训头） | −0.095 / 0.727 / 0.869 | +0.335 / 0.933 / 0.935 | **+0.311 / 0.759 / 1.064**；**+0.482 / 0.830 / 1.106** | 0.874 / 0.855 | 0.767 / 0.742 |
| t3（冻结主干） | −0.144 / 0.433 / 0.825 | −0.051 / 0.622 / 0.822 | −0.071 / 0.396 / 0.830；+0.129 / 0.559 / 0.891 | 0.911 / 0.873（**Δ=0**） | 0.856 / 0.835（Δ=0） |

rough 上 t2 的 F̄ 头 R² +0.629 / +0.514（vm 只有 +0.028 / −0.059），flat 上 0.905。
世界系合力全量 R²：c0 0.565（stairs 平均）→ c0T 0.598；t2 0.487。contact F1 相对 c0T：
t2 −0.037 / −0.019、onset −0.089 / −0.093；t3 逐位不变（`trunk_sha256` 与 c0T 相同、
`trunk_diff_keys=[]`、可训练参数 516 / 79008）。

### 8.4 预注册判据

| 判据 | 阈值 | 实测 | 结论 |
| --- | --- | --- | --- |
| P1 力矩特征：stairs 世界系 x 轴力 R² 平均 | ≥ +0.05 | **+0.0027** | FAIL |
| P1 力矩特征：stairs 支撑相 vm R² 平均 | ≥ +0.03 | **−0.2178** | FAIL |
| P2 解耦头：冻结后 F̄ 头斜率 | ∈[0.80,1.00] | 0.396 / 0.559 | FAIL |
| P2 解耦头：冻结后 F̄ 头均值比 | ∈[0.92,1.10] | 0.830 / 0.891 | FAIL |
| P2 解耦头：contact F1 降幅 | ≤0.01 | **0.000 / 0.000** | PASS |
| P3 标定（ls）：vm 斜率 | ∈[0.85,1.05] | 0.372 / 0.535 | FAIL |
| P3 标定（ls）：vm 均值比 | ∈[0.95,1.10] | 0.927 / 0.915 | FAIL（但比标定前 0.825 / 0.822 好） |
| P3 标定（ls）：rough |ΔR²| | ≤0.03 | +0.038 / +0.039 | FAIL |
| P3 标定（inverse）：斜率 | ∈[0.85,1.05] | 0.565 / 0.812 | FAIL |
| P3 标定（inverse）：R² 变化 | 不塌 | −0.401 / −0.345 | FAIL |
| L5 方向性（t2 头 vs c0T vm） | 方向一致 | R² 0.311>−0.144、0.482>−0.051；均值比 1.064 vs 0.825、1.106 vs 0.822 | PASS（**跨通道基线，§9.3 已判定这种比法无效**） |

总判：**P1 NOT CONFIRMED**（力矩既没补上切向力、也没补上支撑相幅值，且与本体感觉高度共线）、
**P2 PARTIAL**（冻结能把接触逐位保住，但头质量掉到斜率 0.40–0.56；同期训头质量好却拖 contact 3.7/1.9 点、
onset 8.9/9.3 点）、**P3 NOT CONFIRMED**（ls 能修均值比与 R²，修不了斜率；inverse 斜率上去了但 R² 崩）、
**L5 PASS（跨通道口径，作废口径）**：这一版拿 F̄ 头（法向力/mg）去比 c0T 的 vector_mag（三轴合力/mg），
两者不是同一个物理量，比出来的「方向一致」不构成证据——§9.3 的 v4 判据已改为同通道基线（w2B.head vs w2A.head），
结论随之改为 **SUSPECT**。本节保留原样只为记录当时的判定过程，不要再引用这一行的 PASS。

### 8.5 能力边界（支撑相口径重写；替代 §6.3 的对应行）

| 量 | 现状（留出数据实测） | 能不能用 |
| --- | --- | --- |
| 触地/离地 | F1 0.855–0.987；多任务会掉 2–4 点，加 F̄ 头掉 1.9–3.7 点 | ✅ 可用（阈值留余量） |
| 世界系 3 轴力 | 全量 R² 0.46–0.95（flat 最高、18cm 最低） | ⚠️ 只可判「有力/没力、相对大小」 |
| 支撑相 F̄（3 轴通道解码） | stairs R² −0.032 / +0.272、斜率 0.51–0.74、均值比 0.86–0.89 | ❌ 支撑相幅值不可用 |
| 支撑相 F̄（专用头，t2） | stairs R² +0.311 / +0.482、斜率 0.76–0.83、均值比 1.06 / 1.11；rough 0.51–0.63、flat 0.91 | ⚠️ 最接近可用，但仍偏乐观/不稳 |
| 接触面积 | stairs R² < 0（c0T −0.368 / −0.135）、rough ≈0 | ❌ 不可用 |
| CoP 绝对值 | 目标近常数（球足点接触） | ❌ 只用相对变化 |
| 切向力（世界系 x） | R² 0.08–0.87，stairs 0.24–0.44；**加力矩没有改善** | ❌ 仍需真机足底传感器 |
| 落地冲击峰值 | 单帧 |F| 可到 1.8 kN，误差最大处 | ⚠️ 用接触概率+差分 |
| 足体系三维力 | 窗口化丢 `force_foot`、未存 `foot_rot` | ❌ 需先补 `foot_rot` 落盘 |

### 8.6 结论：这轮否掉了什么、下一步该做什么

1. **关节力矩不是瓶颈**：与本体感觉 max|r| 0.89–0.96，预注册两项都失败；唯一非预注册正收益是世界系合力
   全量 R² +0.033。要再试只应作 critic/辅助输入，不要当主通道。
2. **口径是结论的一部分**：同一权重从「只掩 contact」换到「支撑相」，stairs R² 从 +0.57/+0.43 掉到
   −0.03/+0.27。此前基于 §6.2 的乐观判断（「F̄ 头 R² 0.62」）在支撑相口径下要打折看。
3. **共用主干的解耦没走通**：同训头好但拖接触、冻结保接触但头废 ⇒ 下一步不是调权重，而是
   **两阶段**（先冻结训头 → 再低 lr 联合微调、或在冻结主干上加一个小 MLP 头提高容量），并给
   `stance` 帧单独加权（现在损失按帧平均，22% 的支撑相被 78% 的 LANDING/SWING 帧淹没）。
4. **仿射标定只能救均值比和 R²**：ls 把均值比 0.825→0.927、R² −0.144→+0.104，斜率仍 0.37–0.64；
   inverse 斜率达标但 R² 崩 ⇒ 幅值收缩不是单一乘性偏差，先在数据/损失层修分布，再谈标定。
5. **数据分布仍是最大瓶颈**：训练地形只有 flat / 15cm / rough-s0，留出 16.5/18cm 与 rough-s1/s2；
   rough-s1/s2 上 c0T 的 vm R² 也只有 +0.168 / +0.032 ⇒ 域外泛化本身就差，换头/换损失救不回来。
6. **能上机器人的只有接触通道**；幅值要么现场标定，要么只用相对量（比值/变化），不要用绝对阈值。

### 8.7 复现（v3）

```powershell
$env:PYTHONPATH="."
# 重采（65 列 = 53 本体感觉 + 12 力矩）
.venv\Scripts\python.exe -X utf8 scripts\collect_tactile_data.py --out .tmp\tactile_est_v3\data
# 4 臂
.venv\Scripts\python.exe -X utf8 scripts\train_tactile_estimator.py --data-dir .tmp\tactile_est_v3\data --out .tmp\tactile_est_v3\c0.pt  --features 53 --epochs 40 --stride-train 2
.venv\Scripts\python.exe -X utf8 scripts\train_tactile_estimator.py --data-dir .tmp\tactile_est_v3\data --out .tmp\tactile_est_v3\c0T.pt --features 65 --epochs 40 --stride-train 2
.venv\Scripts\python.exe -X utf8 scripts\train_tactile_estimator.py --data-dir .tmp\tactile_est_v3\data --out .tmp\tactile_est_v3\t2.pt  --features 65 --epochs 40 --stride-train 2 --w-forcenorm 0.5 --w-fbar-rel 0.5
.venv\Scripts\python.exe -X utf8 scripts\train_tactile_estimator.py --data-dir .tmp\tactile_est_v3\data --out .tmp\tactile_est_v3\t3.pt  --features 65 --epochs 40 --stride-train 2 --w-forcenorm 0.5 --w-fbar-rel 0.5 --init-from .tmp\tactile_est_v3\c0T.pt --freeze-trunk
# 标定（只用训练地形 val 回合）→ 评估（主口径支撑相 + 旧口径对照）
.venv\Scripts\python.exe -X utf8 scripts\calibrate_tactile_estimator.py --checkpoint .tmp\tactile_est_v3\t3.pt --data-dir .tmp\tactile_est_v3\data --out .tmp\tactile_est_v3\calibration_t3.json --stride 2
.venv\Scripts\python.exe -X utf8 scripts\eval_tactile_estimator.py --checkpoint .tmp\tactile_est_v3\t3.pt --data-dir .tmp\tactile_est_v3\data --out .tmp\tactile_est_v3\report_t3.md --stride 2 --latency --calibration .tmp\tactile_est_v3\calibration_t3.json
```

注：`--features 53` 的权重去评 65 列数据会显式打印 `[info] 按前 N 列前缀评估`；反之（65 列权重配 `--features 53`）会直接报错，避免把带宽度的臂误评成纯观测臂。

## 9. 支撑相加权 / 两阶段解耦 / 数据多样性 / 限幅标定（v4，2026-10-22）

`experiment_id=2026-10-22-tactile-est-v4`，Trace `.tmp/trace/tactile_est_v4.jsonl`（agent 级）
与 `.tmp/tactile_est_v4/runs/<arm>_s0/trace.jsonl`（run 级，5 臂各一份），汇总
`.tmp/tactile_est_v4/trace_summary.json` / `trace_report.md`，判据数字 `.tmp/tactile_est_v4/rollup.json`，
代码 diff `.tmp/change_tactile_est_v4.diff`。

§8.6 给了四条下一步；本轮把它们做成可判定的四个臂（同一 65 列数据、同 seed 0、同 window 20 / hidden 128）：

| 臂 | 变量 | 训练数据 | 损失 / 结构 |
| --- | --- | --- | --- |
| `c0T` | 基线（v3 权重，只重评不重训） | v3 4 个 train case | 旧四项损失 |
| `w1` | ①支撑相加权 | v3 4 个 case | `--w-stance 3.0`（force 项权重 `1+3·stance`） |
| `w2A` | ②两阶段 A：冻结主干只训头（组合臂：两阶段 + 支撑相加权） | v3 4 个 case | `--init-from c0T --freeze-trunk --head-hidden 64 --w-forcenorm 0.5 --w-fbar-rel 0.5 --w-stance 3.0` |
| `w2B` | ②两阶段 B：解冻低 lr 联合微调（同上损失权重） | v3 4 个 case | `--init-from w2A --lr 1e-4 --epochs 10 --head-hidden 64 --w-forcenorm 0.5 --w-fbar-rel 0.5 --w-stance 3.0` |
| `w3` | ③训练数据/策略多样性 | **v4 10 个 train case** | 旧四项损失（只换数据） |

多样性 = v3 的 4 个 case + 6 个新 case（`EXTRA_CASES`，全 train）：rough seed 3 用 flat 策略、rough seed 3 用 stair15
策略、rough seed 4 用 stair15 策略、15cm × stair16.5 策略、15cm × stair18 策略、flat × stair16.5 策略
（`--include-extra`，共 14 个 npz）。训练窗口 29107 / val 3203（v3 是 13924 / 1552）。

### 9.1 新增的东西

- **支撑相加权**：`EstimatorConfig.w_stance`；`multi_task_loss` 里 force 项按 `1 + w_stance·(phase>2.5)` 乘权重，
  F̄ / ΔF̄ / 峰值三项用 `frame_w = contact·(1 + w_stance·stance)`。
  ⚠️ **计权不对称（必须知道）**：force 项是**全帧**计权（非接触帧权重 1、支撑相 4），而 F̄ 三项只对接触帧计权。
  实测支撑相&接触帧只占全部帧的 6.2%（3828 / 61904）、占接触帧的 19.3%，`frame_w` 全帧均值 0.506
  ⇒ 12 维力回归仍被 71% 的非接触帧稀释，S1 只拿到斜率 +0.02~+0.04 就是这条的直接后果。
- **头容量**：`EstimatorConfig.head_hidden`；`head_hidden>0 且 w_forcenorm>0` 时 F̄ 头变
  `Linear(h→64) → ReLU → Linear(64→4)`（键 `forcenorm_head.0.*` / `.2.*`，可训练 8516 / 87008 参数）。
  `init_model_from_checkpoint` 只容忍 `*_head.*` 缺失（其它 missing / unexpected 一律报错）。
- **限幅逆标定**：`fit_affine(method='ls'|'inverse', a_min, a_max)`，`inverse` 方向保留
  `mean_y/mean_t/var_y/var_t/cov/a_raw/clipped`，标定保留均值（`b = mean_t − a·mean_y`）；
  `calibrate` 增 `--a-min/--a-max`（只作用 inverse）。
- **斜率限幅扫描**：`eval` 读标定 json 后输出 `clamp_sweep`（`CLAMP_GRID` 并集
  `1.0/1.2/1.3/1.4/1.6/1.8/2.0/2.5`，判定只用预注册 `1.0/1.3/1.6/2.0`），报告新增「斜率限幅扫描」表。
- **数据多样性**：`collect_tactile_data.py` 增 `EXTRA_CASES`（6 条）与 `--include-extra`、`--only`；
  留出守卫按前缀（`stair16_5`/`stair18`/`rough_s1`/`rough_s2`）拒收训练 case。
- 测试 `.venv/Scripts/python.exe -m unittest tests.test_tactile_estimator tests.test_tactile_estimator_v4`
  → **94 tests OK**（v4 新增 21 条：frame_w 闭式/weight=0 等价/MLP 头键名/n_trainable/限幅逆标定闭式/
  init_from 三态/CLI 往返/CLAMP_GRID 精确网格/姿态掩码严格 >/扭矩布局/训练路径确实喂 phase）。
- 标定拟合源按臂对齐训练数据（评审 B4）：c0T/w1/w2A/w2B 只在 v3 的 4 个 case split 上拟合
  （val 1552 窗口 / 336 帧），w3 用 v4 的 10 个 case split（val 3203 窗口 / 874 帧）；
  拟合域 = 评估域 = `stance(phase>2.5) & contact`，calibration json 里 `data_dir`/`val_episode_ids` 可核。
  ⚠️ 由此**标定后单元格不可跨臂直接比**：c0T/w1/w2A/w2B 的系数来自 4 个 case 的 336 帧，w3 来自 10 个 case 的 874 帧；
  跨臂比较只用未标定的主口径数字（S1–S3 都是）。

### 9.2 结果（支撑相口径，contact & phase>2.5，单 seed 0；stairs = 16.5cm / 18cm）

| 臂 | 16.5cm vm R² / 斜率 / 均值比 | 18cm vm R² / 斜率 / 均值比 | F̄ 头 R² / 斜率 / 均值比（16.5 / 18） | contact F1 | onset recall |
| --- | --- | --- | --- | --- | --- |
| c0T | −0.144 / 0.433 / 0.825 | −0.051 / 0.622 / 0.822 | — | 0.911 / 0.873 | 0.856 / 0.835 |
| w1 | −0.011 / 0.451 / 0.876 | +0.155 / 0.657 / 0.931 | — | 0.914 / 0.882 | 0.858 / 0.841 |
| w2A（冻结） | 与 c0T **逐位相同** | 同左 | 0.251 / 0.589 / 0.984；0.428 / 0.774 / 1.038 | 0.911 / 0.873（Δ=0） | Δ=0 |
| w2B（两阶段） | +0.013 / 0.510 / 0.908 | +0.120 / 0.689 / 0.911 | **0.220 / 0.655 / 1.045**；**0.293 / 0.819 / 1.109** | **0.919 / 0.880** | 0.860 / 0.830 |
| **w3（数据多样性）** | **+0.445 / 0.643 / 1.018** | **+0.567 / 0.885 / 1.048** | — | **0.963 / 0.970** | **0.944 / 0.961** |

rough 上：c0T vm 0.168 / 0.032、w1 0.177 / −0.007、w3 **0.432 / 0.188**（均值 0.310 vs c0T 0.100）、
w2B 的头 0.613 / 0.405。世界系合力全量 R²（stairs 平均）：c0T 0.598 → w3 0.782。

冻结自证（`.tmp/tactile_est_v4/freeze/w2A_s0_freeze.json`）：`trunk_sha256` 前后相同
（`d8a1bab7…6b4f1`）、`trunk_diff_keys=[]`、missing 恰 4 个 `forcenorm_head.*`、unexpected=[]、
可训练 8516 / 87008、头形状 `[64,128]/[64]/[4,64]/[4]`。两阶段 B：`missing=[]`、主干与头都相对 A 有更新、
`lr=1e-4`。L3 冒烟另做了确定性双跑（3 epoch 两次 state_dict 逐位相同）与冻结自证。

### 9.3 预注册判据

| 判据 | 阈值（rev2 钉死） | 实测 | 结论 |
| --- | --- | --- | --- |
| S1 支撑相加权：stairs vm R² 均值 | ≥ max(c0T 均值 +0.10, **+0.0025**) | **+0.072**（c0T 均值 −0.0977 ⇒ 公式值 +0.00233，取更严的安全下限 **+0.0025**） | PASS |
| S1：两条楼梯 vm 斜率 | 各自 ≥0.75 | 0.451 / 0.657 | FAIL |
| S1：contact F1 降幅 | ≤0.01 | F1 反而 +0.003 / +0.009 | PASS |
| S2 解耦：F̄ 头 R² | 各自 ≥0.50 | 0.220 / 0.293 | FAIL |
| S2：F̄ 头斜率 | 各自 ∈[0.80,1.05] | 0.655 / 0.819 | FAIL（18cm 达标） |
| S2：F̄ 头均值比 | 各自 ∈[0.95,1.10] | 1.045 / 1.109 | FAIL（仅 18cm 越界 0.009） |
| S2：contact F1 / onset 降幅 | ≤0.01 / ≥−0.03 | **F1 反而 +0.008 / +0.007**，onset +0.004 / −0.006 | PASS |
| S3 数据多样性：rough vm R² 均值 | ≥ c0T +0.15 | **0.310**（c0T 0.100） | PASS |
| S3：stairs vm R² 降幅 | ≤0.05 | 反而 **+0.589 / +0.618** | PASS |
| S3：contact F1 降幅（stairs + rough） | ≤0.01 | 反而 +0.052 / +0.096 / +0.023 / +0.021 | PASS |
| S4 限幅逆标定：存在 a∈{1.0,1.3,1.6,2.0} | 两楼梯斜率 ≥0.75 且 rough \|ΔR²\| ≤0.05 | a=1.0 斜率 0.643 / 0.885（16.5cm 不达标）；a=1.3 斜率 0.835 / 1.151 但 rough ΔR² −0.255 / −0.193；a=1.6 斜率 1.028 / 1.416、rough ΔR² −0.676 / −0.496；a=2.0 斜率 1.285 / 1.770、rough ΔR² −1.519 / −1.082（全部 FAIL） | FAIL |
| L5 方向性（预注册：w2B 头 vs c0T vm，**跨通道**） | 头 R² 更高且 \|均值比−1\| 更小 | 0.220>−0.144、0.293>−0.051；1.045/1.109 vs 0.825/0.822 | PASS（但基线不是同一个量） |
| L5 方向性（**同通道**：w2B 头 vs w2A 头） | 头 R² 更高且 \|均值比−1\| 更小 | R² 0.251→0.220、0.428→0.293（**2/2 反向**）；\|均值比−1\| 0.016→0.045、0.038→0.109（**2/2 反向**）；斜率 0.589→0.655、0.774→0.819（2/2 更接近 1） | **SUSPECT** |

总判：**S1 PARTIAL**（R² 均值跨过阈值、接触零代价，但斜率没推上去，被全帧计权稀释）、
**S2 PARTIAL**（接触完全保住甚至更好，头质量仍不够：R² 0.22–0.29、斜率 0.66 / 0.82）、
**S3 CONFIRMED**（只换训练数据就把两条留出楼梯从不可用推到可用，且接触 / 时延同步变好）、
**S4 NOT CONFIRMED**（一个全局 a 无法同时满足两条楼梯的斜率与 rough 的 R² 保持）、
**L5 SUSPECT**（跨通道的预注册对照满足方向，但换成唯一同通道基线 w2A 头后，两阶段 B 在 R² 与均值比上
2/2 反向 —— 只有斜率更接近 1。跨通道比较把「有 F̄ 头」这件事当成了改进，是假绿）。

### 9.4 能力边界（v4 后，替代 §8.5 对应行）

| 量 | 现状（留出数据实测） | 能不能用 |
| --- | --- | --- |
| 触地 / 离地 | F1 0.86–0.99；w3 0.963 / 0.970，onset 0.944 / 0.961 | ✅ 可用 |
| 世界系 3 轴力 | 全量 R² stairs 0.78（w3）/ 0.60（c0T） | ⚠️ 相对量可用，绝对值仍需标定 |
| 支撑相 F̄（3 轴通道） | w3 stairs R² +0.445 / +0.567、斜率 0.643 / 0.885、均值比 1.018 / 1.048 | ⚠️ **第一次到可用边缘**（16.5cm 斜率仍偏软） |
| 支撑相 F̄（专用头） | w2B R² 0.220 / 0.293、斜率 0.655 / 0.819、均值比 1.045 / 1.109 | ❌ 均值方向对但 R² 不够；contact 零代价说明还能再加容量 |
| 接触面积 / CoP 绝对值 | 不变（stairs R² <0 / 目标近常数） | ❌ |
| 切向力（世界系 x） | 不变 | ❌ 仍需真机足底传感器 |
| 限幅逆标定 | a=1.3 能修 16.5cm 斜率（0.835）但 rough R² 掉 0.19–0.26、18cm 过冲到 1.151 | ❌ 全局单参数不可用；只能按地形 / 相位分组或只用 ls 修均值 |
| 足体系三维力 | 仍缺 `foot_rot` | ❌ |

### 9.5 结论

1. **数据 / 策略覆盖是这一轮唯一被确认的杠杆**：只换训练集（不加损失、不加容量），stairs vm R²
   −0.14 / −0.05 → **+0.45 / +0.57**、均值比 0.82 → 1.02 / 1.05、F1 +0.05 / +0.10、onset +0.09 / +0.13。
   ⚠️ 混杂必须写清：新增 case 里有 15cm ×（stair16.5 策略 / stair18 策略），而留出集正是这两条策略在
   16.5 / 18cm 上的记录 ⇒ 增益里既有「地形高度外推」也有「策略覆盖」，本轮无法把两者分开（要分开就得再加
   同策略不同高度的 case）。
2. **支撑相加权方向对、量级不够**：R² 均值较 c0T +0.17（绝对 +0.072）、接触零代价，但斜率只动 +0.02~+0.04。
   force 项是全帧计权（非接触帧 1、支撑相 4），支撑相&接触帧只占 6.2% 全帧 ⇒ 想让它起作用必须把 force 项
   也换成 `frame_w`（会改变与 v3 的可比性，需要单独一臂），或改成「每足支撑相样本等权」的采样。
3. **两阶段解耦：A 段成功（不伤接触），B 段反而把头弄差**：A 段冻结主干只训 64 宽 MLP 头，
   接触逐位不变（ΔF1=0）而头已可用（R² 0.251 / 0.428、均值比 0.984 / 1.038）；B 段低 lr 解冻 10 epoch 后
   F1 再涨 0.008 / 0.007、onset 不降，但**同通道对比（w2B.head vs w2A.head）头本身变差**：
   R² 0.251→0.220、0.428→0.293，`|均值比−1|` 0.016→0.045、0.038→0.109；只有斜率更靠 1（0.589→0.655、
   0.774→0.819）。⇒ 「斜率拉近但散布变大」是典型的小 lr 短程解冻过拟合，**不是**「两段式有效」的证据。
   下一步该试：stage B 只训头不解冻主干、或更多 epoch + 更大 head + 支撑相样本过采样（而不是继续解冻）。
4. **全局仿射标定到头了**：`inverse` 把 w3 的 16.5cm 斜率从 0.643 推到 0.839，但 18cm 过冲到 1.156、
   rough R² 掉 0.19–0.26。想用标定只能**分组**（按地形 / 相位 / 接触高度），且必须先有能区分地形的输入
   （obs 里没有地形高度，只能靠触觉 / 力矩间接推断 ⇒ 与 S3 的结论互相支撑：先把覆盖补上）。
5. **可交付的当前最优**：`w3` 权重（10 case 训练、旧四项损失、无标定）——stairs 支撑相 F̄
   R² 0.45 / 0.57、均值比 1.02 / 1.05、F1 0.96 / 0.97、onset 0.94 / 0.96，是「接触可用 + 幅值可作相对量」的
   第一条配置；绝对幅值仍建议现场按地形标定，不要用固定阈值。

### 9.6 复现（v4）

```powershell
$env:PYTHONPATH="."
# 14 个 case（8 基础 + 6 extra，全 train）
.venv\Scripts\python.exe -X utf8 scripts\collect_tactile_data.py --out .tmp\tactile_est_v4\data --include-extra
# w1 支撑相加权
.venv\Scripts\python.exe -X utf8 scripts\train_tactile_estimator.py --data-dir .tmp\tactile_est_v3\data --out .tmp\tactile_est_v4\ckpts\w1_s0.pt --features 65 --epochs 40 --stride-train 2 --w-stance 3.0
# w2A 冻结 + 64 宽 MLP 头，w2B 低 lr 解冻
.venv\Scripts\python.exe -X utf8 scripts\train_tactile_estimator.py --data-dir .tmp\tactile_est_v3\data --out .tmp\tactile_est_v4\ckpts\w2A_s0.pt --features 65 --epochs 40 --stride-train 2 --w-stance 3.0 --w-forcenorm 0.5 --w-fbar-rel 0.5 --head-hidden 64 --init-from .tmp\tactile_est_v3\c0T.pt --freeze-trunk
.venv\Scripts\python.exe -X utf8 scripts\train_tactile_estimator.py --data-dir .tmp\tactile_est_v3\data --out .tmp\tactile_est_v4\ckpts\w2B_s0.pt --features 65 --epochs 10 --stride-train 2 --w-stance 3.0 --w-forcenorm 0.5 --w-fbar-rel 0.5 --head-hidden 64 --init-from .tmp\tactile_est_v4\ckpts\w2A_s0.pt --lr 1e-4
# w3 只换数据
.venv\Scripts\python.exe -X utf8 scripts\train_tactile_estimator.py --data-dir .tmp\tactile_est_v4\data --out .tmp\tactile_est_v4\ckpts\w3_s0.pt --features 65 --epochs 40 --stride-train 2
# 每臂：标定（用 val 回合）→ 评估（14 个 npz，支撑相口径 + 限幅扫描）
.venv\Scripts\python.exe -X utf8 scripts\calibrate_tactile_estimator.py --checkpoint .tmp\tactile_est_v4\ckpts\w3_s0.pt --data-dir .tmp\tactile_est_v4\data --out .tmp\tactile_est_v4\calibration_w3.json --stride 2 --a-min 1.0 --a-max 2.5
.venv\Scripts\python.exe -X utf8 scripts\eval_tactile_estimator.py --checkpoint .tmp\tactile_est_v4\ckpts\w3_s0.pt --data-dir .tmp\tactile_est_v4\data --out .tmp\tactile_est_v4\report_w3.md --calibration .tmp\tactile_est_v4\calibration_w3.json --stride 2 --latency
```

判据复算：`.tmp/tactile_est_v4/tools/rollup.py`（读 `report_<arm>.json` → `rollup.json`）。

---

## 10. v5 Wave A：按高度类型补训练覆盖（2026-10-22，已执行）

> 本章数字口径与 §9 不同：v5 一律 **支撑相（contact & phase>2.5）+ 未标定 vector_mag**，且判据阈值来自本轮 D3 噪声底（MDE=0.179；Wave D 用 MDE_3seed=0.0797），**不可与 §8/§9 的旧口径数字直接比较**。

### 10.1 本轮目标与预注册协议

用户批准的五步顺序：诊断波 → 数据（Wave A）→ 解耦（Wave B）→ 深度/窗口（Wave C）→ 确认（Wave D）。本轮只执行到 **Wave A**（含诊断波中的 D3），任务书在 `.tmp/trace/tactile_est_v5.jsonl` 首行（sha256 见 trace_summary）。

- 主指标：`consumer.vector_mag`（|F|/mg）的 **未标定 R²**；判定分组用 **stairs_interp_mean = 10.5/13.5cm 算术平均**；16.5/18cm 只用于 A-P3「外推不退化」；19.5cm 只作描述。
- 噪声底：D3 在同一份数据（`arms/A0`）上重训 seed 0/1/2，逐 case 求 R² 的 sd，staris_interp 的平均 sd = **0.0844** ⇒ **MDE = 1.5·√2·σ = 0.1791**；绝对下限 0.10；阈值取两者较大值。
- 单 seed 且 |Δ|<0.02 一律标 `noise`（工具 `.tmp/tactile_est_v5/tools/rollup_waveA.py`）。

### 10.2 数据与四臂

37 个 npz（65 列 = 53 维本体感觉 + 12 维归一化执行器力矩），全部带特权列（`priv_foot_z/clearance/foot_vz/contact_true/terrain_h/base_z`，**不进特征**，只留给 D1 信息上限探测）。

| 臂 | 训练集 | 训练 case | 帧数 | 训练耗时 |
| --- | --- | --- | --- | --- |
| A0 | 先行 10 case（flat、flat×stair16.5 策略、rough seed 0/3/4、stair15、stair15×两条高楼梯策略） | 10 | 66 548 | 177.1 s |
| A1 | A0 + 同类型回合加倍（10 个 `*_x4`） | 20 | 90 994 | 239.6 s |
| A2 | A0 + 新高度类型 **4.5/6/7.5/9/12cm** | 15 | 82 402 | 219.0 s |
| A3 | A2 + 每高度第二策略（5 条 `pol2`） | 20 | 98 541 | 248.1 s |

训练高度覆盖：0（平）、4.5、6、7.5、9、12、15cm + rough（2–5cm，seed 0/3/4）。留出：**插值** 10.5/13.5cm、**legacy** 16.5/18cm、**外推** 19.5cm、rough seed 1/2。四臂 ckpt 的 config 互差为 `{}`（唯一差异 `data_dir`）⇒ 单因子成立；n_params 全 78 492。

### 10.3 三名 stairs 均值（未标定 vector_mag R²）

| 臂 | 插值 10.5/13.5 | legacy 16.5/18 | 外推 16.5/18/19.5 |
| --- | --- | --- | --- |
| A0 | 0.1012 | 0.4452 | 0.3090 |
| A1 | 0.1380 | 0.5173 | 0.2999 |
| **A2** | **0.3807** | 0.4765 | 0.3026 |
| **A3** | **0.4043** | **0.5258** | **0.4190** |

逐 case（R² / 斜率 / 均值比 / 接触 F1 / 支撑相帧数）：

| case | 语义 | A0 | A1 | A2 | A3 |
| --- | --- | --- | --- | --- | --- |
| stair10_5 | 插值 | 0.040/0.573/0.915/0.930/736 | 0.011/0.647/0.897/0.935 | 0.367/0.641/0.977/0.955 | 0.311/0.688/0.969/0.957 |
| stair13_5 | 插值 | 0.162/0.548/0.979/0.938/698 | 0.265/0.642/0.973/0.935 | 0.395/0.659/1.062/0.957 | **0.497/0.788/1.067/0.967** |
| stair16_5 | legacy | 0.429/0.665/1.020/0.966/500 | 0.520/0.767/1.058/0.969 | 0.447/0.677/1.090/0.972 | 0.462/0.590/1.020/0.970 |
| stair18 | legacy | 0.462/0.978/1.078/0.968/630 | 0.515/0.965/1.136/0.975 | 0.506/0.886/1.162/0.972 | **0.590/0.813/1.062/0.975** |
| stair19_5 | 外推 | 0.037/0.865/1.027/0.929/606 | −0.135/0.782/1.073/0.937 | −0.045/0.667/1.096/0.942 | **0.205/0.724/1.051/0.941** |
| rough_s1 | rough | 0.369/0.564/1.070/0.987 | 0.412/0.642/1.103/0.989 | 0.302/0.607/1.101/0.986 | 0.348/0.628/1.082/0.984 |
| rough_s2 | rough | 0.135/0.332/1.077/0.990 | 0.259/0.411/1.094/0.992 | 0.116/0.371/1.087/0.990 | 0.134/0.374/1.074/0.989 |

### 10.4 预注册判据

| 判据 | 内容 | 阈值 | 观测 | 结论 |
| --- | --- | --- | --- | --- |
| A-P1 | A2−A0 插值均值 | ≥0.1791 | **+0.2796**（逐 case +0.326/+0.233） | PASS |
| A-P2 | A2−A1 插值均值（A1=数量上界） | ≥0.1791 | **+0.2427** | PASS |
| A-P3 | A2 legacy 相对 A0 | ≥−0.05 | +0.0313 | PASS |
| A-P4 | A3−A2（多因子） | 数值不判 | +0.3032 | INCONCLUSIVE |
| A-P5 | max over A1/A2/A3 of F1(A0)−F1(arm) | ≤0.01 | 0.0034 | PASS（noise=True） |
| L5 | 方向（A2>A0、外推不退化） | — | 对齐 3/3 | PASS（方向） |

### 10.5 误差结构与不确定度

每 case 现在同时报 **bias / slope / scatter**（scatter = 仿射 oracle 拟合后的残差 R²，样本内）与 **按回合 cluster bootstrap 2000 次的 95% CI**（`report_<arm>.json` 的 `decomp`/`bootstrap`）。示例：

- A0 stair10_5：bias −0.047、slope 0.573、scatter 0.312，R² 95% CI [−0.16, +0.20]；
- A3 stair13_5：bias +0.040、slope 0.788、scatter 0.589，CI [+0.43, +0.55]；
- A3 stair18：bias +0.039、slope 0.813、scatter 0.655，CI [+0.56, +0.62]。

读法：**bias 小、slope 明显 <1、scatter 高** ⇒ 主要误差是「幅值压缩 + 散布」，不是整体偏高/偏低。CI 宽的原因：每 case 仅 500–736 个支撑相帧，且窗口 stride=2 有 50% 重叠、回合内强自相关。

### 10.6 能力边界（v5 Wave A 之后）

- **可用**：接触判定全 case F1 0.93–0.99、onset 0.87–0.99；支撑相 |F| 的**相对幅值**在 10.5/13.5/16.5/18cm 上 R² 0.31–0.59（A3），已可用于「哪只脚在承重、承重变化趋势」。
- **仍不可用**：绝对幅值的精度——斜率 0.59–0.89（多数 <0.8），A0 的全局仿射标定只能把斜率拉到 0.67 左右；19.5cm 仅 A3 到 +0.205（CI 跨 0）；rough seed 2 仍只有 0.13。
- **覆盖边界**：训练高度只到 15cm、策略只 4 条 + 交叉，>19.5cm、未见策略、未见速度、真机域均未验证。
- **口径边界**：楼梯上 contact 帧里只有 ~17–22% 是 STANCE（接触在楼梯上高频抖动，每帧重触地都会重置 5 帧 LANDING 窗口）⇒ 支撑相口径样本少，所有 R² 的 CI 都宽。
- **后续波次**：Wave B（解耦/容量矩阵）、Wave C（window 40）、Wave D（3 seed）、D1（特权输入上界）、D2（kNN 歧义）见 **§11**。本章写作时它们尚未跑，所以「斜率上不去是因为输入歧义」在 §10 里还只是机制假设；§11 的结论是**该假设不成立**：干净特权输入几乎无增益（ΔR² −0.0024），旧稿「+0.13 上界」是 `priv_contact_true` 泄漏。

### 10.7 结论

1. **「补高度类型」是本轮唯一被确认的杠杆**：只换数据（A2），插值留出 0.101 → 0.381；A3 进一步到 0.404，且 legacy/外推同时不降（A0→A3：0.445→0.526、0.309→0.419）。
2. **加同类型数量远不如加类型**：A1 帧数比 A2 还多 10%，插值只到 0.138（A2−A1 = +0.243）⇒ 之前的瓶颈是**高度覆盖的缺口**，不是帧数。
3. **接触通道零代价**：四臂 F1 全部 0.928–0.992，A-P5 最大下降 0.0034 ⇒ 后续加数据不必担心接触退化。
4. **幅值压缩（slope<1）没有随数据消失**：A3 也只有 0.59–0.81；这与 §9 的结论一致——数据解决「看不到的高度」，不解决「看到的帧幅值偏小」。§11 用 Wave B/C 与 D1/D2 拆这个问题：结构侧**更深**（3 层共享，slope 0.67–0.88）有正收益，解耦与更长窗口无效；输入侧**没有**找到可挖空间（干净特权输入 ΔR² −0.0024）。
5. **可交付最优**：A3 权重（`.tmp/tactile_est_v5/ckpts/A3_s0.pt`）+ 现场分组标定，绝对阈值仍不可用，只用相对量与接触。

### 10.8 复现（Wave A）

    # 数据（37 case，含特权列；约 216s）
    .venv\Scripts\python.exe -X utf8 scripts\collect_tactile_data.py --out .tmp\tactile_est_v5\data --include-v5 --include-pol2 --include-a1 --privileged
    # 臂目录（硬链接）
    .venv\Scripts\python.exe -X utf8 .tmp\tactile_est_v5\tools\make_arm_dirs.py
    # 四臂训练（每臂 40ep / stride 2 / seed 0 / features 65）
    .venv\Scripts\python.exe -X utf8 scripts\train_tactile_estimator.py --data-dir .tmp\tactile_est_v5\arms\A2 --out .tmp\tactile_est_v5\ckpts\A2_s0.pt --features 65 --epochs 40 --seed 0
    # 噪声底 D3（seed 1/2 重训 + 评估聚合）
    pwsh -File .tmp\tactile_est_v5\tools\run_d3v5.ps1
    # 每臂 标定 → 评估（7 个留出 case，支撑相口径 + bootstrap）
    pwsh -File .tmp\tactile_est_v5\tools\run_eval_arms.ps1
    # 判据

## 14. 估计器侧的阻塞与修改点（2026-10-23）

§13 给出的是「要不要上闭环」的系统结论；**只看估计器这一层**的阻塞清单、六个方向的逐项分析（在线闭环对估计器的五条硬要求、评估靶子、任务定义、前瞻接触头、策略多样性数据、接触不一致与降频）见 [doc/tactile_estimator_blockers.md](tactile_estimator_blockers.md)。

其中与本章数字直接相关的三条新证据：①主口径（contact ∩ STANCE）只覆盖 17–28% 的接触帧；②时移扫描最优就是 0 帧（±1 帧 R² 从 +0.515 掉到 +0.198/−0.195），接触起点在当帧识别、无提前量；③因果低通 10 帧后 R² 0.776–0.884 但标签标准差只剩 0.49–0.61 ⇒ 降频只是任务重定义。

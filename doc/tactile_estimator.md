# 足底触觉估计器（Tactile Estimator）

从**本体感觉**（53 维观测：关节角/角速度、机身姿态/角速度、上一步动作、机身系线速度、三轴指令）估计上一轮加入的足底触觉量。
盲走时代的观测里没有足底信息；现在把「足底有什么」当作**可估计量**，而不是传感器输入。

## 1. 估计什么、为什么

| 量 | 参考目标 | 本文结论 |
| --- | --- | --- |
| 三维接触力（世界系，4 足 × 3 轴） | `tactile.force_vec`（N） | **可估计**：留出地形 RMSE 24–30 N（0.16–0.20 体重）、R² 0.56–0.81 |
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

`robotdog/tactile_estimator.py`：**单层 GRU（hidden 128）+ 四个头**（力 12 / 接触 4 / 面积 4 / CoP 8），窗口 20 步、stride 2。
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

## 6. 复现

```powershell
$env:PYTHONPATH="."
.venv\Scripts\python.exe -X utf8 scripts\collect_tactile_data.py --out .tmp\tactile_estimator\data
.venv\Scripts\python.exe -X utf8 scripts\train_tactile_estimator.py --out .tmp\tactile_estimator\estimator_final.pt --window 20 --hidden 128 --epochs 40
.venv\Scripts\python.exe -X utf8 scripts\eval_tactile_estimator.py --checkpoint .tmp\tactile_estimator\estimator_final.pt --out .tmp\tactile_estimator\report_final.md
```

产物（全部在 `.tmp/tactile_estimator/`）：`report_final.md` / `.json`（逐地形指标）、`report_final.axes.txt`（分轴表，由 eval 脚本产出）、`estimator_final.pt` / `.json`（val 指标）、`data/*.npz` 与 `data/collection_summary.json`、`estimator_v2.pt`（消融）。

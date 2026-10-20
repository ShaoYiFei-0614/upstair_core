# 足底触觉（TactileStep 对应物）

实现：`robotdog/tactile.py`；接入：`robotdog/env.py`（观测/相位机/`info["tactile"]`）、
`robotdog/reward.py`（三个相位条件策略）；配置：`config.yaml` 的 `env.tactile` 与
`env.obs.tactile` / `env.obs.privileged_tactile` / `reward.tactile_*`。

参考论文：TactileStep（arXiv:2609.28959，CoRL 2026，清华，Unitree G1）。论文的核心是
**足底触觉特征本征提取 + 相位条件奖励 + 非对称 critic**：每只脚从 taxel 阵列提取
`[F̄, p_cop, Ā]`（归一化法向合力、压力中心、接触面积），作为**部署期观测**（feature-level
sim2real，不迁移 raw taxel）；仿真里用 raycast 分配刚体接触力到虚拟 taxel 网格并做 kNN
空间扩散，真机用薄型压力鞋垫 + MLP 标定。

## 1. 本仓的测量（`TactileSensor.update`）

每控制步（50 Hz）对四只脚各算一次：

| 量 | 定义 | 归一化 | 说明 |
| --- | --- | --- | --- |
| `force_n` | 该足**法向**接触力之和 `Σ|f_n|` [N] | ÷ `force_ref`（默认 = 模型总重 m·g） | 即论文的 `F̄`（体重倍数） |
| `force_tan` | 该足切向力之和 [N] | ÷ `force_ref` | 用于后续滑移检测（本版未进策略） |
| `area` / `area_norm` | 有力 taxel 数 × taxel 面积；归一化 ÷ 足底总面积 | — | 即论文的 `Ā`；`taxel_force_min` 滤噪声 |
| `cop` / `cop_norm` | 力加权接触中心（足体系，m）；归一化到 [-1,1] | ÷ 半足长/半足宽 | 即论文的 `p_cop` |
| `margin` | CoP 到足底虚拟矩形边界的归一化距离，`min(1−\|cop_norm\|)` ∈ [0,1] | — | 支撑裕度：越靠边越接近 0 |
| `vz` | 足端竖直速度（世界系，向下为负）[m/s] | — | 相位机与 PreLanding 惩罚用 |
| `delta_force_norm`、`impact_peak` | 与上一步的 F̄ 变化、LANDING 窗内 F̄ 峰值 | ÷ `force_ref` | Landing impact 用 |
| `cop_jump` | 与上一步的 CoP 归一化位移 | — | Stance support 的稳定性项 |

**虚拟 taxel 网格**：足底矩形 `sole_length × sole_width` 上 `n_rows × n_cols` 个 taxel 中心；
每个仿真接触点先按 `mj_contactForce` 取法向/切向力，再把接触点投到最近的 taxel，然后用
**行归一化高斯 kNN 核**（σ = 0.7×格距，半径 `spread_cells` 格）扩散到邻域 —— 单点接触
被抹成一两千平方毫米的“压力斑”，与真机鞋垫的空间扩散同量级。

**sim2real 边界（重要）**：本实现**不读地形真值** —— 力来自 MuJoCo 接触对，但只用
「接触力/接触点」这种真机上由鞋垫也能测到的量；台阶高度、地形剖面一概不查。因此它属于
「有真机对应物的传感器口径」，而不是 `base_height` 那种把仿真真值塞进策略的特权量。
（真机上还需要一层 MLP 把鞋垫原始 taxel 标定成本仓的 `[F̄, Ā, CoP]`，见论文 §IV。）

## 2. 四相位状态机

`SWING → PRELANDING → LANDING → STANCE → SWING`（`robotdog/tactile.py` 的常量与
`PHASE_NAMES`）：

- **Swing**：无接触、且（离地 > `prelanding_clearance` 或 下落速度 ≥ −`prelanding_vz`）。
- **PreLanding**：无接触、离地 < `prelanding_clearance` **且** 竖直速度 < −`prelanding_vz`
  （足端正在向下接近地面）。
- **Landing**：**包含首次触地那一步在内共 `impact_window_steps` 步**（默认 5 ⇒ 触地后连续 5 帧 LANDING），窗内记录 `impact_peak`；
  触地首步记录 `impact_peak`。
- **Stance**：接触且已过 Landing 窗口。

相位机是**逐足**的（4 个 0–3 整数），`info["tactile"]["phase"]` 报的是相位**名**，便于人类读；
`TactileState.phase` 内部用整数编码，奖励结算用整数比较。

## 3. 三个相位条件策略（`reward.py` 第 11 段）

| 项 | 相位门控 | 公式（每项先逐足算再对四足取 mean） | 默认权重 |
| --- | --- | --- | --- |
| `tactile_prelanding` | PreLanding | `w · clip(−vz / prelanding_vz_ref, 0, 1)²` | 0（关闭） |
| `tactile_impact` | Landing | `w · [clip(F̄/impact_force_ref,0,2) + clip(\|ΔF̄\|/impact_delta_ref,0,2) + clip(peak/impact_peak_ref,0,2)]` | 0（关闭） |
| `tactile_stance` | Stance | `w · [clip(Ā/stance_area_target,0,1) + margin − clip(cop_jump/stance_cop_jump_ref,0,1)]` | 0（关闭） |

设计意图（与论文一一对应）：

1. **PreLanding compliance**：落地前还带着大的向下速度 = 硬砸；平方项让“轻轻放”有梯度、
   “砸下去”迅速饱和。只罚**向下**（向上抬脚本不该被罚）。
2. **Landing impact**：落地窗内平均法向力 + 力变化 + 峰值三个分量，各自按参考量级归一化后
   相加（1.0 ≈ 刚好到参考量级，2.0 封顶）。
3. **Stance support**：奖励**接触面积**（到目标即封顶，避免追求满掌）与 **CoP 裕度**，
   罚 CoP 跳变 —— 支撑越稳、越靠足底中心，支撑质量越高。

三个权重默认全 0，且 `compute_reward` 在“三项权重都为 0”时**整段跳过**：不产生键、
不产生计算，逐步奖励与旧实现逐位一致（`tests/test_tactile.py` 钉住）。

## 4. 观测接入（三档，逐级加信息）

| 开关 | 维数 | 谁看到 | 用途 |
| --- | --- | --- | --- |
| `env.obs.tactile: true` | +16（4 足 × [F̄, Ā, CoPx, CoPy]） | **actor** | 论文的部署期观测；盲走约束已由用户在 2026-10-20 作废，可正面评估 |
| `env.obs.privileged_tactile: true` | +16 | **仅 critic**（非对称 AC） | 先只给价值函数，评估“特权触觉”是否有增益 |
| 都不开（默认） | +0 | — | 与历史 checkpoint 逐位兼容 |

`step()` 每步返回 `info["tactile"]`（力/面积/CoP/相位/vz/峰值等），无论观测开关是否打开；
它只是**只读诊断**，不进观测也不进奖励（除非奖励权重非 0）。

## 5. 怎么打开（A/B 最小改动）

```yaml
env:
  tactile:
    enabled: true          # 必须先开传感器
  obs:
    tactile: true          # A 臂：actor 看触觉（+16 维）
    # privileged_tactile: true   # B 臂：只给 critic（actor 维数不变）
reward:
  tactile_prelanding: -0.5   # ① 落地前向下速度惩罚
  tactile_impact: -1.0       # ② 落地冲击惩罚
  tactile_stance: 0.8        # ③ 支撑质量（面积 + CoP 裕度）
```

注意：`env.tactile.enabled=false` 时若配了 `obs.tactile` 或非零 `reward.tactile_*`，
`Go2StairsEnv.__init__` 会**构造期报错**（而不是静默忽略）。

## 6. 已知边界与后续工作

- **面积/CoP 是“支撑质量”的代理量**，不是脚掌压力分布本身；论文指出它对鞋垫标定误差敏感，
  真机部署前要做 MLP 标定与温度/磨损漂移检查。
- ⚠ **CoP 的绝对位置依赖虚拟网格的坐标约定**：Go2 足端是球面点接触（不是平面鞋垫），
  网格是覆盖足底的 6×4 cm 矩形；实测站立时 CoP 恒落在网格一角（`cop_norm≈(−0.73, ±0.25)`、
  `margin≈0.275`、`Ā≈0.20`），行走时最大 |CoP|≈0.73 —— 力加权中心与「接触点投影」一致，
  但**不能把 `cop` 当作足底真实压力中心的空间位置**。用法：`cop_jump`/`margin`/随载荷的
  相对移动方向（这些与网格原点无关）；要绝对位置需先把网格中心对齐足端球心投影并标定。
- **切向力/滑移没进策略**（`force_tan` 已测）：滑移检测需要再确认摩擦系数与步态时序，
  后续可加 `slip` 项。
- **kNN 扩散参数未与真机鞋垫匹配**（`spread_cells`/σ 是几何估计）；开关级 A/B 前建议先用
  `.tmp` 探针核对 `area` 的量级（当前 6cm 台阶上 0–4.8 cm²、F̄ 峰值 ≈0.8 体重）。
- **相位机阈值是按当前 2 Hz 爬行步态定的**（`prelanding_clearance=0.06`、`prelanding_vz=0.05`）；
  换 trot/高步频后要重新标定，否则 PreLanding 窗口可能不存在或恒存在。
- 与既有 contact 奖励组（`foot_scuff`/`foot_impact`/`load_share`/`com_margin`）的关系：那四项
  是 2026-10-17 按“接触事件”直接给的权重（已否决、默认关）；本模块提供的是**相位结构化**的
  触觉量，两者可共存但不要同时开同类项（会重复惩罚同一事件）。

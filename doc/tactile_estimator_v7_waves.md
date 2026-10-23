# v7 Wave A–E 完整设计（rev4，按 2026-10-23 评审意见重排）

> 上游：doc/tactile_estimator_v7_waves.md(rev3)、doc/tactile_estimator_blockers.md、.tmp/trace/tactile_est_v7.jsonl。
> rev4 的三处结构性变更：① 取消原 Wave 1 的假设检验，数据扩展改为无条件执行（Wave A）；② 新增容量复测波（Wave B）回答「新数据下容量最优是否变化」；③ Wave C/D/E 走快路径，只测组合臂，归因按需补测。

## 0. 变更对照（rev3 → rev4）

| 项 | rev3 | rev4 |
|---|---|---|
| 数据 | 20 策略 × 5 高度 × 2 回合 = 100 case | **10 策略 × 12 地形条件 × 2 回合 = 240 回合**（策略先过零摔筛选；rough 进训练） |
| 原 Wave 1 | 检验「策略多样性是否有用」的 gate | **取消**：数据扩展无条件做，扩展效果只作描述性报告 |
| 容量结构 | 固定 3×128 | **新增容量阶梯复测**：1×128 / 2×128 / 3×128 / 6×128（+3×256 可选） |
| 双频 | 2×2 四臂全跑 | **快路径**：B3 组合臂先行（3 seed），通过即进 Wave D；失败才补 B1/B2 |
| 结构分解 | D0–D4 十五次训练 | **快路径**：D0（复用）+ D3（3 seed）；通过后可选 D4；失败才补 D1/D2 |
| 成本 | ≈ 15.8 h | **≈ 5.5–7.5 h** |

## 1. 共同协议（不变）

| 项 | 约定 |
|---|---|
| 主口径 | contact > 0.5 且 phase > 2.5（STANCE），标签取在消费帧 |
| 主指标 | consumer.vector_mag.r2（未标定，mg） |
| 阈值 | max(0.05, MDE_arm)，MDE = 1.5·√2·σ/√3，σ 取本波同族 3 seed 的 per-case sd |
| 种子 | 判据臂 3 seed（0/1/2）；快路径的补测臂 2 seed；同 seed 配对 |
| 单因子 | 除被检因子外 config/data_dir/epochs/lr/early_stop/stride/batch 全同 |
| 基线 | 每波基线在同批数据、同脚本版本、同 seed 数上重训，禁止搬旧数字 |
| 快路径披露 | 只测组合臂时，报告必须写 ATTRIBUTION_NOT_MEASURED（效应成立，但未分离 A/B 贡献） |
| 禁止 | 留出拟合标定/聚类；calibrated 数字作判据；只报均值；vector_z 冒充 vector_mag |

## 2. 数据（Wave A，无条件执行）

### 2.1 策略集合：10 个零摔策略

筛选规则：在**完整 12 条件 × 2 回合**下 0 摔倒。候选来自 v7 探针（137.2 s，100 rollout）里 5/5 通过且 0 摔的 10 个：

| # | 策略 | ckpt |
|---|---|---|
| 1 | l0_v2 | runs/l0_v2/checkpoints/stair15_final.zip |
| 2 | l1_all | runs/l1_all/checkpoints/stair15_final.zip |
| 3 | l1_st_v2 | runs/l1_st_v2/checkpoints/stair15_final.zip |
| 4 | asym15 | runs/asym15_ladder/checkpoints/stair15_final.zip |
| 5 | crawl24_15 | runs/crawl24/checkpoints/stair15_final.zip |
| 6 | ab_15 | runs/ab_15_ctl/checkpoints/stair15_final.zip |
| 7 | hi165 | runs/hi165/checkpoints/stair16_5_final.zip |
| 8 | hi195 | runs/hi195/checkpoints/stair19_5_final.zip |
| 9 | lr15_9e5 | runs/lr15_9e5/checkpoints/stair15_final.zip |
| 10 | lr15_7e5 | runs/lr15_7e5/checkpoints/stair15_final.zip |

被剔除（有摔）：l1_pl、asym_upstairs9、crawl16_9、crawl21_10_5、crawl22_12、crawl23_13_5、ab_6、asym12、asym13_5、lr15_2e4；完全爬不上的：l1_st、l1_st2、l0_base。

### 2.2 12 个地形条件（每个策略 × 12 × 2 回合 = 240 回合）

| 类别 | 条件 | 参数 | 说明 |
|---|---|---|---|
| 平整 | flat | flat_spec() | vx 0.45 |
| 平整 | stair4_5 | stair_from_height(4.5, tread=0.30, n_up=8) | vx 0.45 |
| 平整 | stair6 | 6.0 cm | vx 0.45 |
| 平整 | stair9 | 9.0 cm | vx 0.45 |
| 平整 | stair12 | 12.0 cm | vx 0.42 |
| 平整 | stair15 | 15.0 cm | vx 0.42 |
| 粗糙 | rough_a2 | rough_spec(0.02, 0.03, seed=0) | amp 2–3 cm，vx 0.35 |
| 粗糙 | rough_a3 | rough_spec(0.03, 0.04, seed=3) | amp 3–4 cm |
| 粗糙 | rough_a4 | rough_spec(0.04, 0.05, seed=4) | amp 4–5 cm |
| 粗糙 | rough_s5 | rough_spec(0.02, 0.05, seed=5) | amp 2–5 cm |
| 粗糙 | rough_s6 | rough_spec(0.02, 0.05, seed=6) | amp 2–5 cm |
| 粗糙 | rough_s7 | rough_spec(0.02, 0.05, seed=7) | amp 2–5 cm |

说明：本仓 rough 地形（robotdog/tools/rough.py）是「平地走廊 + 随机方块高度场」，与楼梯几何互斥 —— 不存在「某高度的 rough 楼梯」这种组合。因此 5 高度 × 2 表面无法字面实现，rev4 用「6 个平整条件（含平地）+ 6 个粗糙条件（振幅带 × seed）」凑成 12 条件，正好满足 10×12×2 = 240 回合的目标，并把「不平整度」拆成振幅带（2–3 / 3–4 / 4–5 / 2–5）与 seed 两个维度。

### 2.3 留出（不动）

| 组 | case | 用途 |
|---|---|---|
| stairs_interp | stair10_5、stair13_5 | 插值高度 |
| stairs_legacy | stair16_5、stair18 | 外推（历史可比） |
| stairs_extrap | +stair19_5 | 外推上界 |
| rough_holdout | rough_s1、rough_s2（seed 1/2，amp 2–5） | **不同种子的随机不平整**，训练只见过 seed 0/3/4/5/6/7 |

### 2.4 策略漂移评测集（新增，只评不训）

用被剔除的 5 个策略（l1_pl、crawl21_10_5、ab_6、asym12、crawl23_13_5）× 3 高度（4.5 / 9 / 15）× 1 回合 = 15 个 rollout，只做「容量选择是否在未见策略上更好」的描述性检查；摔倒不剔除（只用摔倒前的帧，meta 记 fall）。

### 2.5 摔落审计（关键流程，不再事后补救）

1. 用最终矩阵先采到 .tmp/tactile_est_v7/data_probe（240 回合，约 10 min）；
2. 逐 case 读 meta.fall_episodes / success_episodes；**任一条件出现摔倒 ⇒ 该策略整条剔除**（并记录原因）；
3. 剔除后重建 arms 目录（N 策略 × 12 × 2 回合）；若 N<10，允许 216–240 回合之间的实际值，报告写实际回合数；
4. 无摔则直接把 data_probe 转正为 data（零重复采集）；
5. 采集脚本必须用与 probe 相同的 reset seed 规则（seed=1000*(seed+offset)+e+1），保证「探针=采集」可复现。

### 2.6 帧数预算（按 v6 实测：楼梯 406 步/回合、平地 700、rough 700）

| 部分 | 回合 | 步数 |
|---|---|---|
| 楼梯 5 高度 × 10 策略 × 2 | 100 | ≈ 40,600 |
| 平地 10 × 2 | 20 | 14,000 |
| rough 6 条件 × 10 × 2 | 120 | 84,000 |
| **合计** | **240** | **≈ 138,600** |

对应训练窗口（stride 2）≈ 68,000，验证 ≈ 7,500。相对 A3_qty（105,083 帧 / 42,530 窗口）帧数 +32%，但**条件覆盖从 22 格扩到 120 格**（策略 22→10 但每策略覆盖 12 条件而非 1–2 条件）。

## 3. Wave A：数据扩展 + 新基线（无 gate，描述性）

| 动作 | 内容 |
|---|---|
| A1 | 采集 240 回合（约 10 min）+ 摔落审计 + 重建 arms |
| A2 | L2 数据契约：65 列、force_norm、无 NaN、per-cell 覆盖表（episodes/frames/reach_rate/contact_rate）、与 v6 同名文件逐位比对、留出未进训练 |
| A3 | 训练新基线 S3（3×128）3 seed（≈ 15 min/次） |
| A4 | 在新基线 + 7 个留出上重评，得到 v7 起点：stairs_interp / legacy / extrap / rough / contact F1 / δ* / 承重 F1 / ΣFz |

报告口径：数据扩展的效果 = 新基线 vs v6 的 S3_qty（同口径、单 seed 0 对比），**只作描述**（不设 PASS/FAIL，因为扩展本身是既定方向）。

## 4. Wave B：容量复测（回答「结构结论是否被推翻」）

### 4.1 被检验的结论

v5/v6 的结论是「共享 3×128（277k）是甜点，继续加容量（302k / 502k / 1.10M）都变差」，而不是「1×128 最好」——1×128（78k）一直是**最低成本基线**（v6 三 seed 均值 interp 0.3201 vs 3×128 的 0.5763）。真正会被新数据推翻的是**容量饱和点**：数据从 105k 帧/22 格扩到 139k 帧/120 格后，偏差-方差平衡会移动，6×128 甚至 3×256 可能重新占优。

### 4.2 阶梯（全部共享主干，不用解耦 —— 解耦在 v5/v6 已被两次否证）

| 臂 | 形状 | 参数量 | seed | 角色 |
|---|---|---|---|---|
| C1 | 1×128 | 78,492 | 3 | 小容量锚点 |
| C2 | 2×128 | 177,564 | 2 | 中途点（诊断） |
| C3 | 3×128 | 276,636 | 3 | 现最优 |
| C4 | 6×128 | 573,852 | 3 | 深 |
| C5 | 3×256 | 1,044,764 | 2 | 宽（可选，最后跑） |

规则：epochs 固定 40 + lr_gamma 0.98 + early_stop 8（保持预算一致）；若 C4/C5 在末轮仍在降 val，报告写 BUDGET_LIMITED，不擅自加轮。

### 4.3 判据

| 编号 | 估计量 | 阈值 | 性质 |
|---|---|---|---|
| B-P1 | C4 − C3 stairs_interp ΔR² | ≥ max(0.05, MDE_3seed) | gate（是否升级容量） |
| B-P2 | C4 contact F1 drop | ≤ 0.01 | 守卫 |
| B-P3 | C4 rough ΔR² | ≥ −0.05 | 守卫 |
| B-P4 | C1 vs C3 | 报 Δ（若 |Δ|<0.02 ⇒ 记 CAPACITY_NOT_NEEDED） | RECORDED |
| B-P5 | 策略漂移集（2.4）上的 C3 vs C4 | 报 Δ | RECORDED |
| B-P6 | 各臂训练墙钟与窗口/参数比 | 表 | RECORDED |

选定规则（预注册）：B-P1 且 B-P2 且 B-P3 通过 ⇒ 后续波次用 C4；否则用 C3。C5 只在 B-P1 通过且 C4 仍在降 val 时才跑。

### 4.4 成本

3 seed × (9 + 15 + 26 min) = 150 min（C2 2 seed ≈ 25 min，C5 可选 31 min）。

## 5. Wave C：双频输出 + 标签去抖（快路径）

因子与实现同 rev3：A = 因果 5 帧均值头（mag5_head，损失仅 stance&contact 帧）；B = 接触目标中心 3 帧多数票。

### 5.1 快路径（按你的要求：先看效果，不先做归因）

1. 先训 **B3（A+B）3 seed**，基线 = Wave B 选出的最佳臂（同数据同 seed 数）；
2. B3 同时满足判据 ⇒ **直接进 Wave D**，报告写 ATTRIBUTION_NOT_MEASURED；
3. B3 不满足 ⇒ 再补 B1（仅 A）、B2（仅 B）各 2 seed 做诊断，据此决定是放弃、还是只保留其中一个因子；
4. 若 B3 的中心 R² 达标但瞬时通道掉 >0.02（W2-P6）⇒ 触发预注册退路 B3b（先训瞬时通道 → 冻结主干只训 mag5 头，3 seed）。

### 5.2 判据（B3 vs 新基线）

| 编号 | 估计量 | 阈值 |
|---|---|---|
| C-P1 | mag5 中心 R²（stance） | ≥ 0.70 |
| C-P2 | R²(+1) | ≥ 0.45 |
| C-P3 | contact F1 drop | ≤ 0.01 |
| C-P4 | label_std_ratio | ≥ 0.70 且披露 |
| C-P5 | 瞬时通道 R² drop | ≤ 0.02 |

成本：B3 3 seed ≈ 45–80 min（视最佳容量）；补测 B1/B2 各 2 seed ≈ +60 min（仅失败时）。

## 6. Wave D：前瞻接触头（快路径）

复用已有 lookahead 管线（k=2 主臂）。

| 编号 | 估计量 | 阈值 |
|---|---|---|
| D-P1 | AUC(k=2) | ≥ 0.93 |
| D-P2 | onset 提前率 | ≥ 0.55 |
| D-P3 | 当前帧接触 F1 drop | ≤ 0.01 |
| D-P4 | 幅值主指标 drop | ≤ 0.02 |

先训 L2（3 seed）≈ 45–80 min；仅当 D-P1 达标但 D-P2 未达，或需要知道 80 ms 上限时，才补 L4（2 seed）。

## 7. Wave E：物理结构分解（快路径）

结构同 rev3：f_z,i = a_i · S，重建项直接监督 vector_mag（与判据同量）。

| 臂 | 内容 | seed | 何时跑 |
|---|---|---|---|
| D0 | 基线（复用 Wave D 最佳配置） | 3 | 复用，不额外训 |
| D3 | a·S + 重建 | 3 | 默认跑 |
| D4 | D3 + 摩擦锥 + CoP 一致 | 3 | 仅当 D3 达标且仍有时间 |
| D1/D2 | 总量辅助 / 分配辅助 | 2 | **仅当 D3 未达标**时做归因 |

### 7.1 判据（D3 vs D0）

| 编号 | 估计量 | 阈值 |
|---|---|---|
| E-P1 | stairs_interp ΔR² | ≥ −0.02 |
| E-P2 | slope（两楼梯各自） | ≥ 0.85 |
| E-P3 | 足间排序 Spearman | ≥ 0.85 |
| E-P4 | argmax 命中 | ≥ 0.93 |
| E-P5 | 承重 F1 | ≥ 0.90 |
| E-P6 | ΣFz 中位相对误差 | ≤ 5% |
| E-P7 | 摩擦锥违反率（D4） | ≤ 2% |
| E-P8 | contact F1 drop | ≤ 0.01 |

成本：D3 3 seed ≈ 45–80 min；D4 可选 +45–80 min；失败补 D1/D2 ≈ +60 min。

## 8. 顺序、成本、风险

~~~
Wave A 数据（240 回合）→ 新基线 S3 3 seed →（描述性报告）
  → Wave B 容量阶梯（C1/C3/C4 各 3 seed，C2 2 seed）→ 选定最佳容量
    → Wave C B3 3 seed →（通过）→ Wave D L2 3 seed →（通过）→ Wave E D3 3 seed →（可选 D4）
~~~

| 阶段 | 训练次数 | 训练时长 | 采集/评估 | 小计 |
|---|---|---|---|---|
| Wave A 数据 | — | — | 采集 10 min + L2 3 min + 评估 6 min | ≈ 20 min |
| Wave A 基线 | 3 | 45 min | 6 min | ≈ 50 min |
| Wave B 容量 | 8（+2 可选） | 150 min（+25 min） | 10 min | ≈ 2.7 h |
| Wave C 双频 | 3 | 45–80 min | 6 min | ≈ 1.2 h |
| Wave D 前瞻 | 3 | 45–80 min | 6 min | ≈ 1.2 h |
| Wave E 结构 | 3（+3 可选） | 45–80 min（+80 min） | 6 min | ≈ 1.2 h |
| **合计** | 20–23 | | | **≈ 6.5–7.5 h** |

可裁剪：C2 取消（−25 min）、C5 取消（默认取消）、D4 取消（−80 min）、失败补测不跑（只有在失败时才需要）。

风险与混杂（写入 trace_summary.confounders）：

1. rough 地形与楼梯几何互斥（本仓实现限制）⇒ 12 条件里 rough 是独立的 6 个条件，不是「某高度的 rough 楼梯」；若要真正的 rough 楼梯需要改 terrain 代码（不在本波范围）。
2. 10 个零摔策略里 3 个是 lr 变体（lr15_9e5/lr15_7e5），策略家族数 < 10；报告必须写清家族构成。
3. rough 全程 700 步 ⇒ 120 回合 rough 占总帧数 60%，会让训练分布偏向 rough（接触率 0.29）；每组指标必须分组报，不能只报总均值。
4. 240 回合只有 2 回合/格，per-cell 方差大；容量比较必须 3 seed 且用 MDE 判；
5. 快路径不测归因 ⇒ 结论只能写「组合有效」，不能写「A 贡献多少」；
6. 仍为单机 CPU float32，无部署延迟测量。

## 9. 实现清单

### Wave A
- scripts/collect_tactile_data.py：CASES_V7（10 策略 × 12 条件，表驱动，每格 2 回合）+ --include-v7；POLICY_SHIFT_CASES（15 rollout，只评）+ --include-polshift；留出守卫扩展（rough seed 1/2 与楼梯留出名单）。
- .tmp/tactile_est_v7/tools/make_arm_dirs_v7.py：data / data_probe / polshift / holdout 四个目录。
- .tmp/tactile_est_v7/tools/fall_audit.py：逐 case 读 meta.fall_episodes，输出剔除清单与最终回合/帧统计。
- .tmp/tactile_est_v7/tools/l2_check_v7.py：契约 + per-cell 覆盖表 + 与 v6 同名文件逐位比对。
- 训练/评估：复用现有 train/eval 脚本（无需改代码）；新增 tools/rollup_A.py 与 tools/rollup_B.py。

### Wave B
- 无需改 estimator（ARCH 用已有的 layers/hidden 参数组合：--layers 1/2/3/6 --hidden 128/256）；ARCH_PARAM_TARGETS 需补 2×128 / 6×128 / 3×256 三个期望值（仅测试用）。
- 测试：tests/test_tactile_estimator_v7.py 断言四个形状的参数量（78,492 / 177,564 / 276,636 / 573,852 / 1,044,764）。

### Wave C / D / E
- 同 rev3 §9 的 W2/W3/W4 实现清单（mag5 头与标签、lookahead 评估、结构分解与重建），本版只改「只跑组合臂」的执行顺序。

## 10. 判据总表

| 波 | 编号 | 估计量 | 阈值 | 性质 |
|---|---|---|---|---|
| B | P1 | C4 − C3 interp ΔR² | ≥ max(0.05, MDE) | gate |
| B | P2/P3 | contact F1 drop / rough Δ | ≤0.01 / ≥−0.05 | 守卫 |
| C | P1/P2 | mag5 中心 R² / R²(+1) | ≥0.70 / ≥0.45 | gate |
| C | P3/P5 | F1 drop / 瞬时 R² drop | ≤0.01 / ≤0.02 | 守卫 |
| D | P1/P2 | AUC(k=2) / onset 提前率 | ≥0.93 / ≥0.55 | gate |
| D | P3/P4 | F1 drop / 幅值 drop | ≤0.01 / ≤0.02 | 守卫 |
| E | P2–P5 | slope / ρ / argmax / 承重 F1 | ≥0.85 / ≥0.85 / ≥0.93 / ≥0.90 | gate |
| E | P6/P8 | ΣFz 中位误差 / F1 drop | ≤5% / ≤0.01 | gate/守卫 |
| E | P7 | 摩擦锥违反率 | ≤2% | gate（仅 D4） |

所有 gate 的阈值统一为 max(绝对下限, MDE_arm)；所有 Δ 必须给逐 case + 组均值 + 按回合 cluster bootstrap 95% CI；|Δ|<0.02 记 noise。
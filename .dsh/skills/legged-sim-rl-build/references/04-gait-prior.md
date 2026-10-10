# 开环步态先验

> 对应 `SKILL.md` 阶段 5。素材取自 upstair_core（Unitree Go2 盲走楼梯 · MuJoCo 3.12 + gymnasium 1.3 + SB3 2.9） 的真实源码与实测记录。
> 具体数值只作**取证**用；换机器人时按文中的「换机器人要改什么」逐条替换。

**未验证项**：文档称楼梯先验对台阶高非单调（2 cm 40% / 4 cm 100% / 6 cm 0%），本次未复跑。

## 1. 步态先验（robotdog/gait.py，191 行，无 mujoco 依赖）
### 1.1 为什么需要（迁移时先判断是否适用）
单进程 `DummyVecEnv`（沙箱禁止 Windows 命名管道，`SubprocVecEnv` 起不来）× 8 环境，PPO 从零学不出步态：两次标定（平地 120k / 150k 步）都以 **0% 成功率、位移约 0.02 m** 收场，学到的策略比零动作站桩**更差**（回合回报 466 vs 496）。原因是**平台期**不是权重 bug：随机关节抖动不产生净推进，前向速度奖励在出现**协调周期步态**之前是平的，50 Hz 采样的 MLP 撞不上。解法 = 残差 RL（Johannink et al. 2019）：先离线搜一条**固定开环**轨迹，策略只学残差。**先验是一条可打印、可存 JSON、可离线重现的轨迹**，不是隐藏控制器（把先验做成相位驱动足端 IK 的方案已否决：不可解释、变黑盒）。
```
target = home + gait_gain * gait.radians(t) + action_scale * action
```
### 1.2 动作管线与两个同名不同义的 `action_scale`
```python
# robotdog/env.py:883-891
def _gait_action(self) -> np.ndarray:
    if not self.cfg.gait_prior or self.cfg.gait_gain == 0.0:
        return np.zeros(self.num_act, dtype=np.float64)
    return self.cfg.gait_gain * self.cfg.gait.radians(self._gait_phase)
# robotdog/env.py:908-911
target = (self.home_angles + self._gait_action() + self.cfg.action_scale * action)
return np.clip(target, self.act_low, self.act_high)     # act_low/high = meta.joint_ranges[:,0/1]
# step(): self._gait_phase = self._gait_t; ...; self._gait_t += self.cfg.control_dt
```
* `GaitParams.action_scale` = **先验幅度**（出货 1.0）；`EnvConfig.action_scale` = **策略权限**（0.30 rad）。两者**刻意分开**；共用一个尺度会让策略探索噪声被先验幅度乘一遍（实测 **2 s 内摔倒**）。
* `EnvConfig.gait_prior` 默认 `False`（纯残差仍可用）；`config.yaml` 的 `env.gait_prior` 默认 `true`。`gait_gain` 默认 `1.0`，无 CLI 开关。
* `control_dt = timestep * decimation = 0.002 * 10 = 0.02 s`（50 Hz）；`ObsConfig.gait_phase=True` 时观测追加 `[sin(theta), cos(theta)]`，`theta = 2π*freq*_gait_phase`（残差策略必须知道自己在周期哪一点）。
### 1.3 `GaitParams` 参数化与 API
```python
TROT_PHASE = (0.0, math.pi, math.pi, 0.0)   # 腿序 FL/FR/RL/RR → 对角小跑（FL+RR 同相）
CALF_SHAPES = ("sin", "dip", "half")
@dataclass
class GaitParams:
    freq: float = 2.0            # Hz
    amp_th: float = 0.0          # 大腿摆动幅值（归一化动作单位）
    amp_ca: float = 0.0          # 小腿摆动幅值
    ph_th: float = 0.0           # 大腿相位偏置 [rad]
    ph_ca: float = 0.0           # 小腿相位偏置 [rad]
    off_th: float = 0.0          # 大腿常量偏置
    off_ca: float = 0.0          # 小腿常量偏置
    calf_shape: str = "sin"      # sin | dip | half
    action_scale: float = 0.70   # 归一化单位 → 弧度（rad per unit）
    leg_phase: tuple[float,float,float,float] = field(default_factory=lambda: tuple(TROT_PHASE))
    def _calf_wave(self, x)      # sin→sin x; dip→-0.5(1-cos x)(≤0); half→-max(0,sin x)
    def leg_phases(self, t) -> np.ndarray    # 2π*freq*t + leg_phase, (4,)
    def actions(self, t) -> np.ndarray       # (12,) 归一化单位
    def targets(self, t, home) -> np.ndarray # home + self.radians(t)
    def radians(self, t) -> np.ndarray       # action_scale * actions(t)  ← 环境真正调用的
    @property period -> float                # 1/freq；freq<=0 → inf
    def normalize(self) -> GaitParams        # 就地夹取，返回 self
    def to_dict(self) -> dict;  def save(self, path) -> Path
    @classmethod from_dict(cls, d) ; @classmethod from_search_json(cls, path, rank: int = 0)
    @classmethod default_trot(cls); @classmethod static(cls);  def summary(self) -> str
def load_gait(path: str | Path | None) -> GaitParams
```
* `actions(t)`：`a = zeros((4,3))`（每条腿 hip/thigh/calf）；**髋列恒为 0**（先验从不做外展）；`a[:,1] = off_th + amp_th*sin(p+ph_th)`；`a[:,2] = off_ca + amp_ca*wave(p+ph_ca)`；`clip(a,-1,1).ravel()`。这条 `(4,3)` 布局与 `leg_phase` 长度 4 是**四足硬编码**（换形态必须重写）。
* `radians()` 而不是 `actions()` 被环境调用：让先验幅度与 `EnvConfig.action_scale` 解耦（否则改动策略权限会静默改变先验）。
* `normalize()`（就地，返回 `self`）：`freq→[0,10]`；`amp_*/off_* → [-1,1]`；`|off|+amp>1` 时整体乘 `s = 1/max(|off|+amp, 1e-9)`（**同时缩 offset 与 amplitude**，保持波形比例）；相位 `atan2(sin,cos)` 折到 `(-π,π]`；`calf_shape ∉ CALF_SHAPES` → `ValueError`；`leg_phase` 必须 4 项 → `ValueError`；**`action_scale` 完全不夹**。它是就地改：`g = p; g.normalize()` 会连 `p` 一起改（`EnvConfig.gait` 与调用方是同一对象）。
* `static()` = `cls(freq=0.0, amp_th=0.0, amp_ca=0.0).normalize()`（等价纯残差）；`period` 是 `inf`，`action_scale` 仍是 0.70。
* `default_trot()` 来自 `_FALLBACK_TROT`（平地 0.550 m/s 那一档胜者，**不是**出货 JSON）：`freq 6.0, amp_th 0.24914502679724, amp_ca 0.5518579715236906, ph_th 1.6754541487333945, ph_ca -2.146505768979483, off_th 0.25539290883226595, off_ca 0.09025271647195035, calf_shape "sin", action_scale 0.90`；`summary: "6.00 Hz, ±0.249 thigh / ±0.552 calf (norm. units), shape=sin, scale=0.90 rad, period=167 ms"`。
### 1.4 JSON (反)序列化格式
* `to_dict()` = `asdict(self)` 且 `leg_phase` 转 `list`；`save()` 建父目录 + `json.dumps(..., indent=2)` + `encoding="utf-8"`。
* `from_dict(d)` **按已知字段名过滤**后构造再 `.normalize()`；`leg_phase` 转 tuple；`calf_shape` 是 `int` → `CALF_SHAPES[int]`（所以 `0` 就是 `"sin"`）。**未知键被静默忽略**：把 `amp_th` 拼成 `amp_thigh` 不报错、只掉到默认 0.0（表现为「先验完全不抬腿」）——写完 JSON 务必 `print(gait.summary())`。
* `load_gait(path)`：`None → default_trot()`；不存在 → `FileNotFoundError(f"gait file not found: {path}")`；**有 `"top"` 键 → `from_search_json(path)`**（优先于普通格式，普通 JSON 里多一个无关 `top` 字段就会炸）；否则 `from_dict`。
* `from_search_json(path, rank=0)`：取 `data["top"][rank]`，用 `entry["gait"]` 并**把顶层 `entry["action_scale"]` 抬进来**；`top` 为空 → `ValueError(f"{path} has no 'top' entries")`；`rank` 越界是裸 `IndexError`。
### 1.5 `artifacts/trot_stairs.json` 里到底存了什么
11,305 B / 397 行。顶层元信息：`objective="stairs"`、`target_speed=null`、`step_height=0.05`、`tread=0.5`、`clearance_weight=6.0`、`seconds=6.0`、`trials=500`、`refine=900`，以及 `"best"`（= `top[0]`，`rank=1`）与 `"top"`（**10 条**，`rank=0..9` 都可用）。每条记录结构：
```json
{"rank":1,"score":22.235335505909532,"action_scale":1.0,
 "gait":{"freq":5.477741882247174,"amp_th":0.2650288316349047,"amp_ca":0.9863652312670887,
         "ph_th":-1.819353252233276,"ph_ca":0.08531503233940636,
         "off_th":0.2933585372533412,"off_ca":0.013634768732911319,"calf_shape":0},
 "eval":{"distance":6.308551063546455,"fell":true,"steps":300,"survived":1.0,
         "mean_speed":1.0572354656849792,"std_speed":0.30455658145875597,
         "mean_height":0.5031929708196063,"height_gain":0.0,"height_max":0.4,
         "apex":[0.4447884685429898,0.4355370206784326,0.39672354698810014,0.37791569408678893],
         "apex_min":0.37791569408678893}}
```
经 `load_gait()` 之后的**真实值**（`calf_shape: 0 → "sin"`，`leg_phase` 缺省 → trot，`action_scale` = 条目上的 **1.0**，不是 dataclass 的 0.70 也不是 fallback 的 0.90）：`freq 5.477741882247174` / `amp_th 0.2650288316349047` / `amp_ca 0.9863652312670887` / `ph_th -1.819353252233276` / `ph_ca 0.08531503233940636` / `off_th 0.2933585372533412` / `off_ca 0.013634768732911319`，`period = 0.18255697721736439 s → 183 ms`，`summary() = "5.48 Hz, ±0.265 thigh / ±0.986 calf (norm. units), shape=sin, scale=1.00 rad, period=183 ms"`。`|off_ca| + amp_ca` 在 float64 下**恰好 == 1.0**，`1.0 > 1.0` 为假 ⇒ `normalize()` 不缩放，JSON 数字位对位保留。`config.yaml` 的 `env.gait_file: artifacts/trot_stairs.json`（路径**相对仓根**，由 `config.py` 经 `paths.WORKSPACE` 解析）。
### 1.6 为另一台机器人录/拟合一条先验的步骤
1. **注意**：产生出货 JSON 的搜索脚本 `scripts/search_trot.py` **不在仓里**（`scripts/` 只有 `download_go2.py` / `play.py` / `train.py`）⇒ `artifacts/trot_stairs.json` 是**不可重建的 artifact**。新机器人必须自己重跑搜索。
2. 写 rollout 评估器：对目标机器人编译场景 → 确定性 reset → 每个控制步 `target = home + 1.0*gait.radians(t)`（等价 `gait_prior=True, action=0, gait_gain=1.0`）→ 走 `PD + mj_step × decimation` → 记录 `(distance, mean_speed, mean_height, apex[4], fell, survived)`。
3. 搜索 7–9 维（`freq, amp_th, amp_ca, ph_th, ph_ca, off_th, off_ca, calf_shape, action_scale`），两阶段规模照抄 artifact（`trials=500` 粗搜 + `refine=900` 精修）。
4. 目标函数按任务定：平地用速度目标（如 `target_speed=0.55`），楼梯用 `objective="stairs"` + `clearance_weight=6.0`（奖励摆动峰值裕量）+ `fell` 惩罚；固定地平线与时长（`seconds=6.0`、`steps=300`）。
5. 结果落盘成 `{"objective":…, "best":{…}, "top":[{rank,score,action_scale,gait{…},eval{…}}, …10 条]}`，`calf_shape` 写整数索引，`action_scale` 写在**条目顶层**（`from_search_json` 从那里取）。
6. 校验：`load_gait(path).summary()`，并在确定性零动作下复测速度/存活/机身高度（本仓 fallback 标定：0.550 m/s、8 s 走 4.39 m、100% 存活、机身 0.268 m）。
7. 换形态必改：`actions()` 里的 `(4,3)` leg/hip-thigh-calf 布局、`TROT_PHASE` 长度 4、`FOOT_GEOM_NAMES`/`LEGS`。**幅值是归一化动作单位**，所以换 `action_scale` 不失效，但换**惯量/尺度/质量分布**必须重搜。
### 1.7 先验的能力边界（迁移时不要指望它）
* **开环楼梯小跑对台阶高不单调**：2 cm → 40%、3 cm → 40%、**4 cm → 100%**、5 cm → 20%、6 cm → 0%。原因：固定 **183 ms** 周期在约 4 cm 处与 50 cm 踏面**共振**。课程起点别选在 2 cm（在先验自身能力之下）。
* 纯先验（零残差）能上 4.4 m 但会摔：1.5 cm → 88%、2.0 cm → 50%、2.5 cm → 12%、3.0 cm → 12%、4.0 cm → 25%。**先验是起点不是解。**
* 摆动峰值 0.1124 m（频率 ×0.6 → 0.1658 m）——开环调参也翻不过去的能力边界约 11 cm 墙。
* 先验越强越要防「策略少学」：有先验时未训练策略第一轮就达标，会让毕业门立刻放行（本仓已取消毕业门，各阶段跑满固定预算）。

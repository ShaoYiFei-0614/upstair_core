# `robotdog/tools/touch.py` —— 足底接触力与前方台阶测距

## 1. 一句话定位

**工具层**（`robotdog/tools/`）里的触觉/测距模块：从 MuJoCo 的接触求解器读出**每只脚的法向力、切向力、滑移比与「蹭立面」标志**，再用一扇**竖直射线**（外加一条锐化距离的平射射线）测出「前方地形抬升多少、台阶还有多远」。它解决的问题是：光有「碰到了没」这个布尔量，没法告诉策略「你左前脚正前方有一级立面，抬脚」。

## 2. 职责边界

**做**：

* `TouchSensor.feet()`：每足法向力 / 切向力 / 滑移比 / `riser_graze` 标志（来自 `mj_contactForce` + 接触系法向）。
* `TouchSensor.forward_probe()`：前向测距，返回 `(hit, range, height, step, far_step)`。
* `TouchReading`：一步的读数容器，外加把读数打包成观测块的 `foot_obs`（9 维）、`probe_obs`（3 维）、`probe_obs_full`（5 维），以及供奖励用的 `trigger()`、`loaded()`、`ahead_of_sensor()`、`total_normal_force`、`as_dict()`。
* `TouchConfig`：所有几何与阈值参数（可序列化进 `summary.json` 并由 `robotdog/training.py:357` 恢复）。
* `cast_ray()`：一条射线的通用封装（归一化方向、零向量、`max_range`、未命中返回 `inf/-1`）。
* `empty_reading()`：首次 `reset` 之前的「什么都没碰到」读数。

**不做**：

* **不做奖励计算**：`trigger()` / `loaded()` / `total_normal_force` 只**提供量**，乘上权重、取平方、做门控都在 `robotdog/env.py` 的 `_reward_terms` 里。
* **不构造观测向量**：`foot_obs` / `probe_obs` / `probe_obs_full` 只返回**本模块的块**；拼进全局观测、归一化、拼 frame 都是 `robotdog/env.py`（`:582-586`）的事。
* **不拥有 `model`/`data`**：它们由环境传入（`__init__` 存引用），本模块不编译模型、不 `mj_step`。
* **不 import `robotdog.env`**，不知道 PPO / 课程 / 训练循环；**也不 import `robotdog.go2_model`** —— 它只按属性名（`floor_geom_id`、`stair_geom_ids`、`foot_geom_ids`）**鸭子类型**读取那个 `meta` 对象。
* **不做接触判定的终止/碰撞逻辑**（那是 `env._contact_flags`）。
* **不写文件、不打印、无副作用**。

## 3. 关键设计决策与理由

### 3.1 为什么这个模块必须存在：规格禁止了接触传感器数据

源码 docstring `Why this module exists` 说得非常直接：

> The original specification (§四) said "do not use contact-sensor data", so the observation carried no touch information at all and the only two contact *facts* in the environment were the boolean masks in `env._foot_contact_mask` (used by the `air_time` shaping) and `env._contact_flags` (used by `body_collision` and the termination test).  Contact **forces** were never read.
>
> That is enough to *punish* a collision but it cannot say "a riser is right in front of your front-left foot, lift it".  This module adds exactly that.

三条后果：

1. **只有布尔遮罩 ⇒ 只能惩罚，不能引导**。`_foot_contact_mask` / `_contact_flags` 能告诉你「你撞了」，但撞之前策略没有任何信号可用 —— 而「抬脚」恰恰必须在**接触之前**发生。
2. **接触力从未被读取** ⇒ 落地冲击、滑移这类连续量完全不可见；高台阶上「落地太狠」与「打滑」都无法塑形。
3. **本模块补的三件事**：每足法向力/滑移比（`feet()`）、`riser_graze`（脚在推**竖直**台阶面，即字面意义的「撞到台阶立面」）、前向测距（`forward_probe()`）。

**被否决的方案**：直接读 `env._foot_contact_mask` 并从中推导力（例如「有接触就算 1」）—— 否决：遮罩没有量级，做不出「落地太狠」的连续惩罚，也做不出滑移比；而且它把工具层绑到环境层的私有属性上（方向反了）。

### 3.2 用**竖直射线扇**而不是斜光束

`TouchConfig.probe_forward` 上方的注释是这条决策的全文：

```
The probe is a **fan of vertical rays**, not a slanted beam.  A vertical
ray measures the terrain height directly, so the reading is immune to the
torso pitching through the trot cycle -- and, unlike a slanted ray, it
never "sees past" a tread it has already climbed over.
```

`forward_probe` 的 docstring 把它展开成三条具体理由：

1. **斜射线必须先下降到下一级踏面的高度才能找到它** —— 所以它对 5 cm 的台阶几乎踩上去才发现（量程与灵敏度都差）。
2. **一旦躯干爬上某级踏面，斜光束会飞过该踏面去报下一级立面** —— 距离读数会**向后跳**，这是一个非物理的观测量，会直接毒化策略。
3. **竖直射线只报「这里地面多高」**，这正是策略需要的量；而且它不受 trot 周期里躯干俯仰的影响 —— heading 同理，`forward_probe` 只用 `yaw`（源码：`the heading uses **yaw only** for the same reason`），不参与任何 roll/pitch 旋转。

**被否决的方案**：单条斜光束（更便宜、更像「激光雷达」）—— 否决，上述三条每一条都足以致命。**另一个被否决的方案**：用 `mujoco.mj_ray` 打**水平**射线当主探针 —— 否决，水平射线只能测「有没有竖直面挡路」，无法给出「地形抬升多少」；它在本模块里降级为**辅助**（见 §3.3）。

### 3.3 平射边射线只用来**锐化距离**，而且只在它报得更近时才采信

`forward_probe` 在扇扫命中之后，额外从 `terrain_height + probe_horizontal_height`（`probe_horizontal_height = 0.045`）沿朝向平射一条射线：

```python
flat_origin = np.array([
    x0 + c * cfg.probe_forward,
    y0 + s * cfg.probe_forward,
    float(terrain_height) + cfg.probe_horizontal_height,
])
flat_dist, _ = cast_ray(self.model, self.data, flat_origin, np.array([c, s, 0.0]))
if np.isfinite(flat_dist):
    flat_range = float(cfg.probe_forward) + float(flat_dist)
    range_m = min(range_m, flat_range)
```

理由（docstring）：`it flies straight at the riser and stops on the vertical face, which is far more precise than the fan spacing -- but it is only trusted when it reports something *nearer* than the fan, because once the sensor origin is past a riser the flat ray happily skips ahead to the next one.`

即：平射射线**精度**远高于扇间距（`probe_span/(probe_rays-1) = 0.60/8 = 0.075 m`），但它有一个已知的**过冲**失效模式（原点越过一级立面后会跳到下一级）。用 `min()` 采信就得到了「精度提升」而不引入那个失效模式 —— 单向的信任，只允许它把读数**拉近**。

扇扫自身的第一级距离也用了一半间距做回退估计：`first_rise_dist = max(d - 0.5 * spacing, float(cfg.probe_forward))` —— 立面位于「这条射线」与「上一条射线」之间，退半个间距是合理估计，并被 `probe_forward` 下界钳住。

**被否决的方案**：只靠平射射线（不加扇扫）—— 否决：它在落差/下坡/传感器越过立面的场景下会退化，而且给不出 `far_step` 与 `height_ahead`。**另一个被否决的方案**：只用扇扫（去掉平射）—— 否决：距离分辨率被间距（0.075 m）限制死，而触发奖励对距离很敏感。

### 3.4 【载荷决策】`geomgroup` 掩码解决自遮挡

源码 `Self-occlusion of the rangefinder` 一节：

> The probes start at the torso and shoot forwards, straight through the volume the front knees sweep.  Casting with "all groups" therefore hits the robot's own legs.  The fix is MuJoCo's `geomgroup` mask: the terrain and floor geoms live in `TERRAIN_GEOM_GROUP` while every robot geom sits in groups 2 (visual) and 3 (collision), so a mask that enables only the terrain group can never see the robot.

模块级常量据此构造：

```python
TERRAIN_GEOM_MASK = np.array(
    [1 if g == TERRAIN_GEOM_GROUP else 0 for g in range(6)], dtype=np.uint8
)
```

（实测值 `[0, 1, 0, 0, 0, 0]`。）注意这是**长度 6** 的掩码（MuJoCo 的 geom group 数是 6），由 `TERRAIN_GEOM_GROUP` 派生，所以改了地形组这里自动跟随。

源码同时点出那条容易忘的副作用：`Geom groups do not affect physics -- but MuJoCo **does** hide groups 3-5 from offscreen renders by default, which is why the terrain group is 1.`（地形组为什么是 1，详见 `docs/robotdog/tools/terrain.md` §3.4 与 `experience.md` §1.2。）

**被否决的方案**：用 `bodyexclude` 排除机器人 —— 否决：只能排除**一个** body，而四只脚分属四个 body。**另一个被否决的方案**：把探针起点挪到机器人前方（几何上避开腿）—— 否决：那会把「传感器标定」与「身体尺寸」绑死，而且躯干俯仰时仍可能扫到；掩码是**构造性**的免疫。

### 3.5 滑移比在**源头**截断（`slip_max = 8.0`）

`feet()` 末尾的注释：

> The ratio is unbounded by construction: a foot that is barely loaded but being scrubbed along a riser edge can read tangential/normal in the hundreds.  That value is meaningless (anything above a few is already "completely sliding") and it feeds both the observation and a **squared** reward term, so cap it here at the source.

```python
slip = np.clip(tangential / np.maximum(normal, 1e-6), 0.0, cfg.slip_max)
```

三个技术点：

1. 分母用 `np.maximum(normal, 1e-6)` 而不是裸除法 —— 无接触时 `normal == 0`，裸除法会产生 `inf`/`nan` 并静默污染整个观测向量（nan 一旦进入 `VecNormalize` 的统计量就会永久污染训练）。
2. **截断位置选在源头**（`feet()` 里），而不是在 `foot_obs()` 或奖励项里。理由：同一份 `foot_slip` 会被观测、`as_dict()`、奖励项（平方）**多处读取**，在源头截断保证所有消费者看到同一个有界量。
3. `slip_max = 8.0` 而不是「理论最大值」：几以上就已经是完全滑动，取 8 留出余量。

**被否决的方案**：只在奖励项里 `clip` —— 否决：`foot_obs` 与 info 仍然会拿到未截断值，观测向量会被极端值主导（而观测是归一化的，一个 300 的离群值会毁掉统计量）。

### 3.6 `riser_graze` 用**接触系法向的 z 分量**判定，并加**力阈值**

```python
frame = np.asarray(c.frame, dtype=np.float64).reshape(3, 3)
if (abs(float(frame[0][2])) < cfg.riser_max_normal_z
        and fn > cfg.riser_min_force):
    riser[k] = True
```

依赖两条**已验证**的 MuJoCo 事实（源码 `Verified MuJoCo facts this file relies on`）：

* `data.contact[i].frame[0:3]` **就是**接触法向，且从 `geom[0]` 指向 `geom[1]`（`mjdata.h`：*"normal is in [0-2], points from geom[0] to geom[1]"*）—— 所以 `frame[0][2]` 是法向的**世界 z 分量**。
* `mj_contactForce(m, d, i, res)` 返回的 `res` 在**接触系**：`res[0]` 法向力、`res[1:3]` 两个切向（摩擦）分量、`res[3:6]` 力矩。源码用 `fn = abs(float(self._force6[0]))`（取绝对值以兼容法向的指向约定）、`ft = float(np.hypot(self._force6[1], self._force6[2]))`。

判定语义：`|n_z| < 0.60` 表示接触法向几乎水平 ⇒ 脚被**竖直面**侧推，而不是被踏面支撑；再要求 `fn > 5.0 N` 才算「真的在推」而不是擦到一下。两个阈值都可在 `TouchConfig` 里调。

**一并验证的物理恒等式**（源码 docstring，并被 `scripts/verifications/step11_touch.py` 当回归断言）：平地站立实测四足法向力 `28.94 / 29.56 / 43.94 / 46.73 N`，**和 149.18 N == 15.2064 kg 体重的 100.0 %**。这是「`mj_contactForce` 读出的确实是以牛顿为单位的真实接触力」这个前提的独立证据。

### 3.7 只统计「脚 geom ↔ 地形 geom」的接触对

`feet()` 遍历 `data.contact[0..ncon-1]` 时做两重过滤：先把 `geom1`/`geom2` 映射到脚索引 `k`（`self._foot_index`），再要求配对的另一个 geom 属于 `self.terrain_geoms`：

```python
self.terrain_geoms: set[int] = {int(meta.floor_geom_id)} | {
    int(g) for g in meta.stair_geom_ids
}
```

理由：排除（a）脚与自己的身体/另一只脚的接触（自碰撞、双腿交叉），（b）任何非地形接触 —— 否则「脚踢到自己小腿」会被当成承重或蹭立面。地面 (`floor_geom_id`) 与所有台阶 geom (`stair_geom_ids`) 都必须算进地形集合，否则平地站立时力读数为 0。

**一只脚可能同时有多个接触**（球面蹭两级踏面边缘）：`normal[k] += fn`、`tangential[k] += ft`，而 `riser[k]` 用 OR 合并（源码：`forces are summed and the riser flag is OR-ed`）。

**被否决的方案**：用 `data.cfrc_ext`（每个 body 的净外力）反推足端受力 —— 否决：它把「整条腿的动力学」混进来，含惯性项与关节力矩，无法区分「脚在承重」与「腿在加速」。

### 3.8 `probe_enabled` 让射线扇可以关掉

`__init__` 里的注释：

> The ray fan is the expensive half (~60 us per control step, roughly half a physics step).  The environment switches it off when neither the observation nor the `stair_probe_lift` reward reads it.

`robotdog/env.py:343-346` 正是这样传的：`probe_enabled=bool(self.cfg.obs.stair_probe or self.cfg.reward.stair_probe_lift != 0.0)`。关闭后 `read()` 走 `hit = False; rng = height_ahead = step = far_step = 0.0` 分支，`probe_range` 被置为 `probe_range_obs`（见 §3.9），接触力部分照常工作。

理由：不读测距的配方（例如只开 `touch` 观测）不该为 9+1 条 `mj_ray` 付 60 µs/控制步 —— 相对于约 120 µs 的物理步，那是**接近一半**的开销。

**被否决的方案**：让 `TouchSensor` 自己在每次 `read()` 时判断「有没有消费者」—— 否决：传感器不可能知道调用方把返回值怎么用；显式开关把决定权交给拥有配置的环境。

### 3.9 `probe_range`（base 相对）与 `ahead_of_sensor()`（传感器相对）是**两个量**

* `TouchReading.probe_range` 是**从 base 中心量**的水平距离 —— docstring 称它 `that is the number worth printing in a log`。
* `ahead_of_sensor(cfg)` = `max(probe_range - probe_forward, 0.0)`，未命中时返回 `inf` —— 观测与奖励需要传感器相对距离。

`probe_obs` 的 docstring 明确说明**为什么**不能用 base 相对距离做 proximity：

> Measured **from the cast origin**, not from the base: the sensor sits `probe_forward` in front of the torso, so a base-relative distance could never fall below that offset and the term would saturate at ~0.45.

此外，`read()` 在**未命中**时把 `probe_range` 重置成 `probe_range_obs`（`1.20`）：

```python
if not hit:
    rng = self.cfg.probe_range_obs
```

理由：让「没有台阶在视野里」与「台阶在量程边缘」映射到同一个 `probe_range`，从而 `probe_obs` 的 proximity 项自然为 0（`probe_hit` 为 `False` 时 `prox` 直接取 `0.0`），而不需要另一条 `nan`/`inf` 分支。**被否决的方案**：未命中时返回 `inf` —— 否决：`inf` 进任何归一化都会变成 `nan`。

### 3.10 `trigger()` 是两个被裁剪斜坡的**乘积**（构造性平滑）

```python
if not self.probe_hit:
    return 0.0
rise = float(np.clip(self.probe_step / cfg.probe_step_ref, 0.0, 1.0))
prox = float(np.clip(1.0 - self.ahead_of_sensor(cfg) / cfg.probe_trig_dist, 0.0, 1.0))
return rise * prox
```

docstring：`Smooth by construction (a product of two clipped ramps) so that the reward keeps a usable gradient instead of switching on and off.`

理由：这是一个**门控**（乘到 `stair_probe_lift` 奖励上），如果用布尔 `and`（`hit and near and tall`），奖励会在 0 与满值之间硬跳变，PPO 拿不到梯度；两个连续斜坡的乘积在边界处是连续的。注意两个斜坡用**不同的归一化尺度**：`rise` 用 `probe_step_ref = 0.030`（拿到满分所需的抬升，很小 ⇒ 3 cm 就够），`prox` 用 `probe_trig_dist = 0.62`（触发距离）。

### 3.11 `TouchReading` 用 `field(default_factory=...)` 而不是可变默认值

所有数组字段都是 `field(default_factory=lambda: np.zeros(4))`（布尔的是 `np.zeros(4, dtype=bool)`）。理由：dataclass 的可变默认值会**在所有实例间共享**（一个实例改写会污染另一个）；`default_factory` 每次构造新数组。这条不是风格问题 —— 共享数组会让「上一回合的力读数」泄漏进下一回合。

`empty_reading()` 就是靠全部默认值构造一个「什么都没碰到、前方什么都没有」的读数：docstring `Used before the first reset so that _obs_frame is well defined.` —— 环境在 `reset()` 之前就要能拼出观测，否则第一步的观测维度/数值未定义。

### 3.12 `read()` 的参数是 **keyword-only**

```python
def read(self, *, base_pos: np.ndarray, yaw: float, terrain_height: float) -> TouchReading:
```

理由：三个参数都是 `float`/数组，位置传参极容易把 `yaw` 与 `terrain_height` 写反（两者都是浮点数，类型检查器无法拦截）。强制关键字传参把这类错误变成 `TypeError`。`TouchSensor.__init__` 的 `probe_enabled` 也是 keyword-only（`*,`），同理：它是一个**行为开关**，不该被误当成 `cfg`。

## 4. 公开 API

### 模块级常量

```python
TERRAIN_GEOM_MASK = np.array(
    [1 if g == TERRAIN_GEOM_GROUP else 0 for g in range(6)], dtype=np.uint8
)
```

实测值：`array([0, 1, 0, 0, 0, 0], dtype=uint8)`。用途：`mj_ray` 的 `geomgroup` 参数（只开地形组）。

### `@dataclass class TouchConfig`

| 字段 | 类型 | 默认值 | 源码注释（单位/语义原文） |
| --- | --- | --- | --- |
| `probe_forward` | `float` | `0.34` | `m, first ray, ahead of the base centre` |
| `probe_span` | `float` | `0.60` | `m, how far the fan reaches past the first ray` |
| `probe_rays` | `int` | `9` | `>= 2; spacing = probe_span/(probe_rays-1)` |
| `probe_ray_top` | `float` | `1.00` | `m, rays start this high and shoot down` |
| `probe_rise_eps` | `float` | `0.015` | `m, rise smaller than this is "still flat"` |
| `probe_horizontal_height` | `float` | `0.045` | `m, side ray that pinpoints a riser` |
| `probe_step_clip` | `float` | `0.24` | `m, clamp on the reported rise` |
| `probe_step_ref` | `float` | `0.030` | `m, rise that earns full trigger credit` |
| `probe_trig_dist` | `float` | `0.62` | `m, trigger ramps up inside this distance` |
| `probe_range_obs` | `float` | `1.20` | `m, distance normalisation for the obs` |
| `force_norm_ref` | `float` | `75.0` | `N, per-foot force that saturates the obs` |
| `impact_force_target` | `float` | `95.0` | `N, hard-landing threshold` |
| `slip_safe` | `float` | `0.55` | `tangential/normal ratio that costs nothing` |
| `slip_loaded_force` | `float` | `10.0` | `N, below this a foot is not "loaded"` |
| `slip_max` | `float` | `8.0` | `cap on the tangential/normal ratio (see feet())` |
| `riser_max_normal_z` | `float` | `0.60` | `\|n_z\| below this == a vertical face` |
| `riser_min_force` | `float` | `5.0` | `N, contact must push this hard` |

**注意**：`impact_force_target` 与 `slip_safe` 在**本模块内部没有任何读取点**（见 §9），实际生效的是 `robotdog/env.py` 的 `RewardConfig.touch_impact_target = 95.0` / `RewardConfig.touch_slip_safe = 0.55`。

### `@dataclass class TouchReading`

字段（全部带默认值）：

| 字段 | 类型 | 默认值 |
| --- | --- | --- |
| `foot_normal_force` | `np.ndarray` | `field(default_factory=lambda: np.zeros(4))` |
| `foot_tangential_force` | `np.ndarray` | `field(default_factory=lambda: np.zeros(4))` |
| `foot_slip` | `np.ndarray` | `field(default_factory=lambda: np.zeros(4))` |
| `foot_contact` | `np.ndarray` | `field(default_factory=lambda: np.zeros(4, dtype=bool))` |
| `riser_graze` | `np.ndarray` | `field(default_factory=lambda: np.zeros(4, dtype=bool))` |
| `probe_hit` | `bool` | `False` |
| `probe_range` | `float` | `0.0`（`horizontal distance to the step (m)`） |
| `probe_height_ahead` | `float` | `0.0`（`terrain height at the first rising ray (m)`） |
| `probe_step` | `float` | `0.0`（`rise of the NEAREST step (m, clipped)`） |
| `probe_slope` | `float` | `0.0`（`rise / horizontal distance ahead`） |
| `probe_far_step` | `float` | `0.0`（`tallest rise anywhere in the fan (m, clipped)`） |

方法/属性（签名逐字）：

```python
@property
def total_normal_force(self) -> float:
    return float(self.foot_normal_force.sum())


def loaded(self, cfg: TouchConfig) -> np.ndarray:
    """Feet that currently carry weight (used to gate the slip penalty)."""
    return self.foot_normal_force > cfg.slip_loaded_force


def riser_graze_any(self) -> bool:
    return bool(self.riser_graze.any())


def foot_obs(self, cfg: TouchConfig) -> np.ndarray:
    """9 values: 4 normalised forces, 4 slip ratios, 1 riser flag."""


def probe_obs(self, cfg: TouchConfig) -> np.ndarray:
    """3 values: normalised rise, proximity, hit flag."""


def probe_obs_full(self, cfg: TouchConfig) -> np.ndarray:
    """5 values: the 3 above plus two quantities that were already computed."""


def ahead_of_sensor(self, cfg: TouchConfig) -> float:
    """Distance from the rangefinder to the obstacle (m)."""


def trigger(self, cfg: TouchConfig) -> float:
    """The "a step is close AND tall" gate the lift reward multiplies."""


def as_dict(self) -> dict[str, float]: ...
```

逐项语义：

| 成员 | 返回 | 语义与边界 |
| --- | --- | --- |
| `total_normal_force` | `float` | `foot_normal_force.sum()` 转 `float`（平地站立恒等式 149.18 N 用的就是它） |
| `loaded(cfg)` | `np.ndarray`（bool，形状 `(4,)`） | `foot_normal_force > cfg.slip_loaded_force`（默认 `> 10.0 N`）；用于给滑移惩罚做门控 |
| `riser_graze_any()` | `bool` | `riser_graze.any()` |
| `foot_obs(cfg)` | `np.ndarray`，**9 维** | `[clip(force/75, 0, 1) ×4, clip(slip/2.0, 0, 1) ×4, 1 if riser_graze.any() else 0]`。注意滑移的归一化分母是**硬编码的 `2.0`**，与 `slip_max = 8.0` 不是一个数 |
| `probe_obs(cfg)` | `np.ndarray`，**3 维** | `[clip(probe_step/probe_step_clip, 0, 1), prox, 1.0 if probe_hit else 0.0]`；`prox = 1 - clip(ahead_of_sensor/probe_range_obs, 0, 1)`，未命中时 `0.0` |
| `probe_obs_full(cfg)` | `np.ndarray`，**5 维** | `probe_obs(cfg)` 之后接 `[clip(probe_slope/5.0, 0, 1), clip(probe_far_step/probe_step_clip, 0, 1)]` |
| `ahead_of_sensor(cfg)` | `float` | 未命中 → `float("inf")`；否则 `max(probe_range - probe_forward, 0.0)` |
| `trigger(cfg)` | `float`（`0.0`~`1.0`） | 未命中 → `0.0`；否则 `clip(probe_step/probe_step_ref, 0, 1) * clip(1 - ahead_of_sensor/probe_trig_dist, 0, 1)` |
| `as_dict()` | `dict[str, float]` | 固定键：`foot_force_sum`、`riser_graze`、`probe_hit`、`probe_range`、`probe_step`、`probe_slope`、`probe_far_step`，外加 `foot_force_{0..3}` 与 `foot_slip_{0..3}`（共 **15** 个键） |

### `cast_ray(...)`

```python
def cast_ray(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    geomgroup: np.ndarray | None = TERRAIN_GEOM_MASK,
    max_range: float | None = None,
) -> tuple[float, int]:
    """Cast one ray; return ``(distance, geom_id)``.

    ``distance`` is ``inf`` and ``geom_id`` is ``-1`` when nothing is hit (or
    when the hit is farther than ``max_range``).  ``direction`` is normalised
    here, so the returned distance is always in metres.
    """
```

| 参数 | 语义 |
| --- | --- |
| `origin` | 射线原点，任意可转成 3 元素的数组（内部 `reshape(3)`） |
| `direction` | 射线方向（**内部归一化**，所以传出距离恒为米） |
| `geomgroup` | keyword-only，默认 `TERRAIN_GEOM_MASK`（只开地形组）；传 `None` 则不过滤 |
| `max_range` | keyword-only，默认 `None`；命中距离 > `max_range` 时**也算未命中** |

* 返回 `(distance, geom_id)`；未命中 / 方向为零向量 / 超出 `max_range` → `(float("inf"), -1)`。方向范数 `<= 1e-12` 时直接返回，不调用 MuJoCo。
* 内部固定传 `flg_static=1`、`bodyexclude=-1`，`geomid` 是长度 1 的 `np.int32` 出参数组。
* 无显式 `raise`。

### `class TouchSensor`

```python
class TouchSensor:
    """Reads contact forces and rangefinder distances out of one MuJoCo data.

    Held by :class:`robotdog.env.Go2StairsEnv` (one instance per environment).
    """

    def __init__(self, model, data, meta, cfg: TouchConfig | None = None,
                 *, probe_enabled: bool = True) -> None:
        self.model = model
        self.data = data
        self.meta = meta
        self.cfg = cfg if cfg is not None else TouchConfig()
        self.probe_enabled = bool(probe_enabled)
        self.terrain_geoms: set[int] = {int(meta.floor_geom_id)} | {
            int(g) for g in meta.stair_geom_ids
        }
        self._foot_index = {
            int(g): k for k, g in enumerate(meta.foot_geom_ids)
        }
        self._force6 = np.zeros(6, dtype=np.float64)

    def feet(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    def forward_probe(
        self, base_pos: np.ndarray, yaw: float, terrain_height: float
    ) -> tuple[bool, float, float, float, float]: ...
    def read(self, *, base_pos: np.ndarray, yaw: float,
             terrain_height: float) -> TouchReading: ...
```

| 成员 | 参数 | 返回 | 语义 |
| --- | --- | --- | --- |
| `__init__` | `model`、`data`（MuJoCo 对象，**不做类型注解**）、`meta`（任何带 `floor_geom_id` / `stair_geom_ids` / `foot_geom_ids` 的对象）、`cfg`（`None` ⇒ `TouchConfig()`）、`probe_enabled`（keyword-only，默认 `True`） | `None` | 只存引用与索引表，**不做任何计算**。两只属性 `terrain_geoms`（set）、`_foot_index`（dict）是全部预处理 |
| `feet()` | 无 | `(normal, tangential, slip, riser_graze)`，四个 `np.ndarray`，`normal`/`tangential`/`slip` 形状 `(4,)` float，`riser_graze` 形状 `(4,)` bool | 遍历 `data.ncon` 个接触，只保留「脚 geom ↔ 地形 geom」对；力相加、riser 标志 OR；`slip = clip(tangential / max(normal, 1e-6), 0, slip_max)`。**内部还有一个 `contact` 布尔数组，但不返回** |
| `forward_probe(base_pos, yaw, terrain_height)` | `base_pos`：世界坐标（只读 `[0]`、`[1]`）；`yaw`：朝向弧度（只用 `cos`/`sin`）；`terrain_height`：base 脚下的地形高 | `(hit, range, height, step, far_step)`：`bool, float, float, float, float` | 竖直射线扇（`max(probe_rays, 2)` 条，间距 `span/(n-1)`）+ 平射锐化；`step`/`far_step` 被 `probe_step_clip` 裁剪；未命中时 `range == probe_range_obs`、`height == terrain_height`、`step == far_step == 0.0` |
| `read(*, base_pos, yaw, terrain_height)` | 全部 keyword-only | `TouchReading` | `feet()` + （`probe_enabled` 时）`forward_probe()`；未命中时把 `rng` 置为 `probe_range_obs`；`foot_contact = normal > 0.0`（**按力重算，不是 `feet()` 里的局部 `contact`**）；`probe_slope = clip(step / max(rng - probe_forward, 1e-6), 0, 5.0)`（未命中时 `0.0`） |

* 异常：`feet()` / `forward_probe()` / `read()` 都不显式 `raise`；`meta` 缺属性会抛 `AttributeError`（鸭子类型没有防御）。
* `cast_ray` 与 `TouchSensor` 使用模块级 `TERRAIN_GEOM_MASK` 作为默认掩码，因此**探针永远看不到机器人**（除非调用方显式传 `geomgroup=None`）。

### `empty_reading()`

```python
def empty_reading() -> TouchReading:
    """A "nothing touched, nothing ahead" reading.

    Used before the first :meth:`reset` so that ``_obs_frame`` is well defined.
    """
    return TouchReading()
```

返回全默认值的 `TouchReading`（力全 0、`probe_hit=False`、`probe_range=0.0` 等）。

## 5. 内部结构与数据流

每个控制步的读取（环境内的一条线）：

```
robotdog/env.py:1064 / :1145
  → TouchSensor.read(base_pos=..., yaw=..., terrain_height=...)
      → TouchSensor.feet()
          → for i in range(self.data.ncon)
              → self.data.contact[i]          → (geom1, geom2) → self._foot_index
              → 过滤 other ∈ self.terrain_geoms
              → mujoco.mj_contactForce(self.model, self.data, i, self._force6)
              → normal[k] += abs(res[0]) ; tangential[k] += hypot(res[1], res[2])
              → frame[0][2] 与 fn 判定 riser_graze[k]
          → slip = clip(tangential / max(normal, 1e-6), 0, slip_max)
          → return (normal, tangential, slip, riser)
      → （if self.probe_enabled）TouchSensor.forward_probe(base_pos, yaw, terrain_height)
          → 对 k in range(max(probe_rays, 2)):
              → cast_ray(self.model, self.data, origin=[x0+c*d, y0+s*d, top], [0,0,-1])
                  → mujoco.mj_ray(m, d, pnt, vec, TERRAIN_GEOM_MASK, 1, -1, geomid)
              → rise = (top - dist) - terrain_height
              → 记录 far_step / 首次 rise > probe_rise_eps 的 (d - spacing/2)
          → 平射锐化：cast_ray(..., origin=[x0+c*pf, y0+s*pf, th+probe_horizontal_height], [c,s,0])
              → range_m = min(range_m, probe_forward + flat_dist)
          → clip(step) / clip(far_step)
      → TouchReading(..., foot_contact = normal > 0.0,
                       probe_slope = clip(step / max(rng - probe_forward, 1e-6), 0, 5))
  → env._touch = <TouchReading>
```

读数到观测（环境层，`env.py:582-586`）：

```
env._touch.foot_obs(cfg.touch)          → 9 维（若 obs.touch）
env._touch.probe_obs_full(cfg.touch)    → 5 维（若 obs.stair_probe and obs.stair_probe_full）
env._touch.probe_obs(cfg.touch)         → 3 维（若 obs.stair_probe 且未开 full）
```

读数到奖励（环境层，`env.py:724-790`）：

```
touch.riser_graze.any() / touch.riser_graze + foot vz      → terms["riser_graze"]      （rc.riser_graze）
touch.foot_normal_force, touch.foot_tangential_force        → terms["touch_impact"]     （rc.touch_impact）
touch.foot_normal_force > cfg.touch.slip_loaded_force,
touch.foot_slip 与 rc.touch_slip_safe                       → terms["touch_slip"]       （rc.touch_slip）
touch.trigger(self.cfg.touch)                               → terms["stair_probe_lift"] （rc.stair_probe_lift）
```

读数到日志（`env.py:1208-1215`）：`probe_hit`、`probe_step`、`probe_range`、`probe_trigger`、`riser_graze`、`total_normal_force`、`max_slip`；`TouchReading.as_dict()` 也可打包 15 个键。

生命周期：

```
Go2StairsEnv.__init__
  → env.py:343  self.touch_sensor = TouchSensor(self.model, self.data, self.meta,
                                                 cfg=self.cfg.touch,
                                                 probe_enabled=bool(self.cfg.obs.stair_probe
                                                                    or self.cfg.reward.stair_probe_lift != 0.0))
  → env.py:348  self._touch: TouchReading = empty_reading()
set_stair() → env.py:363-372 _build_model()  ⇒ 新 model/data ⇒ 传感器重建（引用必须换新）
```

## 6. 关键常量与默认值

| 项 | 真实值 |
| --- | --- |
| `TERRAIN_GEOM_MASK` | `[0, 1, 0, 0, 0, 0]`（`dtype=uint8`，长度 6；由 `TERRAIN_GEOM_GROUP = 1` 派生） |
| `TouchConfig.probe_forward` | `0.34` m |
| `TouchConfig.probe_span` | `0.60` m |
| `TouchConfig.probe_rays` | `9`（约束 `>= 2`；扇间距 = `0.60/8 = 0.075` m） |
| `TouchConfig.probe_ray_top` | `1.00` m |
| `TouchConfig.probe_rise_eps` | `0.015` m |
| `TouchConfig.probe_horizontal_height` | `0.045` m |
| `TouchConfig.probe_step_clip` | `0.24` m |
| `TouchConfig.probe_step_ref` | `0.030` m |
| `TouchConfig.probe_trig_dist` | `0.62` m |
| `TouchConfig.probe_range_obs` | `1.20` m |
| `TouchConfig.force_norm_ref` | `75.0` N |
| `TouchConfig.impact_force_target` | `95.0` N（本模块内无读取点） |
| `TouchConfig.slip_safe` | `0.55`（本模块内无读取点） |
| `TouchConfig.slip_loaded_force` | `10.0` N |
| `TouchConfig.slip_max` | `8.0` |
| `TouchConfig.riser_max_normal_z` | `0.60` |
| `TouchConfig.riser_min_force` | `5.0` N |
| `foot_obs` 的力归一化 | `force_norm_ref = 75.0` |
| `foot_obs` 的滑移归一化 | **硬编码 `2.0`**（源码 `np.clip(self.foot_slip / 2.0, 0.0, 1.0)`） |
| `probe_obs_full` 的坡度归一化 | **硬编码 `5.0`**（`np.clip(self.probe_slope / 5.0, 0.0, 1.0)`） |
| `read()` 的 `probe_slope` 上限 | `5.0`（`np.clip(step / ahead, 0.0, 5.0)`） |
| 脚的数量 | `n = 4`（`feet()` 内部硬编码） |
| 滑移分母下限 | `1e-6`（`np.maximum(normal, 1e-6)`） |
| `cast_ray` 的零向量阈值 | `1e-12`（`if norm <= 1e-12: return float("inf"), -1`） |
| `cast_ray` 的默认掩码 | `TERRAIN_GEOM_MASK` |
| `cast_ray` 的 `flg_static` / `bodyexclude` | `1` / `-1` |
| `TouchSensor` 的 `probe_enabled` 默认 | `True` |
| `_force6` 缓冲 | `np.zeros(6, dtype=np.float64)`（复用，避免每接触分配） |
| `as_dict()` 的键数 | `15`（7 个标量 + 4 个 `foot_force_{k}` + 4 个 `foot_slip_{k}`） |
| 射线扇开销（源码注释） | `~60 us per control step, roughly half a physics step` |
| 平地站立法向力（源码注释） | `28.94 / 29.56 / 43.94 / 46.73 N`，和 `149.18 N == 100.0 %` of `15.2064 kg` |
| 计算值：`probe_obs` 的 proximity 上限 | `1 - 0.34/1.20 ≈ 0.7167`（见 §9 的文档不一致） |
| 计算值：`trigger()` 的 prox 上限 | `1 - 0.34/0.62 ≈ 0.4516` |

## 7. 与其它模块的关系

**上游**：第三方 `mujoco`、`numpy`；stdlib `dataclasses`；以及工具层内部唯一一条依赖 —— `robotdog/tools/terrain.py` 的 `TERRAIN_GEOM_GROUP`。

**下游**：

| 位置 | 使用的符号 | 依赖方向为什么是这样 |
| --- | --- | --- |
| `robotdog/env.py:44-48` | `TouchConfig, TouchReading, TouchSensor, empty_reading` | 环境层拥有 `model`/`data`，所以它创建传感器并把读数接进观测/奖励 |
| `robotdog/env.py:343-348` | 构造 + `empty_reading()` | 模型编译后立刻建传感器；`reset()` 之前就要有合法读数 |
| `robotdog/env.py:582-586` | `foot_obs` / `probe_obs` / `probe_obs_full` | 观测块的**内容**由工具层定义，**拼接与顺序**由环境层决定（`ObsConfig.stair_probe` / `stair_probe_full` 控制维度） |
| `robotdog/env.py:724-790` | `riser_graze`、`foot_normal_force`、`foot_slip`、`trigger`、`slip_loaded_force` | 权重与门控属于奖励设计（环境层），量属于工具层 |
| `robotdog/env.py:1208-1215` | `probe_hit/step/range`、`trigger`、`total_normal_force`、`foot_slip` | info/日志 |
| `robotdog/training.py:357, 400-402` | `TouchConfig`、`{f.name for f in _fields(TouchConfig)}` | 从 `summary.json` 恢复配置并按字段名白名单过滤（未知键不会污染 dataclass） |

**为什么方向是「环境层 → touch 工具层」**：本模块需要 `model`/`data`/`meta` 才能工作，而这些是环境层编译产物；反过来本模块不 import `env` 也不 import `go2_model`（**只鸭子类型读 `meta` 的三个属性**），于是工具层不必知道 `Go2Meta` 的类型定义，也不会与 `go2_model.py` 成环。这是「宽进（任何带那三个属性的对象）严出（只返回数组/浮点）」的设计。

**与 `terrain.py` 的关系**：`TERRAIN_GEOM_GROUP` 是地形几何的组号，本模块据此构造射线掩码。两者是**工具层内部的单向依赖**（`touch → terrain`），也是 `robotdog/tools/__init__.py` 里字母序恰好等于加载序的原因。

## 8. 实测与经验教训

* 地形 geomgroup 默认隐藏（射线掩码与渲染可见性同源）见 `experience.md` §1.2。
* 「蹭立面惩罚与任务方向打架」——`riser_graze` 作为惩罚项时的副作用，见 `experience.md` §2.5。
* 「落地力阈值是常数 ⇒ 高台阶上从提醒变成乱罚」，见 `experience.md` §2.6。
* 探针读数与独立评测不一致（`probe()` 与评估对不上），见 `experience.md` §5.1。
* 小样本评估的 ±15 pt 错觉会掩盖这里所有阈值的真实效果，见 `experience.md` §5.7。
* 观测块维度（`touch` 9 维 / `stair_probe` 3 维 / `stair_probe_full` +2）与相应奖励权重在训练配方中的位置，见 `train.md` §1（本文档不重复其结论）。

## 9. 已知限制与坑

* **`TouchConfig.impact_force_target` 与 `TouchConfig.slip_safe` 从未被读取**。全仓只有它们的定义处（`robotdog/tools/touch.py:82-83`）。实际生效的是**环境层里另写的一份同值常量**：`robotdog/env.py:161` 的 `RewardConfig.touch_impact_target: float = 95.0` 与 `robotdog/env.py:164` 的 `RewardConfig.touch_slip_safe: float = 0.55`。改 `TouchConfig` 里的这两个字段**不会有任何效果**。
* **`probe_obs` 的 docstring 数值与常量对不上**。它写 `the term would saturate at ~0.45`，但按 `probe_range_obs = 1.20`、`probe_forward = 0.34` 计算，base 相对距离导致的饱和值应是 `1 - 0.34/1.20 ≈ 0.7167`；`~0.45` 对应的是 `trigger()` 用的 `probe_trig_dist = 0.62`（`1 - 0.34/0.62 ≈ 0.4516`）。即那句注释里的比例尺来自另一个常量。
* **`TouchReading.probe_hit` 上方残留了一条错位注释**：`#: a downhill-facing ray normal at the contact (0 = flat tread, 1|numeric)` —— 它描述的既不是 `probe_hit`（bool），也不对应本类任何字段，是从早期设计残留下来的。**不要按它理解字段语义**。
* **`foot_obs` 的滑移归一化写死 `/2.0`，与 `slip_max = 8.0` 不一致**：`foot_slip` 截断到 8，但观测里 2.0 就饱和。这是刻意的（2 以上都属于「完全滑动」），但改动 `slip_max` 不会自动改变观测的饱和点。
* **`feet()` 内部算出的 `contact` 布尔数组不被返回**，`TouchReading.foot_contact` 是 `read()` 里用 `normal > 0.0` **重算**的。若将来想用「有力但符号为负」之类的情形，两边会分叉。
* **`meta` 是鸭子类型、没有防御**：传入缺 `floor_geom_id` / `stair_geom_ids` / `foot_geom_ids` 的对象会在 `__init__` 抛 `AttributeError`；属性名拼错不会在类型检查阶段被发现。
* **`model` / `data` 的类型注解缺失**（`__init__` 的三个参数无注解），IDE 与类型检查器无法帮忙。
* **必须重建传感器才能换地形**：`set_stair()` 会重新编译 `model`/`data`（`env.py:363-372`），旧的 `TouchSensor` 内部持有的 `model`/`data` 引用会指向**已废弃**的对象。环境层每次都重建（`env.py:343` 在 `_build_model()` 内），但直接复用旧实例会读到过期数据。
* **`probe_rays < 2` 会被静默抬到 2**（`n = max(int(cfg.probe_rays), 2)`），并把间距改成 `span/(n-1)`；`probe_span < 0` 被 `max(..., 0.0)` 钳到 0（此时所有射线都在同一点）。
* **`probe_step_clip = 0.24` 会掩盖真实台阶高度**：`probe_step` 与 `probe_far_step` 都被裁剪到 0.24 m，所以「24 cm 台阶」与「40 cm 墙」在这里读数相同 —— 区分二者要靠 `probe_slope`（`probe_obs_full` 才带）。
* **探针只看机器人**正前方**（`yaw`）**，不扫描侧向；侧向台阶与窄楼梯的边缘不在观测里。
* **`forward_probe` 只用 `base_pos[0]`、`base_pos[1]` 与 `yaw`**：`base_pos[2]` 被忽略，地形高由调用方通过 `terrain_height` 提供 —— 传错这个值会让射线起点整体偏移 `probe_ray_top` 对应的量（读数系统性偏差）。
* **未命中时 `probe_range` 被写成 `probe_range_obs`（1.20）而不是 0 或 `inf`**：日志里 `probe_range == 1.20` 既可能是「量程边缘的台阶」也可能是「什么都没看到」，要靠 `probe_hit` 区分。
* **`trigger()` 在 `probe_hit` 为假时返回 `0.0`**：`stair_probe_lift` 奖励因此在平地/无台阶场景恒为 0（符合预期），但也意味着「下坡」永远拿不到这个奖励。
* **源码引用的回归脚本不在仓内**：`touch.py:33` 与 `terrain.py:53` 都引用了 `scripts/verifications/step11_touch.py`（平地站立法向力恒等式与「地形必须渲染可见」的断言），但当前快照里 `glob scripts/verifications/*.py` **返回 no files found** —— 那些断言目前没有可执行的载体。
* **`~60 us per control step` 的开销是注释里给的量级**，不是运行时可测的常量；关掉 `probe_enabled` 是唯一可用的开关，本模块不做自适应降频。
* **接触力只在「脚 ↔ 地形」对上统计**：机器人踩到非地形刚体（未来若加入可动障碍）时读数为 0。
* **`cast_ray` 传 `geomgroup=None` 会打到机器人自己**（默认掩码正是为了避免这一点）；`terrain.mujoco_mj_ray` 就是这种不过滤的用法，两者语义不同，不要互相替换。

## 10. 示例用法

```python
import numpy as np

from robotdog.tools.touch import (
    TERRAIN_GEOM_MASK,
    TouchConfig,
    TouchReading,
    TouchSensor,
    cast_ray,
    empty_reading,
)

# 1) 纯数据处理（不需要 MuJoCo）：阈值、观测块、触发门控
cfg = TouchConfig()
r = TouchReading(
    foot_normal_force=np.array([30.0, 40.0, 0.0, 0.0]),
    foot_tangential_force=np.array([5.0, 60.0, 0.0, 0.0]),
    foot_slip=np.array([0.1, 5.0, 0.0, 0.0]),
    riser_graze=np.array([False, True, False, False]),
    probe_hit=True,
    probe_range=0.70,
    probe_height_ahead=0.20,
    probe_step=0.06,
    probe_slope=1.5,
    probe_far_step=0.12,
)
print(r.total_normal_force)            # 70.0
print(r.loaded(cfg))                   # [True True False False]  (> slip_loaded_force=10 N)
print(r.riser_graze_any())             # True
print(r.foot_obs(cfg).shape)           # (9,)
print(r.probe_obs(cfg).shape)          # (3,)
print(r.probe_obs_full(cfg).shape)     # (5,)
print(r.ahead_of_sensor(cfg))          # 0.35999999999999993  (0.70 - probe_forward=0.34)
print(r.trigger(cfg))                  # 1.0 * (1 - 0.36/0.62) ≈ 0.4194
print(len(r.as_dict()))                # 15

# 2) 首次 reset 之前的占位读数
print(empty_reading().probe_hit, empty_reading().total_normal_force)   # False 0.0

# 3) 与已编译的模型配合（model/data/meta 由环境层提供）
#    sensor = TouchSensor(model, data, meta, cfg=TouchConfig(), probe_enabled=True)
#    print(sensor.terrain_geoms, sensor._foot_index)
#    reading = sensor.read(base_pos=env.base_pos, yaw=env.yaw,
#                          terrain_height=env.terrain_height_under_base)
#    print(reading.foot_obs(env.cfg.touch))

# 4) 单独投一条射线（默认只开地形组，看不到机器人）
#    dist, geom_id = cast_ray(model, data, origin=np.array([2.0, 0.0, 1.0]),
#                             direction=np.array([0.0, 0.0, -1.0]),
#                             geomgroup=TERRAIN_GEOM_MASK, max_range=2.0)
#    print(dist, geom_id)
print(TERRAIN_GEOM_MASK)               # [0 1 0 0 0 0]
```

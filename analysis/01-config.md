# 01 · 配置入口：`robotdog/config.py` + `config.yaml`

审计对象：`robotdog/config.py`（**443 行**）、`config.yaml`（**184 行**）、`robotdog/curriculum.py`（阶段表生成部分见 `04-training-scripts.md`）。

`config.py` 的自我描述（模块 docstring）是：「读取 `config.yaml`，把它变成 `EnvConfig` 与课程阶段表……**未知的键（打错的字段名）会报错并列出现有字段**」。下面 C-1 / C-2 说明这条承诺**只对一半的配置段生效**。

---

## C-1（blocker）`ppo:` / `run:` / `play:` 三段零键校验（`curriculum:` 顺带，见下）

### 现状

只有三段走 `_apply()`，因而才有未知键检查：

```
config.py  config_to_env():  _apply(cfg, env, where="env")            ← env: 段校验
                             _apply(cfg.reward, ...)                  ← reward: 段校验
                             _apply(cfg.command, ..., where="env.command")   ← env.command 校验
                             _apply(cfg.obs, ..., where="env.obs")           ← env.obs 校验
```

另外三段**直接返回原字典**，或只做 per-key `setdefault`：

```python
def config_to_ppo(raw):
    # docstring: 「键的合法性由 build_ppo 校验」
    return _section(raw, "ppo")          # ← 零校验
def config_to_run(raw):
    out = _section(raw, "run")           # ← 零校验
    out.setdefault("name", "course1"); out.setdefault("seed", 0); ...
def config_to_play(raw):
    out = _section(raw, "play")          # ← 零校验
    out.setdefault("episodes_gif", 10); ...
```

`curriculum:` 段介于两者之间：`config_to_stages()` 会对**它认识的键**做逐项语义校验（`heights_cm` 必须非负升序、长度必须匹配、`reward_overrides` 的键要过 `_validate_reward_keys`），但**拼错的键名仍然被静默忽略**（例如 `heights_cm` 写成 `height_cm` ⇒ 直接退回 `STAGE_HEIGHTS_CM` 兜底，一声不吭）。所以严格的说法是「三段零校验 + 一段半校验」。

### 实测证据（`.tmp/zz_audit_probe.py`）

```text
=== 5. config_to_ppo on a typo'd dict (F1 演示) ===
config_to_ppo 返回值里是否含拼错键: True -> 999.0
build_ppo 是否接受 **kwargs: False
config_to_run 里 seedd 是否出现: True
config_to_play 里 episodes_giff 是否出现: True
```

`config.yaml` 的 `ppo:` 段写 `ent_coeff: 999` 时：

1. `config_to_ppo()` 原样带回 `ent_coeff`，**不报错**；
2. 该键随后被彻底忽略 —— `build_ppo()` 的签名是显式关键字参数、**没有 `**kwargs`**，调用点是 `build_ppo(venv, seed=..., n_steps=ppo.get("n_steps", 256), ..., ent_coef=ppo.get("ent_coef", 0.002), ...)`，所以 `ent_coeff` 既不进 PPO、也不报错。

同理：`run:` 段 `seedd: 7`、`play:` 段 `episodes_giff: 3` 都会静默通过并静默失效。

### 为什么这是 blocker

本仓的整个工作流（`agent.md §4`）是「**超参一律进 `config.yaml`，没有别的开关**」。既然配置文件是唯一入口，那么**配置键的拼写正确性就是唯一的用户输入校验面**。当前状态下：

* 用户想调 `ent_coef`，写成 `ent_coeff`（多打一个 f）→ 训练照跑、日志照出、**系数没变**。这类错误最容易被解释成「这个超参没用」，从而得出错误的实验结论。
* `config.py` 自己的 docstring 与 `docs/robotdog/config.md:41` 都拿「`forward_vel_sigma` 写成 `forward_vel_sigme`」当**设计动机**举例 —— 而这个例子恰好落在**会**报错的 `reward:` 段，给人「整份配置都会报错」的错觉。

### 建议

在 `config_to_ppo` / `config_to_run` / `config_to_play` / `config_to_stages` 里复用同一套检查。`build_ppo` 已经是权威的键表来源，可以直接由签名生成白名单，避免再维护一份列表：

```python
# 伪代码
_VALID_PPO = set(inspect.signature(build_ppo).parameters) - {"venv", "tensorboard_log", "verbose"}
_VALID_RUN = {"name", "seed", "save_every", "log_every"}
_VALID_PLAY = {"episodes_gif", "episodes_eval", "seed", "camera", "width", "height",
               "fps", "scale", "stages", "batch", "stage_height_cm", "device"}
_VALID_CURRICULUM = {"tread", "n_up", "heights_cm", "timesteps", "learning_rate",
                     "command_vy_max", "command_yaw_max", "command_vx",
                     "flat_success_distance", "flat_success_needs_progress",
                     "reward_overrides"}
```

并统一报错文案（与 `_apply()` 完全一致：`config.yaml 的 {where}: 无法识别的键 {sorted(unknown)}；可用键 = {sorted(valid)}`）。注意 `play.device` 目前是**被读取但不在任何白名单里**的键（见 S-5），加白名单时要一并处理。

### 已验证无问题

* `env:` / `reward:` / `env.obs` / `env.command` 四处的未知键检查**确实生效**，且报错文本会列出可用键。
* `_section()` 返回 `dict(value)` 副本，所以 `config_to_env()` 里的 `env.pop("obs"/"command"/"n_envs"/"gait_file")` **不会改动调用方的字典** —— 实测连续两次 `config_to_env(raw)` 得到**完全相同**的 `gait.freq`，`raw["env"]` 的键集合在调用前后不变。`env.py`/`play.py`/`train.py` 里同一份 `cfg` 被复用是安全的。
* `config_to_env()` 在 `env:` 段里显式 `pop` 的四个键（`obs`、`command`、`n_envs`、`gait_file`）确实是「先摘出来单独处理」，不是漏校验。

---

## C-2（high）同一份配置两种人格：`env:`/`reward:` 硬报错，`ppo:` 静默丢弃

严格说这是 C-1 的一个切片，但值得单列，因为**它的误导性最强**：`docs/robotdog/training.md` 把 PPO 键表写得很全（`n_steps` / `batch_size` / `n_epochs` / …），读者会默认「照这份表改就是安全区」。实际上：

| 段 | 未知键 | 类型错误 |
| --- | --- | --- |
| `env:` | `SystemExit` 并列出可用键 | **静默通过**（无强制转换） |
| `reward:` | `SystemExit` 并列出可用键 | **静默通过** |
| `env.obs` / `env.command` | `SystemExit` 并列出可用键 | **静默通过** |
| `ppo:` | **静默忽略** | **静默通过** |
| `run:` / `play:` | **静默忽略** | **静默通过** |
| `curriculum:` | **静默忽略** | 部分逐项校验 |

### 类型校验也缺失

`_apply()` 的全部实现是：

```python
valid = set(getattr(target, "__dataclass_fields__", {}) or {})
unknown = sorted(set(values) - valid)
if unknown:
    raise SystemExit(f"config.yaml 的 {where}: 无法识别的键 {unknown}；可用键 = {sorted(valid)}")
for key, value in values.items():
    setattr(target, key, value)          # ← 无 isinstance / float() / int()
```

于是 `env.action_scale: "0.30"`（YAML 里加了引号就是 str）会在加载期通过，直到第一次 `home + gait_gain*... + action_scale*action` 做算术时才炸，错误信息离根因很远。YAML 的隐式类型转换（`on`/`off`/`yes`/`no`、`1e-4` 的科学计数法、`0.30` 与 `0.3`）让这类事故有现实概率。

建议：`_apply` 里用 `dataclasses.fields()` 拿 `f.type`，对 `float`/`int`/`bool`/`str`/`tuple`/`list` 做**宽松但显式**的转换（bool 要特别小心 `"false"` 是真值字符串），失败时报出「键 / 期望类型 / 实际类型」。

---

## 已验证无问题（课程表生成）

`config_to_stages()` 的语义链是自洽的，逐条核过：

* **平地阶段必须避开 0 cm 台阶**：平地走 `flat_spec(n_up=0, n_down=0, step_height=0.0, tread=tread)`，台阶走 `demo_stairs_spec(step_height=h_cm/100.0, tread=tread, n_up=n_up)`。实测 `flat_spec()` 的 `height_at(0.95) = 0.0`、`height_at(3.0) = 0.0`，没有零高度 box。
  * ⚠ 勘误（见 `05-tooling-and-docs.md` 的 T-9f）：`flat_spec()` **自己并不会**把 `step_height` 归一成 0 —— `terrain.py:213-216` 只 `setdefault("n_up", 0)` / `setdefault("n_down", 0)`，`StairSpec.step_height` 仍是类默认的 `0.05`。这里是**调用方显式传了 `step_height=0.0`**，所以训练链路正确；但 `scripts/teleop.py:77` 与 `scripts/cmd_demo.py:152` 调的是 `flat_spec(tread=..., width=..., start_x=...)`，于是观测里出现「平地 + 5 cm 台阶」。
* **`heights_cm` 必须非负且升序**，违反即 `SystemExit` —— 与课程单调变难的假设一致。
* **重名阶段有去重**：`flat` / `flat_2` / `flat_3`（`_unique_name`），label 会带上「成功距离」或「需要前进」以区分平地变体。
* **`reward_overrides` 的键会先校验**（`_validate_reward_keys` 对 `RewardConfig.__dataclass_fields__`），长度必须等于阶段数，`None` 表示该级不覆盖 —— 这三条都有硬报错。
* **`_spread()` 的插值保总量**：`timesteps` / `learning_rate` 长度不匹配时会线性重采样，并把 `round` 余数补给最难的一级（`out[-1] += max(0, int(round(sum)) - sum(out))`）。这一点是刻意的，注释也写了。

### 已验证无问题（键位交叉核对）

实测 `config_to_play(load_config())` 返回的键集合恰为

```
{'episodes_gif', 'episodes_eval', 'seed', 'camera', 'width', 'height',
 'fps', 'scale', 'stages', 'batch'}
```

即 `config.yaml` 的 `play:` 段**没有** `device` 键（`stage_height_cm` 被注释掉）。这一点在 `04-training-scripts.md` 的 S-5 里作为一个「被读取但取不到」的键单独记录。

实测 `config_to_ppo(load_config())` 返回 15 个键：`net_arch, n_steps, batch_size, n_epochs, learning_rate, gamma, gae_lambda, clip_range, ent_coef, vf_coef, max_grad_norm, target_kl, log_std_init, norm_reward, device` —— 与 `build_ppo()` 的签名逐项对得上，`norm_reward` 在 `train.py` 里另取。

# `robotdog/config.py` — 配置解析

对应代码：`robotdog/config.py`。每个键的取值理由在 `config.yaml` 的注释里；
调参教训见 `experience.md`；本文只讲抽象结构。

## 职责

把仓根 `config.yaml` 变成三类对象：`EnvConfig`（环境/奖励/观测的全部可调项）、
课程阶段表（`StageSpec` 列表）、以及 `ppo:` / `run:` / `play:` 三段普通 dict。
**config.yaml 是唯一的超参入口**

> 例外（受控）：`ablations/*.yaml` 是**消融/还原配置快照**，同样是「一份 yaml 定全部超参」，
> 用来做单变量对照（见 `doc/ablation.md`）。`.gitignore` 忽略的 `configs/` 是本地训练快照，
> 不进版本库；`ablations/` 进版本库。 —— 脚本不再暴露命令行开关（只剩 `--run-name` /
`--config` / `--run` 这类运行标识）。

## 核心原则：打错的键必须当场报错

配置文件是「唯一入口」的前提是**拼错不可静默**。两条机制：

- dataclass 段（`env:` / `reward:` / 嵌套的 `obs:` / `command:`）走 `_apply`：
  对照 `__dataclass_fields__`，未知键 `SystemExit` 并列出可用键。
- 非 dataclass 段（`ppo:` / `run:` / `play:` / `curriculum:`）是脚本自己 `.get()`
  的，没有那层结构 —— 所以有 `SECTION_KEYS` 白名单补查。校验挂在
  `config_to_env()` 里，因为它是三个脚本共同的第一步。
- `env.n_envs` / `env.vec_env` 是运行参数（不是环境物理属性），由
  `n_envs()` / `vec_env()` 读取；后者接受 `auto` / `dummy` / `subproc`，默认
  `auto`（训练 `n_envs>1` → `subproc`，单环境评估/遥控 → `dummy`；解析见
  `robotdog/training.py::resolve_vec_env`）。

## 每级归一化（课程向量的展开）

课程允许「标量 = 每级相同 / 列表 = 逐级取值」两种写法，四个归一化函数把它展开成
与级数等长的向量，语义各不相同（**刻意不通用化**）：

- `_timesteps` — 标量或列表；列表长度不等时线性重采样保持总量，余数补给最难的级。
- `_floats` — 学习率/指令量程这类标量或列表，列表重采样用插值。
- `_vx_ranges` — 前向指令区间，三种写法（无 / 共用区间 / 逐级一对）；
  `上界为 0` 表示「该级用 env 定值基线」，此时下界必须为 0。
- `_reward_override_table` — 奖励覆盖，`None` = 该级不动这个字段。

布尔量不做插值（`_bools` 已随需求消失而删除）—— 这就是不能抽象成一个通用函数的
原因：边界语义是每个函数自己的。

## `config_to_env` / `config_to_stages`

- `config_to_env`：写 `env:` + `reward:` 两段进 dataclass，重建嵌套的
  `command`（含当前值）与 `obs`，加载步态先验文件。`env.n_envs` 与
  `env.vec_env` 是运行参数，先 pop 掉再校验，不属于 `EnvConfig`。
  **gait_amp_scale 只缩幅值**
  （保留步频），缩完过一遍 `normalize()`。
- `config_to_stages`：展开课程。阶段名由高度决定，重名自动去重
  （`flat` / `flat_2` / …）—— checkpoint 与 tensorboard 按阶段名分目录，重名会
  互相覆盖。

## play 段

只留有语义的开关（回合数、seed、帧上限、渲染哪些阶段、批次标签）。相机/分辨率/
帧率/缩放**故意不进 yaml** —— 它们是「看画面」的事，留成旋钮只会让人以为调它
有意义（固定为 `scripts/play.py` 的模块常量）。

"""读取 ``config.yaml``，把它变成 :class:`~robotdog.env.EnvConfig` 与课程阶段表。

**为什么有这个模块**：训练与播放的超参集中在仓根 ``config.yaml`` 一处，脚本不再
暴露一堆命令行开关。这里只做三件事——读 YAML、按代码默认值补全缺失的键、把已知
的键写进 dataclass 字段；不认识的键直接报错（打错一个字母要立刻看得见，而不是被
静默忽略）。

用法::

    from robotdog.config import load_config, config_to_env, config_to_stages

    raw = load_config()                       # 默认 <仓根>/config.yaml
    cfg = config_to_env(raw)                  # EnvConfig（含 trot 先验）
    stages = config_to_stages(raw)            # 六级 StageSpec

配置文件的完整键与含义见仓根 ``config.yaml`` 的注释，以及 ``README.md`` §4。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from robotdog.curriculum import (
    DEFAULT_N_UP,
    DEFAULT_TREAD,
    STAGE_HEIGHTS_CM,
    StageSpec,
)
from robotdog.env import EnvConfig
from robotdog.gait import GaitParams, load_gait
from robotdog.reward import RewardConfig
from robotdog.tools import paths
from robotdog.tools.terrain import demo_stairs_spec, flat_spec

#: 默认配置文件位置（仓根）。
DEFAULT_CONFIG_PATH: Path = paths.WORKSPACE / "config.yaml"

#: 配置文件省略 ``curriculum.timesteps`` 时的每级预算（平地 : 台阶 = 1 : 2.5）。
DEFAULT_STAGE_TIMESTEPS: list[int] = [100_000] + [250_000] * (len(STAGE_HEIGHTS_CM) - 1)

#: 配置文件省略 ``curriculum.learning_rate`` 时的统一学习率。
DEFAULT_LR: float = 1.5e-4


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """读 ``config.yaml``；``path=None`` 时用仓根的默认文件。

    文件不存在、不是合法 YAML、顶层不是映射，都会 ``SystemExit`` 并给出中文原因。
    """
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not target.is_file():
        raise SystemExit(f"找不到配置文件：{target}")
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:                      # 语法错误
        raise SystemExit(f"配置文件解析失败（{target}）：\n{exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise SystemExit(f"配置文件顶层必须是映射，实际是 {type(data).__name__}")
    return data


def _section(raw: Mapping[str, Any], name: str) -> dict[str, Any]:
    """取出一个子段；缺失 = 空段（全部走默认值）。"""
    value = raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise SystemExit(f"config.yaml 的 {name}: 必须是映射（键值对）")
    return dict(value)


#: 非 dataclass 段（``ppo:`` / ``run:`` / ``play:`` / ``curriculum:``）的合法键。
#: 这些段是脚本自己 ``.get()`` 出来的，没有 ``_apply`` 的字段校验 —— 不查一遍
#: 就会静默用上默认值。打错键必须当场报错，这是「config.yaml 是唯一超参入口」
#: 的前提。
SECTION_KEYS: dict[str, frozenset[str]] = {
    "ppo": frozenset({
        "device", "net_arch", "n_steps", "batch_size", "n_epochs", "gamma",
        "gae_lambda", "clip_range", "ent_coef", "vf_coef", "max_grad_norm",
        "target_kl", "log_std_init", "learning_rate", "norm_reward",
    }),
    "run": frozenset({"name", "seed", "save_every", "log_every"}),
    "play": frozenset({
        "episodes_gif", "episodes_eval", "seed", "gif_max_frames", "stages", "batch",
        "stage_height_cm",
    }),
    "curriculum": frozenset({
        "tread", "n_up", "heights_cm", "timesteps", "learning_rate",
        "command_vy_max", "command_yaw_max", "command_vx", "reward_overrides",
    }),
}


def _check_section_keys(raw: Mapping[str, Any]) -> None:
    """按 :data:`SECTION_KEYS` 检查四个非 dataclass 段有没有拼错的键。"""
    for name, valid in SECTION_KEYS.items():
        section = _section(raw, name)
        unknown = sorted(k for k in section if k not in valid)
        if unknown:
            raise SystemExit(
                f"config.yaml 的 {name}: 无法识别的键 {unknown}；"
                f"可用键 = {sorted(valid)}"
            )


def _timesteps(value: Any, n_stages: int, *, where: str) -> list[int]:
    """把 ``timesteps`` 归一化成 ``n_stages`` 个正整数。

    允许给一个标量（每级相同）或一个列表。列表长度不等时按比例重新采样到
    ``n_stages``（保持总量，余数补给最难的一级），这样改课程级数时不必手改预算。
    """
    if isinstance(value, (int, float)):
        raw = float(value)
        if raw < 0:
            raise SystemExit(f"{where}: 步数不能为负（{value}）")
        return [max(1, int(round(raw)))] * n_stages
    if not isinstance(value, (list, tuple)) or not value:
        raise SystemExit(f"{where}: 期望数字或非空列表，实际是 {type(value).__name__}")

    numbers = [float(v) for v in value]
    if any(v < 0 for v in numbers):
        raise SystemExit(f"{where}: 步数不能为负（{numbers}）")
    if len(numbers) == n_stages:
        resampled = numbers
    else:
        total = sum(numbers)
        if total <= 0:
            raise SystemExit(f"{where}: 之和必须为正")
        unit = total / n_stages                      # 均值，保持总量
        resampled = [w * unit for w in _spread(numbers, n_stages)]

    out = [max(1, int(round(v))) for v in resampled]
    out[-1] += max(0, int(round(sum(numbers))) - sum(out))   # 余数给最难的一级
    return out


def _spread(weights: list[float], n: int) -> list[float]:
    """把 ``weights`` 重新采样成 ``n`` 个权重（线性插值，保持总量）。"""
    if n == 1:
        return [sum(weights)]
    m = len(weights)
    if m == n:
        return weights
    out = []
    for i in range(n):
        pos = i * (m - 1) / (n - 1)
        lo = int(pos)
        hi = min(lo + 1, m - 1)
        frac = pos - lo
        out.append(weights[lo] * (1 - frac) + weights[hi] * frac)
    return out


def _floats(value: Any, n_stages: int, *, where: str, default: float) -> list[float]:
    """把 ``learning_rate`` 之类的标量/列表归一化成 ``n_stages`` 个浮点。"""
    if value is None:
        return [default] * n_stages
    if isinstance(value, (int, float)):
        return [float(value)] * n_stages
    if isinstance(value, (list, tuple)) and value:
        values = [float(v) for v in value]
        if len(values) == n_stages:
            return values
        if len(values) == 1:
            return values * n_stages
        return _spread(values, n_stages)
    raise SystemExit(f"{where}: 期望数字或列表，实际是 {type(value).__name__}")


def _vx_ranges(value: Any, n_stages: int, *, where: str
               ) -> list[tuple[float, float]]:
    """把 ``curriculum.command_vx`` 归一化成每级一对 ``(vx_min, vx_max)``。

    三种写法：``None``（每级都不采样）、``[lo, hi]``（所有级共用这个区间）、
    ``[[lo, hi], ...]``（每级一对，长度必须等于级数）。``hi <= 0`` 表示该级
    用 ``env.command.vx`` 定值；``hi < lo`` 直接报错，否则会在 ``reset()`` 里
    抽出一个 ``lo > hi`` 的空区间。
    """
    if value is None:
        return [(0.0, 0.0)] * n_stages
    if not isinstance(value, (list, tuple)) or not value:
        raise SystemExit(f"{where}: 期望 [lo, hi] 或 [[lo, hi], ...]")

    def _pair(item: Any, where_item: str) -> tuple[float, float]:
        if isinstance(item, (int, float)):
            # 标量 = 定值
            return (0.0, float(item))
        if isinstance(item, (list, tuple)) and len(item) == 2:
            lo, hi = float(item[0]), float(item[1])
            if hi and lo > hi:
                raise SystemExit(f"{where_item}: vx 区间下界 {lo} 大于上界 {hi}")
            if not hi and lo > 0.0:
                raise SystemExit(
                    f"{where_item}: 上界为 0 表示「用 env.command.vx 定值」，"
                    f"此时下界必须为 0，实际是 lo={lo} hi={hi}"
                )
            return (lo, hi)
        raise SystemExit(f"{where_item}: 期望标量或 [lo, hi]，实际是 {item!r}")

    if isinstance(value[0], (int, float)):
        return [_pair(value, where)] * n_stages
    pairs = [_pair(item, f"{where}[{i}]") for i, item in enumerate(value)]
    if len(pairs) == n_stages:
        return pairs
    if len(pairs) == 1:
        return pairs * n_stages
    raise SystemExit(f"{where}: 列表长度 {len(pairs)} 必须等于级数 {n_stages} 或为 1")


# --------------------------------------------------------------------------- #
# 环境配置
# --------------------------------------------------------------------------- #


def config_to_env(raw: Mapping[str, Any]) -> EnvConfig:
    """把 ``env:`` + ``reward:`` 两段写进 :class:`EnvConfig`。

    没写的键保持 dataclass 默认值；未知的键（打错的字段名）报错并列出现有字段。
    顺带把四个非 dataclass 段的键也校验一遍（见 :data:`SECTION_KEYS`）——
    ``config_to_env`` 是三个脚本共同的第一步，校验放这里就够了。
    """
    _check_section_keys(raw)
    env = _section(raw, "env")
    obs_raw = env.pop("obs", None)       # 嵌套子段，单独处理
    cmd_raw = env.pop("command", None)   # 嵌套子段，单独处理
    env.pop("n_envs", None)              # 属于运行参数，不属于 EnvConfig
    gait_file = env.pop("gait_file", None)
    cfg = EnvConfig()

    _apply(cfg, env, where="env")
    _apply(cfg.reward, _section(raw, "reward"), where="reward")

    if cmd_raw is not None:
        if not isinstance(cmd_raw, Mapping):
            raise SystemExit("config.yaml 的 env.command 必须是映射")
        _apply(cfg.command, dict(cmd_raw), where="env.command")
        cfg.command.__post_init__()      # 把 vx 同步进当前指令

    if obs_raw is not None:
        if not isinstance(obs_raw, Mapping):
            raise SystemExit("config.yaml 的 env.obs 必须是映射")
        _apply(cfg.obs, dict(obs_raw), where="env.obs")

    if cfg.gait_prior:
        cfg.gait = _load_gait(gait_file)
        if cfg.gait_amp_scale != 1.0:
            # 只缩振幅、保留高步频：先验自身 2.05 m/s，而指令区间是 0.2~0.6 m/s，
            # 不缩振幅残差就得常驻 −0.5。缩振幅后 |vx 误差| 降约 3.4 倍（见 env.py
            # 的 EnvConfig.gait_amp_scale 注释）。
            cfg.gait.amp_th *= cfg.gait_amp_scale
            cfg.gait.amp_ca *= cfg.gait_amp_scale
            cfg.gait = cfg.gait.normalize()
    return cfg


def _load_gait(gait_file: Any) -> GaitParams:
    """读先验文件；路径相对仓根解析。"""
    if gait_file is None:
        return GaitParams.default_trot()
    path = Path(str(gait_file))
    if not path.is_absolute():
        path = paths.WORKSPACE / path
    if not path.is_file():
        raise SystemExit(f"找不到步态先验文件：{path}")
    return load_gait(path)


def _apply(target: Any, values: Mapping[str, Any], *, where: str) -> None:
    """把 ``values`` 写进 dataclass ``target``，未知键报错。"""
    valid = set(getattr(target, "__dataclass_fields__", {}))
    unknown = sorted(k for k in values if k not in valid)
    if unknown:
        raise SystemExit(
            f"config.yaml 的 {where}: 无法识别的键 {unknown}；"
            f"可用键 = {sorted(valid)}"
        )
    for key, value in values.items():
        setattr(target, key, value)


# --------------------------------------------------------------------------- #
# 课程
# --------------------------------------------------------------------------- #


def _unique_name(base: str, seen: dict[str, int]) -> str:
    """把重名的阶段（例如三个平地级）改成 ``flat`` / ``flat_2`` / ``flat_3``。

    checkpoint 与 tensorboard 都按阶段名分目录，重名会让后一级覆盖前一级，
    所以「同一高度出现多次」必须能生成不同的名字。
    """
    seen[base] = seen.get(base, 0) + 1
    if seen[base] == 1:
        return base
    return f"{base}_{seen[base]}"


def config_to_stages(raw: Mapping[str, Any]) -> list[StageSpec]:
    """把 ``curriculum:`` 段展开成阶段表（默认就是平地 + 3/6/9/12/15 cm）。

    阶段名由高度决定：平地是 ``flat``，其余是 ``stair<整数厘米>``（``3.5`` 会写成
    ``stair3.5``）。步数与学习率按 ``heights_cm`` 的长度对齐。
    """
    cur = _section(raw, "curriculum")
    tread = float(cur.get("tread", DEFAULT_TREAD))
    n_up = int(cur.get("n_up", DEFAULT_N_UP))

    heights = cur.get("heights_cm")
    if heights is None:
        heights = [h * 100.0 for h in STAGE_HEIGHTS_CM]
    if not isinstance(heights, (list, tuple)) or not heights:
        raise SystemExit("config.yaml 的 curriculum.heights_cm 必须是非空列表")
    heights = [float(h) for h in heights]
    if any(h < 0 for h in heights):
        raise SystemExit("curriculum.heights_cm: 台阶高不能为负")
    if heights != sorted(heights):
        raise SystemExit(f"curriculum.heights_cm 必须按高度升序：{heights}")

    n = len(heights)
    budgets = _timesteps(cur.get("timesteps", DEFAULT_STAGE_TIMESTEPS), n,
                         where="curriculum.timesteps")
    lrs = _floats(cur.get("learning_rate"), n,
                  where="curriculum.learning_rate", default=DEFAULT_LR)
    vy_maxes = _floats(cur.get("command_vy_max"), n,
                       where="curriculum.command_vy_max", default=0.0)
    yaw_maxes = _floats(cur.get("command_yaw_max"), n,
                        where="curriculum.command_yaw_max", default=0.0)

    stages: list[StageSpec] = []
    override_table = _reward_override_table(cur, n)
    vx_ranges = _vx_ranges(cur.get("command_vx"), n,
                           where="curriculum.command_vx")
    seen: dict[str, int] = {}
    for h_cm, steps, lr, vy_max, yaw_max, row, vx_range in zip(
        heights, budgets, lrs, vy_maxes, yaw_maxes, override_table, vx_ranges,
    ):
        steps = int(steps)
        if h_cm <= 0.0:
            spec = flat_spec(n_up=0, n_down=0, step_height=0.0, tread=tread)
            name, label = _unique_name("flat", seen), "平地（指令跟随）"
        else:
            spec = demo_stairs_spec(step_height=h_cm / 100.0, tread=tread, n_up=n_up)
            name = _unique_name(f"stair{h_cm:g}", seen)
            label = f"{h_cm:g}cm 台阶"
        overrides = dict(row)
        stages.append(
            StageSpec(
                name=name,
                step_height=float(spec.step_height),
                spec=spec,
                timesteps=steps,
                learning_rate=float(lr),
                label=label,
                description=f"{label} 上行走并到达顶层平台",
                command_vy_max=float(vy_max),
                command_yaw_max=float(yaw_max),
                command_vx=vx_range,
                reward_overrides=overrides,
            )
        )
    return stages


#: 回放时横向/偏航指令取量程的哪个比例（几乎拉满但留一点余量）。
PLAY_COMMAND_FRAC: float = 0.8

#: 每个回合最多写进 GIF 多少帧（渲染帧很大，逐阶段累积曾把内存吃光；超限后
#: 按当前帧数反算步幅稀疏取帧）。
PLAY_GIF_MAX_FRAMES: int = 300


def command_for_stage(stage: StageSpec, *, frac: float = PLAY_COMMAND_FRAC,
                      vx: float | None = None) -> tuple[float, float, float]:
    """给一个课程阶段挑一条**有代表性**的回放指令 ``(vx, vy, yaw_rate)``。

    每级的指令量程不同，同一个固定指令在不同级上含义完全不同 —— 必须按阶段挑。
    规则：量程为 0 的方向给 0（该级本来就不练这个方向）；有量程的方向给
    ``frac × 量程``；``vx`` 显式给定时覆盖，否则取该级采样区间的中点（回放要的是
    落在训练分布中央的典型指令，不是端点）。
    """
    if vx is None:
        lo, hi = stage.command_vx
        out_vx = float((lo + hi) / 2.0) if hi else 0.0
    else:
        out_vx = float(vx)
    vy = float(stage.command_vy_max) * frac if stage.command_vy_max else 0.0
    yaw = float(stage.command_yaw_max) * frac if stage.command_yaw_max else 0.0
    return out_vx, round(vy, 4), round(yaw, 4)


def _validate_reward_keys(values: Mapping[str, Any], *, where: str) -> None:
    """奖励字段名的合法性检查（``RewardConfig`` 是唯一真值来源）。"""
    valid = set(RewardConfig.__dataclass_fields__)
    unknown = sorted(k for k in values if k not in valid)
    if unknown:
        raise SystemExit(
            f"config.yaml 的 {where}: 无法识别的奖励字段 {unknown}；"
            f"可用字段 = {sorted(valid)}"
        )


def _reward_override_table(cur: Mapping[str, Any], n: int
                           ) -> list[dict[str, float]]:
    """把 ``curriculum.reward_overrides`` 展开成「每级一个 dict」。

    两种写法都支持：

    * 标量 —— 该字段在**每一级**都用这个值；
    * 列表 —— 必须与 ``heights_cm`` 等长，逐级取值。

    ``None`` 表示「这一级不动这个字段」。
    """
    raw = cur.get("reward_overrides") or {}
    if not isinstance(raw, Mapping):
        raise SystemExit("config.yaml 的 curriculum.reward_overrides 必须是映射")
    _validate_reward_keys(raw, where="curriculum.reward_overrides")
    table: list[dict[str, float]] = [{} for _ in range(n)]
    for key, value in raw.items():
        if isinstance(value, (list, tuple)):
            if len(value) != n:
                raise SystemExit(
                    f"curriculum.reward_overrides.{key} 有 {len(value)} 项，"
                    f"但 heights_cm 有 {n} 级"
                )
            for i, item in enumerate(value):
                if item is None:
                    continue
                table[i][key] = float(item)
        elif value is None:
            continue
        else:
            for i in range(n):
                table[i][key] = float(value)
    return table


# --------------------------------------------------------------------------- #
# PPO 与播放
# --------------------------------------------------------------------------- #


def config_to_ppo(raw: Mapping[str, Any]) -> dict[str, Any]:
    """``ppo:`` 段原样返回（键的合法性由 :func:`robotdog.training.build_ppo` 校验）。"""
    return _section(raw, "ppo")


def config_to_run(raw: Mapping[str, Any]) -> dict[str, Any]:
    """``run:`` 段，补上默认值。"""
    run = _section(raw, "run")
    run.setdefault("name", "course1")
    run.setdefault("seed", 0)
    run.setdefault("save_every", 50_000)
    run.setdefault("log_every", 20)
    return run


def config_to_play(raw: Mapping[str, Any]) -> dict[str, Any]:
    """``play:`` 段，补上默认值。

    相机/分辨率/帧率/缩放**没有**这四个键：渲染参数固定在 ``scripts/play.py``
    的模块常量里，写进 yaml 只会让人以为调它有意义（残留的键会被「未知键报错」
    拦下，直接删掉即可）。
    """
    play = _section(raw, "play")
    play.setdefault("episodes_gif", 10)
    play.setdefault("episodes_eval", 100)
    play.setdefault("seed", 0)
    #: ``play.gif_max_frames`` —— 单回合最多写进 GIF 的帧数（见 ``PLAY_GIF_MAX_FRAMES``）。
    play.setdefault("gif_max_frames", PLAY_GIF_MAX_FRAMES)
    #: ``play.stages`` —— ``gif`` 模式要渲染哪些课程阶段（地形 × 指令）。
    #: ``"curriculum"`` = 配置里 ``curriculum:`` 段的全部级；``"auto"``（缺省）
    #: 在运行目录里有 ``checkpoints/<阶段>_final.zip`` 时按检查点定级、否则
    #: 退回当前课程的阶段表。也接受阶段名列表或高度列表（cm）。
    play.setdefault("stages", "auto")
    #: ``play.batch`` —— 这一批 GIF 的标签。用来把「不同批次训练出来的 GIF」
    #: 分开放：产物落在 ``runs/<run>/gifs/<批次>/<阶段>/<阶段>.gif``。
    play.setdefault("batch", "")
    return play


def n_envs(raw: Mapping[str, Any]) -> int:
    """``env.n_envs``，缺省 8。"""
    return int(_section(raw, "env").get("n_envs", 8))


def device(raw: Mapping[str, Any]) -> str:
    """``ppo.device``，缺省 ``cpu``。"""
    return str(_section(raw, "ppo").get("device", "cpu"))

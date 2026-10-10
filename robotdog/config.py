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


def _bools(value: Any, n_stages: int, *, where: str, default: bool) -> list[bool]:
    """把布尔标量/列表归一化成 ``n_stages`` 个布尔（``needs_progress`` 用）。

    与 :func:`_floats` 的差别：**不做线性插值**（布尔插值没有意义），列表长度
    必须正好是级数或 1。
    """
    if value is None:
        return [default] * n_stages
    if isinstance(value, bool):
        return [value] * n_stages
    if isinstance(value, (list, tuple)) and value:
        values = [bool(v) for v in value]
        if len(values) == n_stages:
            return values
        if len(values) == 1:
            return values * n_stages
        raise SystemExit(f"{where}: 列表长度 {len(values)} 必须等于级数 {n_stages} 或为 1")
    raise SystemExit(f"{where}: 期望布尔或列表，实际是 {type(value).__name__}")


# --------------------------------------------------------------------------- #
# 环境配置
# --------------------------------------------------------------------------- #


def config_to_env(raw: Mapping[str, Any]) -> EnvConfig:
    """把 ``env:`` + ``reward:`` 两段写进 :class:`EnvConfig`。

    没写的键保持 dataclass 默认值，所以配置文件只需要列出你想改的项。
    未知的键（打错的字段名）会报错并列出现有字段。
    """
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
    flat_distances = _floats(cur.get("flat_success_distance"), n,
                             where="curriculum.flat_success_distance",
                             default=0.0)
    vxs = _floats(cur.get("command_vx"), n,
                  where="curriculum.command_vx", default=0.0)
    needs_progress = _bools(cur.get("flat_success_needs_progress"), n,
                            where="curriculum.flat_success_needs_progress",
                            default=True)
    seen: dict[str, int] = {}
    for h_cm, steps, lr, vy_max, yaw_max, row, flat_d, vx, need_prog in zip(
        heights, budgets, lrs, vy_maxes, yaw_maxes, override_table,
        flat_distances, vxs, needs_progress,
    ):
        steps = int(steps)
        if h_cm <= 0.0:
            spec = flat_spec(n_up=0, n_down=0, step_height=0.0, tread=tread)
            name, label = _unique_name("flat", seen), "平地"
            if not need_prog:
                label = "平地（原地指令，撑满回合）"
            elif flat_d > 0.0:
                label = f"平地（成功距离 {flat_d:g} m）"
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
                command_vx=float(vx),
                flat_success_distance=float(flat_d),
                flat_success_needs_progress=bool(need_prog),
                reward_overrides=overrides,
            )
        )
    return stages


#: 回放时横向/偏航指令取量程的哪个比例。
#: 用满量程（``1.0``）能看出「能不能跟上极限指令」，但早期课程的量程很小、
#: 满量程反而接近噪声；取 ``0.8`` 是「几乎拉满但仍留一点余量」的折中。
PLAY_COMMAND_FRAC: float = 0.8

#: 每个回合最多写进 GIF 多少帧。渲染帧是 960×540 的 uint8（一帧约 1.5 MB），
#: 训练和渲染同时跑时内存很紧：一回合 500 步按 ``FRAME_STRIDE=2`` 就是 250 帧，
#: 但摔倒前若一直站着，回合能拖到 500 帧以上，逐阶段累积就会 ``MemoryError``。
#: 上限一到就按当前帧数反算步幅、稀疏取帧 —— 宁可 GIF 短一点，也不能在
#: 渲染中途崩掉（2026 年 6 月那次 ``MemoryError`` 就是没有这个上限）。
PLAY_GIF_MAX_FRAMES: int = 300


def command_for_stage(stage: StageSpec, *, frac: float = PLAY_COMMAND_FRAC,
                      vx: float | None = None) -> tuple[float, float, float]:
    """给一个课程阶段挑一条**有代表性**的回放指令 ``(vx, vy, yaw_rate)``。

    为什么要按阶段挑：课程里每一级的指令量程不同（平地预热级从
    ``vy_max=0.1`` 逐级放到 ``0.4``，楼梯级全是 0），同一个固定指令在不同级
    上含义完全不同 —— 拿 ``vy=0.4`` 去播放只要量程 0.1 的第 1 级，等于问它
    「跟一个你从没见过的指令」，看画面会误判成「策略不会横移」。

    规则：量程为 0 的方向给 0（该级本来就不练这个方向）；有量程的方向给
    ``frac × 量程``。``vx`` 显式给定时覆盖阶段的前进指令。
    """
    out_vx = float(stage.command_vx if vx is None else vx)
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
    """``play:`` 段，补上渲染默认值。

    ⚠ 这里**没有**相机/分辨率/帧率/缩放那四个键：渲染参数固定在
    ``scripts/play.py`` 的模块常量里（``CAMERA_GIF`` / ``FRAME_WIDTH`` /
    ``FRAME_HEIGHT`` / ``GIF_FPS`` / ``GIF_SCALE``），写进 yaml 只会让人以为
    调它有意义。所以 ``config.yaml`` 的 ``play:`` 段若残留这四个键，
    ``_section`` / ``_apply`` 那套「未知键报错」会在别处先炸 —— 直接删掉即可。
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
    #: 分开放：产物落在 ``runs/<run>/gifs/<批次>/<阶段>.gif``。
    play.setdefault("batch", "")
    return play


def n_envs(raw: Mapping[str, Any]) -> int:
    """``env.n_envs``，缺省 8。"""
    return int(_section(raw, "env").get("n_envs", 8))


def device(raw: Mapping[str, Any]) -> str:
    """``ppo.device``，缺省 ``cpu``。"""
    return str(_section(raw, "ppo").get("device", "cpu"))

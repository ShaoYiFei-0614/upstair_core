"""第 7 步: 训练基础设施 (PPO + 向量化环境 + 检查点)。

**单进程约束**: 沙箱禁止 Windows 命名管道, ``SubprocVecEnv`` /
``multiprocessing`` 会报 ``PermissionError: [WinError 5]``, 所以"并行环境数 8"
只能实现为 ``DummyVecEnv(8)`` —— 同一进程里顺序推进 8 个环境。实测约 560
向量步/秒 (100 万步 ≈ 30 分钟), 不要试图改用线程池 (``mj_step`` 不释放 GIL)。

**观测归一化**: 关节速度 (±20 rad/s) 与关节角 (±3 rad) 量纲差异大, 用
``norm_obs=True, clip_obs=10.0``。奖励归一化默认关闭, 理由见
:func:`make_vec_env`。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    VecNormalize,
)

from robotdog.env import EnvConfig, Go2StairsEnv
from robotdog.tools.paths import RUNS_DIR
from robotdog.tools.terrain import StairSpec

# --------------------------------------------------------------------------- #
# Vectorised environment
# --------------------------------------------------------------------------- #


def make_env_fn(
    stair: StairSpec,
    cfg: EnvConfig | None = None,
    seed: int = 0,
) -> Callable[[], Go2StairsEnv]:
    """A zero-arg factory, as required by SB3's VecEnv constructors."""

    def _init() -> Go2StairsEnv:
        return Go2StairsEnv(stair=stair, cfg=cfg, seed=seed)

    return _init


def make_vec_env(
    stair: StairSpec,
    n_envs: int = 8,
    *,
    cfg: EnvConfig | None = None,
    seed: int = 0,
    normalize: bool = True,
    monitor: bool = True,
    norm_reward: bool = False,
) -> VecEnv:
    """DummyVecEnv(n_envs) -> VecMonitor -> VecNormalize.

    ``norm_reward`` 默认关闭: 课程会在阶段间切换地形, 在 5 cm 上学到的奖励
    尺度到 11 cm 就漂移了。它被保留为旋钮, 是因为侧向惩罚主导了每步奖励
    (``straight_line`` 一项占 72.7%, 而 ``forward_vel`` 只贡献 +0.356), 挣扎中
    的策略回报接近 -6000, critic 拟合不动; 归一化能恢复一个可用的 critic 目标。

    但**增益很小**, 是在 3 cm 台阶、8 环境、150k 步上用两个 arm 各自的
    归一化器直接量到的 (``.tmp/norm_ab.py``):

    ==========  ==============  ==============  ==============
    arm         成功率          ``mean|a|``     11 cm
    ==========  ==============  ==============  ==============
    off         0.474           0.070 -> 0.063  无成功
    on          0.539           0.080 -> 0.083  无成功
    ==========  ==============  ==============  ==============

    所以它是温和的稳定器而非解药。本 docstring 的早期草稿声称它让残差增长
    三倍、把 11 cm 摔倒率从 0.88 降到 0.12 —— 那些数字来自一个后来被判定
    报错回合的探针, 已撤回。
    """
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    fns = [make_env_fn(stair, cfg, seed + i) for i in range(n_envs)]
    venv: VecEnv = DummyVecEnv(fns)
    if monitor:
        venv = VecMonitor(venv)
    if normalize:
        # VecMonitor sits *inside* so episode returns stay unnormalised.
        venv = VecNormalize(
            venv,
            training=True,
            norm_obs=True,
            norm_reward=norm_reward,
            clip_obs=10.0,
        )
    return venv


def set_terrain(venv: VecEnv, stair: StairSpec) -> None:
    """Swap terrain on every sub-env and reset.

    Reuses the same ``VecNormalize`` wrapper on purpose: observation statistics
    accumulated on the previous stage stay valid, which is exactly what makes
    the curriculum transfer cleanly. Terrain is baked into the compiled MJCF, so
    each sub-env recompiles its model (~250 ms x n_envs).
    """
    venv.env_method("set_stair", stair)
    venv.reset()


def set_terrain_per_env(venv: VecEnv, specs: Sequence[StairSpec]) -> None:
    """给每个子环境**各自**一个地形, 然后统一 reset 一次。

    ``--mixed-heights`` 用它: 一个阶段内让各子环境处于不同台阶高度, 使整条
    能力带都出现在每个 PPO 梯度批次里, 从而避免灾难性遗忘已经爬过的台阶。
    耗时与 :func:`set_terrain` 相同 —— 每个子环境都只重编译一个模型。
    """
    n = int(venv.num_envs)
    if len(specs) != n:
        raise ValueError(f"需要 {n} 个地形, 收到 {len(specs)} 个")
    for i, spec in enumerate(specs):
        venv.env_method("set_stair", spec, indices=[i])
    venv.reset()


# --------------------------------------------------------------------------- #
# PPO
# --------------------------------------------------------------------------- #

DEFAULT_NET_ARCH = (256, 256)


def build_ppo(
    venv: VecEnv,
    *,
    seed: int = 0,
    n_steps: int = 256,
    batch_size: int = 512,
    n_epochs: int = 5,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.002,
    vf_coef: float = 0.5,
    max_grad_norm: float = 1.0,
    target_kl: float | None = 0.025,
    log_std_init: float = -0.7,
    net_arch: Sequence[int] = DEFAULT_NET_ARCH,
    tensorboard_log: str | Path | None = None,
    device: str = "cpu",
    verbose: int = 1,
) -> PPO:
    """PPO with settings tuned for 420-dim locomotion observations on CPU.

    ``log_std_init=-0.7`` (std ~0.5 in normalised action units) matters: the
    default 0.0 means std=1.0, i.e. the residual action saturates the full
    +/-0.30 rad joint offset every step, which is pure noise for a standing dog.
    """
    return PPO(
        "MlpPolicy",
        venv,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        policy_kwargs=dict(
            net_arch=dict(pi=list(net_arch), vf=list(net_arch)),
            activation_fn=nn.Tanh,
            log_std_init=log_std_init,
        ),
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
        device=device,
        seed=seed,
        verbose=verbose,
    )


def load_ppo(path: str | Path, venv: VecEnv, *, device: str = "cpu") -> PPO:
    return PPO.load(str(path), env=venv, device=device)


#: ``EnvConfig`` 里是元组而不是列表的字段。``summary.json`` 经过 JSON 往返后
#: 全部变成 list, 重建时必须转回 tuple, 否则 ``kp``/``kd`` 之类会被当成可变序列。
_ENV_CONFIG_TUPLE_FIELDS = frozenset({"kp", "kd", "gravity", "friction"})


def find_run_dir(start: str | Path, *, max_up: int = 4) -> Path | None:
    """从 ``start`` 往上找最近的含 ``summary.json`` 的目录。"""
    p = Path(start)
    if p.is_file():
        p = p.parent
    for _ in range(max_up + 1):
        if (p / "summary.json").exists():
            return p
        if p.parent == p:
            break
        p = p.parent
    return None


def _model_observation_dim(model_path: str | Path) -> int | None:
    """从 SB3 的 ``.zip`` 里读出保存的观测维度（不解包、不加载模型）。

    SB3 2.9 把 ``data`` 存成 JSON, 且用自定义编码器把每个对象写成
    ``{":type": ..., ":serialized": "<base64 pickle>"}``, 所以要
    JSON → base64 → pickle 三步; SB3 < 2.0 直接 pickle 整个 ``data``, 保留退路。
    """
    import base64
    import json
    import pickle
    import zipfile

    def _shape_of(obj: object) -> int | None:
        shape = getattr(obj, "shape", None)
        if shape is None and isinstance(obj, dict):
            shape = obj.get("shape")
        if shape is None:
            return None
        try:
            return int(shape[0])
        except (TypeError, IndexError, ValueError):
            return None

    try:
        with zipfile.ZipFile(model_path) as zf:
            raw = zf.read("data")
    except Exception:  # noqa: BLE001 - 探测函数, 失败就走别的分支
        return None

    # 路径 1: SB3 >= 2.0, data 是 JSON
    if raw[:1] == b"{":
        try:
            data = json.loads(raw.decode("utf-8"))
            space = data.get("observation_space")
            if isinstance(space, dict) and ":serialized:" in space:
                space = pickle.loads(base64.b64decode(space[":serialized:"]))
            return _shape_of(space)
        except Exception:  # noqa: BLE001
            return None

    # 路径 2: 旧版 SB3, data 是整个 pickle
    try:
        return _shape_of(pickle.loads(raw).get("observation_space"))
    except Exception:  # noqa: BLE001
        return None


def _obs_config_candidates(total: int, history_len: int = 10) -> list[ObsConfig]:
    """所有能拼出 ``total`` 维的可选观测块组合（可能不止一个）。"""
    from itertools import product

    from robotdog.env import ObsConfig

    # 顺序即优先级: ``product`` 让**靠后**的标志变化最快，所以「新标志」必须排在
    # 同维度的「老标志」**前面**，才能让维度反推优先回到老配置。
    # ``base_lin_vel_body`` 与老的 ``base_lin_vel`` 都是 +3 维且互斥，把它放在
    # ``base_lin_vel`` 之前 => 反推 660 维时先命中 ``base_lin_vel=True``（老）而不是
    # ``base_lin_vel_body=True``（新）。三个触觉/测距/漂移标志同理排在老标志之前。
    flags = ("stair_probe_full", "touch", "stair_probe", "drift",
             "base_lin_vel_body", "base_lin_vel",
             "base_height", "foot_height", "gait_phase")
    out: list[ObsConfig] = []
    for combo in product((False, True), repeat=len(flags)):
        kw = dict(zip(flags, combo))
        # ``stair_probe_full`` 只在 ``stair_probe`` 打开时才有意义（它多给 2 维）。
        if kw["stair_probe_full"] and not kw["stair_probe"]:
            continue
        cand = ObsConfig(history_len=history_len, **kw)
        if cand.size == total:
            out.append(cand)
    return out


def _obs_config_for_dim(total: int, history_len: int = 10) -> ObsConfig | None:
    """反推是哪几个可选观测块拼出了 ``total`` 维。

    各块维度不互质 (``base_lin_vel``/``base_lin_vel_body``/``stair_probe`` 都是
    3, ``gait_phase``/``drift`` 都是 2), 所以存在维度相同的不同组合。本函数返回
    枚举顺序里的第一个, 而枚举顺序被刻意排成「新标志最后开」, 使老维度总是反推回
    老配置; 要判断是否有歧义用 :func:`_obs_config_candidates`。首选路径始终是 run
    目录里的 ``summary.json`` (精确记录)。
    """
    cands = _obs_config_candidates(total, history_len)
    return cands[0] if cands else None


def env_config_for_model(model_path: str | Path) -> EnvConfig:
    """给一个 ``.zip`` 模型, 恢复出它能加载的 :class:`EnvConfig`。

    首选 run 目录里的 ``summary.json``（精确）; 没有时从 ``.zip`` 里保存的观测
    维度**反推**是哪几个可选观测块 (这条路恢复不了 ``--cfg`` 覆盖过的超参, 会打
    一行提示)。必须有它是因为 ``EnvConfig()`` 默认 420 维, 而带
    ``--obs-foot-height`` 训练的模型是 490 维, 直接用默认配置 ``PPO.load`` 会抛
    ``ValueError: Observation spaces do not match``。
    """
    run_dir = find_run_dir(model_path)
    if run_dir is not None:
        return env_config_from_run(run_dir)

    dim = _model_observation_dim(model_path)
    if dim is None:
        print(f"[配置] {Path(model_path).name} 里读不到观测维度, "
              f"退回 EnvConfig() 默认值")
        return EnvConfig()
    obs = _obs_config_for_dim(dim)
    if obs is None:
        raise SystemExit(
            f"模型观测维度 {dim} 无法用 EnvConfig 的可选观测块还原"
            f"（期望 42/44/47/49 x history_len 的整数倍）。"
            f"请用 --cfg 显式指定, 或提供 run 目录的 summary.json。")
    ambiguous = _obs_config_candidates(dim)
    if len(ambiguous) > 1:
        _flag_names = ("base_lin_vel", "base_height", "foot_height",
                       "gait_phase", "touch", "stair_probe", "drift")
        names = ", ".join(
            "{" + "+".join(n for n in _flag_names if getattr(c, n)) + "}"
            for c in ambiguous
        )
        print(f"[配置] ⚠ 观测维度 {dim} 对应 {len(ambiguous)} 种可选块组合 "
              f"({names}), 已选第一个; 想要精确配置请提供 run 目录的 summary.json。")
    cfg = EnvConfig(obs=obs)
    prior_note = ""
    if obs.gait_phase:
        # 观测里有步态时钟 -> 几乎必然开了先验（train.py --gait-prior 会自动
        # 打开 obs.gait_phase）。不开的话机器人只会站在原地，成功率恒 0，
        # 会被误读成「策略什么都没学」。
        cfg.gait_prior = True
        prior_note = ", gait_prior=True(反推)"
    print(f"[配置] 模型旁没有 summary.json, 已从观测维度 {dim} 反推: "
          f"frame={obs.frame_size} (base_height={obs.base_height}, "
          f"foot_height={obs.foot_height}, gait_phase={obs.gait_phase}), "
          f"history={obs.history_len}{prior_note}。**注意**: --cfg 覆盖过的超参"
          f"(如 episode_length_s) 无法反推, 已用默认值。")
    return cfg


def env_config_from_run(run_dir: str | Path) -> EnvConfig:
    """从 ``runs/<name>/summary.json`` 恢复训练时用的 :class:`EnvConfig`。

    ``summary.json`` 里存了 ``env_config`` / ``obs_config`` / ``reward_config``
    三份 asdict, 这里把它们拼回来。找不到 summary.json 时退化为默认
    :class:`EnvConfig`（并如实说明）。
    """
    import json as _json
    from dataclasses import fields as _fields

    from robotdog.env import ObsConfig, RewardConfig
    from robotdog.tools.touch import TouchConfig

    summary_path = Path(run_dir)
    if summary_path.is_dir():
        summary_path = summary_path / "summary.json"
    if not summary_path.exists():
        print(f"[配置] 没有 {summary_path.name}, 退回 EnvConfig() 默认值 "
              f"(若模型观测不是 420 维, 加载会失败)")
        return EnvConfig()

    data = _json.loads(summary_path.read_text(encoding="utf-8"))
    env_kw = dict(data.get("env_config") or {})
    obs_kw = dict(data.get("obs_config") or {})
    rew_kw = dict(data.get("reward_config") or {})

    for key in _ENV_CONFIG_TUPLE_FIELDS:
        if isinstance(env_kw.get(key), list):
            env_kw[key] = tuple(env_kw[key])

    if rew_kw:
        known = {f.name for f in _fields(RewardConfig)}
        env_kw["reward"] = RewardConfig(**{k: v for k, v in rew_kw.items() if k in known})
    if obs_kw:
        known = {f.name for f in _fields(ObsConfig)}
        env_kw["obs"] = ObsConfig(**{k: v for k, v in obs_kw.items() if k in known})

    # ``gait`` 是嵌套 dataclass, asdict 会把它摊平成 dict; 直接塞回 EnvConfig
    # 会让 ``cfg.gait.radians()`` 炸。这里重建 GaitParams。
    if isinstance(env_kw.get("gait"), dict):
        from robotdog.gait import GaitParams

        gait_kw = dict(env_kw["gait"])
        if isinstance(gait_kw.get("calf_shape"), int):
            from robotdog.gait import CALF_SHAPES

            gait_kw["calf_shape"] = CALF_SHAPES[int(gait_kw["calf_shape"])]
        known_gait = {f.name for f in _fields(GaitParams)}
        env_kw["gait"] = GaitParams(**{k: v for k, v in gait_kw.items()
                                       if k in known_gait})

    # Same story for the step-11 tactile geometry block.
    if isinstance(env_kw.get("touch"), dict):
        touch_kw = dict(env_kw["touch"])
        known_touch = {f.name for f in _fields(TouchConfig)}
        env_kw["touch"] = TouchConfig(**{k: v for k, v in touch_kw.items()
                                         if k in known_touch})

    known_env = {f.name for f in _fields(EnvConfig)}
    unknown = sorted(set(env_kw) - known_env)
    if unknown:
        print(f"[配置] summary.json 里有当前 EnvConfig 不认识的字段, 已忽略: {unknown}")
    cfg = EnvConfig(**{k: v for k, v in env_kw.items() if k in known_env})
    print(f"[配置] 从 {summary_path.name} 恢复 EnvConfig: "
          f"观测 {cfg.obs.size} 维 (frame={cfg.obs.frame_size}, "
          f"history={cfg.obs.history_len}), episode={cfg.episode_length_s}s, "
          f"action_scale={cfg.action_scale}")
    return cfg


# --------------------------------------------------------------------------- #
# Run directory bookkeeping
# --------------------------------------------------------------------------- #


class RunDir:
    """``runs/<name>/``: 模型 / 检查点 / tensorboard / 汇总报告。"""

    def __init__(self, name: str, root: str | Path | None = None) -> None:
        self.name = name
        self.root = Path(root) if root else (RUNS_DIR / name)
        self.checkpoints = self.root / "checkpoints"
        self.tensorboard = self.root / "tensorboard"
        self.artifacts = self.root / "artifacts"
        for d in (self.root, self.checkpoints, self.tensorboard, self.artifacts):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def final_model(self) -> Path:
        return self.root / "model_final"

    @property
    def best_model(self) -> Path:
        return self.root / "model_best"

    @property
    def vecnormalize(self) -> Path:
        return self.root / "vecnormalize.pkl"

    @property
    def summary(self) -> Path:
        return self.root / "summary.json"

    def save_model(self, model: PPO, path: Path, venv: VecEnv | None = None) -> None:
        model.save(str(path))
        target = venv if venv is not None else getattr(model, "env", None)
        vec = getattr(model, "get_vec_normalize_env", lambda: None)() or target
        vec_norm = _find_vecnormalize(vec)
        if vec_norm is not None:
            vec_norm.save(str(path) + "_vecnormalize.pkl")

    def write_summary(self, payload: dict[str, Any]) -> Path:
        self.summary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return self.summary


def _find_vecnormalize(venv: Any) -> VecNormalize | None:
    seen = 0
    while venv is not None and seen < 10:
        if isinstance(venv, VecNormalize):
            return venv
        venv = getattr(venv, "venv", None)
        seen += 1
    return None


def restore_vecnormalize(venv: VecEnv, path: str | Path) -> bool:
    """把 :meth:`RunDir.save_model` 写出的观测统计还原进 ``venv``。

    ``PPO.load`` 只还原网络权重; ``VecNormalize`` 的 running mean/var 是单独存成
    ``<模型>_vecnormalize.pkl`` 的。不还原它, 策略就会拿到按 ``mean=0/var=1``
    归一化的输入, 而它训练时看到的是真统计 —— 等于换了一套输入坐标系。

    ``VecNormalize.load`` 要求传入**未包装**的 ``VecEnv`` (它会新建包装器), 这里
    不想再套一层, 所以加载完只把统计量拷回现有包装器。返回是否成功。
    """
    p = Path(path)
    if not p.exists():
        return False
    vec = _find_vecnormalize(venv)
    if vec is None:
        return False
    loaded = VecNormalize.load(str(p), vec.venv)
    vec.obs_rms = loaded.obs_rms
    vec.ret_rms = loaded.ret_rms
    vec.clip_obs = loaded.clip_obs
    vec.clip_reward = loaded.clip_reward
    vec.epsilon = loaded.epsilon
    return True


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


class Stopwatch:
    def __init__(self) -> None:
        self.t0 = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.t0

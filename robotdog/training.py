"""训练基础设施：向量化环境 + PPO + 运行目录 + 模型加载。

**单进程约束**：沙箱禁止 Windows 命名管道，``SubprocVecEnv`` 不可用，只能
``DummyVecEnv`` 串行推进；``mj_step`` 不释放 GIL，线程池没有收益。
**观测归一化必须开、奖励归一化默认开**：不归一化时关节速度与关节角量纲差三个
数量级、critic 拟合不动；惩罚主导时残差学不起来。实测依据见 ``experience.md`` §4。
"""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from typing import Any, Callable

import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    VecNormalize,
)

from robotdog.env import EnvConfig, Go2StairsEnv
from robotdog.policy import ObsNorm, env_config_for_model, policy_arch
from robotdog.tools.paths import RUNS_DIR
from robotdog.tools.terrain import StairSpec

#: 与本项目实测匹配的网络宽度（SB3 默认的 64×64 太窄）。
DEFAULT_NET_ARCH = (512, 256)


# --------------------------------------------------------------------------- #
# 向量化环境
# --------------------------------------------------------------------------- #


def make_env_fn(
    stair: StairSpec | None,
    cfg: EnvConfig | None = None,
    seed: int = 0,
) -> Callable[[], Go2StairsEnv]:
    """SB3 的 ``VecEnv`` 构造器要求一个零参工厂。

    ⚠ **每个环境必须拿到自己的一份 cfg**：``Go2StairsEnv._command`` 就是
    ``cfg.command._now``（同一个 list），若各 env 共享 cfg，那么一个 env 写指令
    （含每步的 OU 扰动）会同时改掉其它 env 的观测/奖励输入。审计实测（2026-10-18
    的 audit_sharedcfg 探针）：两个共享 cfg 的 env 中 ``e0.set_command_now(...)``
    之后 ``e1._command`` 也变了。deepcopy 不影响 :mod:`scripts.train` 的逐级下发
    —— 它用 ``venv.env_method`` 对每个子环境单独调用 ``set_reward_overrides`` /
    ``set_commands``。
    """

    def _init() -> Go2StairsEnv:
        return Go2StairsEnv(stair=stair,
                            cfg=copy.deepcopy(cfg) if cfg is not None else None,
                            seed=seed)

    return _init


def make_vec_env(
    stair: StairSpec | None,
    n_envs: int = 8,
    *,
    cfg: EnvConfig | None = None,
    seed: int = 0,
    normalize: bool = True,
    monitor: bool = True,
    norm_reward: bool = True,
) -> VecEnv:
    """``DummyVecEnv(n_envs) → VecMonitor → VecNormalize``。

    ``VecMonitor`` 放在 ``VecNormalize`` **里面**，这样 ``info["episode"]["r"]``
    保持未归一化的真实回报，报表读到的就是原始数值。
    """
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    venv: VecEnv = DummyVecEnv(
        [make_env_fn(stair, cfg, seed + i) for i in range(n_envs)]
    )
    if monitor:
        venv = VecMonitor(venv)
    if normalize:
        venv = VecNormalize(
            venv, training=True, norm_obs=True, norm_reward=norm_reward, clip_obs=10.0
        )
    return venv


# --------------------------------------------------------------------------- #
# PPO
# --------------------------------------------------------------------------- #


def build_ppo(
    venv: VecEnv,
    *,
    seed: int = 0,
    n_steps: int = 256,
    batch_size: int = 512,
    n_epochs: int = 5,
    learning_rate: float = 1.5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.002,
    vf_coef: float = 0.5,
    max_grad_norm: float = 1.0,
    target_kl: float | None = 0.03,
    log_std_init: float = -2.0,
    net_arch: tuple[int, ...] = DEFAULT_NET_ARCH,
    tensorboard_log: str | Path | None = None,
    device: str = "cpu",
    verbose: int = 1,
) -> PPO:
    """构造 PPO。

    ``log_std_init=-2.0``（归一化动作单位下 std ≈ 0.135）是折中：默认 0.0 让残差
    每步顶满 ``action_scale``、等于纯噪声；过小则残差恒零、退化成纯开环步态。
    ``batch_size`` 与样本数不整除时 SB3 会 warn —— 对只做 ``predict`` 的推理场景
    无意义，定向压掉。
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*mini-batch size.*",
                                category=UserWarning)
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


def load_policy(
    model_path: str | Path,
    *,
    cfg: EnvConfig | None = None,
    stair: StairSpec | None = None,
    device: str = "cpu",
    norm_reward: bool = False,
) -> tuple[PPO, ObsNorm | None]:
    """加载一个 checkpoint，返回 ``(模型, 观测归一化统计)``。

    统计量来自同目录同名的 ``*_vecnormalize.pkl``，**必须装回模型自带的
    ``VecNormalize``**（``restore_vecnormalize``）—— 只返回一份拷贝而不装回的话，
    ``model.get_env()`` 路径等于喂未归一化的观测，与裸 env 路径给出相反的评测
    结论（experience.md §5.9）。网络结构用 checkpoint 里保存的那一套
    （``policy_arch``），不用默认值建网。
    """
    model_path = Path(model_path)
    if not model_path.is_file() and model_path.with_suffix(".zip").is_file():
        model_path = model_path.with_suffix(".zip")
    cfg = cfg or env_config_for_model(model_path)
    # 网络结构必须用**训练时那一套**：按默认 (512, 256) 建网再 set_parameters，
    # 用别的结构训出来的权重会 size mismatch（见 policy_arch 的 docstring）。
    arch = policy_arch(model_path) or DEFAULT_NET_ARCH
    venv = make_vec_env(stair, n_envs=1, cfg=cfg, seed=0, norm_reward=norm_reward)
    model = build_ppo(venv, seed=0, device=device, net_arch=arch)
    model.set_parameters(str(model_path))

    cand = model_path.with_name(model_path.stem + "_vecnormalize.pkl")
    if cand.is_file():
        restore_vecnormalize(venv, cand)
        norm = ObsNorm.from_path(cand)
    else:
        norm = None
    return model, norm


# --------------------------------------------------------------------------- #
# VecNormalize 统计
# --------------------------------------------------------------------------- #


def _find_vecnormalize(venv: Any) -> VecNormalize | None:
    """沿 ``.venv`` 链下钻找 ``VecNormalize``（最多 10 层，防环）。"""
    seen = 0
    while venv is not None and seen < 10:
        if isinstance(venv, VecNormalize):
            return venv
        venv = getattr(venv, "venv", None)
        seen += 1
    return None


def save_vecnormalize(model: PPO, base: str | Path) -> Path | None:
    """把观测统计写进 ``<base>_vecnormalize.pkl``；拿不到包装器时返回 ``None``。"""
    vec = getattr(model, "get_vec_normalize_env", lambda: None)()
    if vec is None:
        vec = _find_vecnormalize(getattr(model, "env", None))
    if vec is None:
        return None
    path = Path(str(base) + "_vecnormalize.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    vec.save(str(path))
    return path


def restore_vecnormalize(venv: VecEnv, path: str | Path) -> VecNormalize | None:
    """把 ``*_vecnormalize.pkl`` 的统计量拷回 ``venv`` 里的包装器并返回它。

    ``VecNormalize.load`` 会新建一个包装器，这里只想搬统计量，所以加载后逐个字段拷。
    """
    p = Path(path)
    if not p.is_file():
        return None
    vec = _find_vecnormalize(venv)
    if vec is None:
        return None
    loaded = VecNormalize.load(str(p), vec.venv)
    vec.obs_rms = loaded.obs_rms
    vec.ret_rms = loaded.ret_rms
    vec.clip_obs = loaded.clip_obs
    vec.clip_reward = loaded.clip_reward
    vec.epsilon = loaded.epsilon
    return vec


# --------------------------------------------------------------------------- #
# 运行目录
# --------------------------------------------------------------------------- #


class RunDir:
    """``runs/<name>/``：模型 / 检查点 / tensorboard / 汇总。"""

    def __init__(self, name: str, root: str | Path | None = None) -> None:
        self.name = name
        self.root = Path(root) if root else (RUNS_DIR / name)
        self.checkpoints = self.root / "checkpoints"
        self.tensorboard = self.root / "tensorboard"
        self.artifacts = self.root / "artifacts"
        # 目录**不在这里创建**：调用方需要先判断 runs/<name> 是否已经存在（避免把
        # 上次的 run 覆盖掉），而 eager mkdir 会让「刚构造出来就已经存在」永远成立。
        # 各写入点自己 mkdir(parents=True)，见 save_model / write_summary / CheckpointCallback。

    def ensure(self) -> Path:
        """建好 ``runs/<name>/`` 及其子目录，返回根路径。"""
        for d in (self.root, self.checkpoints, self.tensorboard, self.artifacts):
            d.mkdir(parents=True, exist_ok=True)
        return self.root

    def __repr__(self) -> str:  # pragma: no cover - 只为日志好看
        return f"RunDir({self.root})"

    @property
    def final_model(self) -> Path:
        return self.root / "model_final"

    @property
    def vecnormalize(self) -> Path:
        return self.root / "model_final_vecnormalize.pkl"

    @property
    def summary(self) -> Path:
        return self.root / "summary.json"

    def save_model(self, model: PPO, base: str | Path,
                   venv: VecEnv | None = None) -> tuple[Path, Path | None]:
        """存权重：``<base>.zip`` + ``<base>_vecnormalize.pkl``。"""
        base = Path(base)
        base.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(base))
        vec = _find_vecnormalize(venv) if venv is not None else None
        if vec is not None:
            p = Path(str(base) + "_vecnormalize.pkl")
            vec.save(str(p))
            return Path(str(base) + ".zip"), p
        return Path(str(base) + ".zip"), save_vecnormalize(model, base)


# --------------------------------------------------------------------------- #
# 报表辅助
# --------------------------------------------------------------------------- #


__all__ = [
    "DEFAULT_NET_ARCH",
    "RunDir",
    "build_ppo",
    "load_policy",
    "make_env_fn",
    "make_vec_env",
    "restore_vecnormalize",
    "save_vecnormalize",
]

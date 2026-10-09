"""训练基础设施：向量化环境 + PPO + 运行目录 + 模型加载。

**单进程约束**：沙箱禁止 Windows 命名管道，``SubprocVecEnv`` / ``multiprocessing``
会 ``PermissionError: [WinError 5]``，所以「并行 8 个环境」只能实现为
``DummyVecEnv(8)``——同一进程里顺序推进 8 个环境。``mj_step`` 不释放 GIL，换线程池
没有收益。实测 250~400 向量步/秒。

**观测归一化必须开**：关节速度（±20 rad/s）与关节角（±3 rad）量纲差三个数量级，
不归一化时 critic 完全拟合不动。

**奖励归一化默认开**：早期版本默认关闭（理由是课程切换地形后奖励尺度会漂移）。
但本项目的奖励里侧向惩罚项曾经主导每一步（实测 ``straight_line`` 一项占 72.7%，
``forward_vel`` 只贡献 +0.356），挣扎中的策略每回合回报接近 -6000，critic 的
``explained_variance`` 长期停在 0.01 附近，PPO 于是把策略压回「几乎不动」的极小
残差。打开奖励归一化后残差才开始增长。唯一例外是 warm-start 一个**未归一化**训出来
的权重：那时统计量对不上，应当关掉。
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
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
    """SB3 的 ``VecEnv`` 构造器要求一个零参工厂。"""

    def _init() -> Go2StairsEnv:
        return Go2StairsEnv(stair=stair, cfg=cfg, seed=seed)

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


def set_terrain(venv: VecEnv, stair: StairSpec) -> None:
    """给每个子环境换地形并 reset。

    地形烘焙在编译后的 MJCF 里，所以每个子环境都要重编译（约 250 ms × n_envs）。
    ``VecNormalize`` 包装器被刻意复用：上一阶段攒下的观测统计继续有效，课程才能
    干净地迁移。
    """
    venv.env_method("set_stair", stair)
    venv.reset()


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

    ``log_std_init=-2.0``（归一化动作单位下 std ≈ 0.135）很关键：默认的 0.0 意味着
    std = 1.0，即残差每一步都把关节目标顶满 ±``action_scale``，对站立的四足来说就是
    纯噪声。但也不能设得过小——如果残差恒为 0，策略就退化成「纯开环步态」，永远学不会
    爬楼梯（旧课程里出现过 ``mean|a| = 0.011``、``|a| < 0.05`` 占 99.9% 的退化）。

    ``batch_size`` 与 ``n_steps * n_envs`` 不整除时 SB3 会 warn。训练时 8 环境
    （2048）整除，但**推理时只建 1 个环境**（256）就会触发；那条警告对只做
    ``predict`` 的场景没有意义，这里直接压掉。
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


def load_ppo(path: str | Path, venv: VecEnv, *, device: str = "cpu") -> PPO:
    """加载检查点；``path`` 带不带 ``.zip`` 后缀都可以。"""
    p = Path(path)
    if not p.is_file() and p.suffix.lower() != ".zip":
        alt = p.with_suffix(".zip")
        if alt.is_file():
            p = alt
    return PPO.load(str(p), env=venv, device=device)


def load_policy(
    model_path: str | Path,
    *,
    cfg: EnvConfig | None = None,
    stair: StairSpec | None = None,
    device: str = "cpu",
    norm_reward: bool = False,
) -> tuple[PPO, ObsNorm | None]:
    """加载一个 checkpoint，返回 ``(模型, 观测归一化统计)``。

    归一化统计来自同目录同名的 ``*_vecnormalize.pkl``。**不还原它，策略就会拿到按
    ``mean=0 / var=1`` 归一化的输入，而训练时看到的是真统计**——等于换了一套输入
    坐标系，确定性评估会从 100% 掉到 0%。所以这里做两件事：

    1. 读裸 pickle，把 ``obs_rms.mean/var`` 拷成 :class:`~robotdog.policy.ObsNorm`
       返回给调用方（``VecNormalize`` 对象本身不能随便访问属性，多读一个不存在的
       名字就 ``RecursionError``）；
    2. 用 :func:`restore_vecnormalize` 把同一份统计量**装回模型自带的 VecEnv**。
       少了这一步，``model.get_env()`` 拿到的包装器还是 ``mean=0 / var=1``，经它
       评估等于喂未归一化的观测 —— 同一个权重走 ``model.get_env()`` 和走裸
       ``Go2StairsEnv`` 会给出相反结论（3 cm 上前者 0% 摔倒、后者 100% 成功）。
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
    norm = ObsNorm.from_path(cand) if cand.is_file() else None
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


def format_duration(seconds: float) -> str:
    """把秒数写成 ``12.3s`` / ``4.5min`` / ``1.20h``。"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"
def mean_abs_action(actions: np.ndarray) -> float:
    """``mean(|a|)``：判断策略是否退化成「纯先验」的关键量（健康值约 0.3~0.45）。"""
    return float(np.mean(np.abs(np.asarray(actions, dtype=np.float64))))


__all__ = [
    "DEFAULT_NET_ARCH",
    "RunDir",
    "build_ppo",
    "format_duration",
    "load_policy",
    "load_ppo",
    "make_env_fn",
    "make_vec_env",
    "mean_abs_action",
    "restore_vecnormalize",
    "save_vecnormalize",
    "set_terrain",
]

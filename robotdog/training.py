"""训练基础设施：向量化环境 + PPO + 运行目录 + 模型加载。

**向量化选择**：``env.vec_env`` 默认 ``auto`` —— 训练（``n_envs > 1``，默认 8）
用 ``SubprocVecEnv``（Windows 下 spawn）把 env 采样摊到多核；单环境评估/遥控/
权重加载（``n_envs == 1``）自动退回 ``DummyVecEnv``（``venv.envs[0]`` 必须是真
env，脚本里有直接访问 ``set_command_now`` 的路径）。必要时可用 ``dummy`` 强制
单进程。``mj_step`` 不释放 GIL，线程池没有收益，所以并行必须是进程级。
**观测归一化必须开、奖励归一化默认开**：不归一化时关节速度与关节角量纲差三个
数量级、critic 拟合不动；惩罚主导时残差学不起来。实测依据见 ``experience.md`` §4。
"""

from __future__ import annotations

import copy
import io
import json
import pickle
import warnings
import zipfile
from functools import partial
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecMonitor,
    VecNormalize,
)

from robotdog.asymmetric import AsymmetricActorCriticPolicy
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

    return partial(_make_env, stair, cfg, seed)


def _make_env(stair: StairSpec | None,
              cfg: EnvConfig | None,
              seed: int) -> Go2StairsEnv:
    """模块级工厂：``SubprocVecEnv`` 用 cloudpickle 序列化后必须在 worker 里
    重新 import 到这个符号，嵌套闭包在 spawn 下不可靠。"""
    return Go2StairsEnv(stair=stair,
                        cfg=copy.deepcopy(cfg) if cfg is not None else None,
                        seed=seed)


def resolve_vec_env(n_envs: int, vec_env: str = "auto") -> str:
    """把 ``env.vec_env`` 配置值解析成实际实现（``dummy`` / ``subproc``）。

    ``auto``（默认）：训练（``n_envs > 1``，默认 8）用 ``subproc``，把采样摊到
    多核；单环境（评估 / 遥控 / 加载权重）用 ``dummy`` —— ``SubprocVecEnv`` 的
    ``venv.envs[0]`` 只是 worker 代理，拿不到真实 env 的属性与 ``set_command_now``
    之类的直接调用。显式写 ``dummy`` / ``subproc`` 时不做任何猜测。
    """
    value = str(vec_env).strip().lower()
    if value == "auto":
        return "subproc" if n_envs > 1 else "dummy"
    if value not in ("dummy", "subproc"):
        raise ValueError(
            f"vec_env 必须是 'auto' / 'dummy' / 'subproc'，收到 {vec_env!r}"
        )
    return value


def make_vec_env(
    stair: StairSpec | None,
    n_envs: int = 8,
    *,
    cfg: EnvConfig | None = None,
    seed: int = 0,
    normalize: bool = True,
    monitor: bool = True,
    norm_reward: bool = True,
    vec_env: str = "auto",
    start_method: str = "spawn",
) -> VecEnv:
    """``DummyVecEnv(n_envs) | SubprocVecEnv(n_envs) → VecMonitor → VecNormalize``。

    ``vec_env`` 取 ``auto``（默认）/ ``dummy`` / ``subproc``：``auto`` 在
    ``n_envs > 1``（训练，默认 8）时用多进程，单环境时退回单进程 —— 解析规则见
    :func:`resolve_vec_env`。
    ``VecMonitor`` 放在 ``VecNormalize`` **里面**，这样 ``info["episode"]["r"]``
    保持未归一化的真实回报，报表读到的就是原始数值。
    """
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    impl = resolve_vec_env(n_envs, vec_env)
    env_fns = [make_env_fn(stair, cfg, seed + i) for i in range(n_envs)]
    if impl == "dummy":
        venv: VecEnv = DummyVecEnv(env_fns)
    else:
        # Windows 的默认 start method 不可用；spawn 需要 worker 重新 import
        # 主模块，因此 env 工厂必须是模块级函数（见 _make_env）。
        venv = SubprocVecEnv(env_fns, start_method=start_method)
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
    actor_obs_index: Sequence[int] | None = None,
    tensorboard_log: str | Path | None = None,
    device: str = "cpu",
    verbose: int = 1,
) -> PPO:
    """构造 PPO。

    ``device`` 以 ``cuda`` 开头但当前 torch 没有 CUDA 时直接报错：SB3 的
    ``get_device`` 会静默回退 CPU，配置写着 cuda 却按 CPU 跑是最难发现的假象。
    ``log_std_init=-2.0``（归一化动作单位下 std ≈ 0.135）是折中：默认 0.0 让残差
    每步顶满 ``action_scale``、等于纯噪声；过小则残差恒零、退化成纯开环步态。
    ``batch_size`` 与样本数不整除时 SB3 会 warn —— 对只做 ``predict`` 的推理场景
    无意义，定向压掉。

    非对称 actor-critic：``actor_obs_index`` 非空时改用
    :class:`robotdog.asymmetric.AsymmetricActorCriticPolicy` —— actor 只看这些
    下标对应的观测维、critic 看全量（下标会写进 ``policy_kwargs``，作为
    checkpoint 自描述的一部分）。``None``（默认）⇒ 仍是 SB3 ``MlpPolicy``，
    与历史训练逐位一致（前向兼容的判据）。
    """
    if str(device).startswith("cuda") and not th.cuda.is_available():
        raise RuntimeError(
            "device 以 cuda 开头，但 torch.cuda.is_available() 为 False；"
            "当前 torch 是 CPU 版或驱动不可用。请安装 CUDA 版 torch，"
            "或把 ppo.device 改回 cpu。"
        )
    policy_class: Any = "MlpPolicy"
    policy_kwargs: dict[str, Any] = dict(
        net_arch=dict(pi=list(net_arch), vf=list(net_arch)),
        activation_fn=nn.Tanh,
        log_std_init=log_std_init,
    )
    if actor_obs_index is not None:
        # 非对称 AC：actor 只看 actor 维，critic 看全量（含特权块）。
        policy_class = AsymmetricActorCriticPolicy
        policy_kwargs["actor_obs_index"] = [int(i) for i in actor_obs_index]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*mini-batch size.*",
                                category=UserWarning)
        return PPO(
            policy_class,
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
            policy_kwargs=policy_kwargs,
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
    # 观测里有特权块时（非对称 AC 训练的权重）必须用同一个策略类重建，否则
    # actor/critic 的输入维数对不上；旧权重的 cfg 里特权块为 0 ⇒ None ⇒ MlpPolicy。
    actor_index = cfg.obs.actor_index if cfg.obs.privileged_frame_size > 0 else None
    model = build_ppo(venv, seed=0, device=device, net_arch=arch,
                      actor_obs_index=actor_index)
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
# 跨观测维度的 warm-start（对称 AC → 非对称 AC）
# --------------------------------------------------------------------------- #


def policy_state_dict(model_path: str | Path) -> dict[str, th.Tensor]:
    """读 SB3 ``.zip`` 里的 ``policy.pth``（纯权重，不建环境）。"""
    with zipfile.ZipFile(model_path) as zf:
        raw = zf.read("policy.pth")
    return th.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


def set_parameters_compat(model: PPO, model_path: str | Path, *,
                          old_obs_dim: int, new_obs_dim: int,
                          actor_index: Sequence[int] | None = None) -> str:
    """把「旧观测维」的权重装进「新观测维」的策略（actor 维不变、critic 变宽）。

    唯一支持的迁移是**非对称 AC 从对称权重起步**：

    * actor（``mlp_extractor.policy_net``）的输入维不变（两边的 actor 都只看
      ``actor_size`` 维）⇒ 原样搬；
    * critic（``mlp_extractor.value_net``）第一层从 ``old_obs_dim`` 变宽到
      ``new_obs_dim``：左侧旧列原样搬、右侧新增列**补零** —— 初始 value 与旧
      权重**逐位相同**，新特权输入的影响由训练学分。

    ``actor_index`` 是旧观测各维在新观测里的下标（``ObsConfig.actor_index``）。
    默认 ``None`` 表示「旧观测就是新观测的前缀」（``history_len == 1`` 的常见情形）；
    ``history_len > 1`` 时特权块在每帧末尾、actor 维不是前缀，必须传它，否则旧列会
    静默落到错误的帧/列上。

    其它任何形状变化都抛 ``ValueError``（不静默降级；也不支持窄化）。返回迁移描述串。
    """
    if int(new_obs_dim) <= int(old_obs_dim):
        raise ValueError(
            f"兼容迁移只支持观测增宽：{old_obs_dim} → {new_obs_dim}（窄化请改用旧配置）")
    if actor_index is not None:
        idx = [int(i) for i in actor_index]
        if len(idx) != int(old_obs_dim) or max(idx) >= int(new_obs_dim):
            raise ValueError(
                f"actor_index 与观测维度不匹配：len={len(idx)}, "
                f"old={old_obs_dim}, new={new_obs_dim}")
    else:
        idx = list(range(int(old_obs_dim)))
    old = policy_state_dict(model_path)
    new = model.policy.state_dict()
    extra = sorted(set(old) - set(new))
    if extra:
        raise ValueError(f"兼容迁移失败：旧权重多出未使用键 {extra}")
    merged: dict[str, th.Tensor] = {}
    expanded = False
    for key, target in new.items():
        if key not in old:
            raise ValueError(f"兼容迁移失败：旧权重缺少 {key}")
        src = old[key]
        if tuple(src.shape) == tuple(target.shape):
            merged[key] = src.to(target.device)
            continue
        if key == "mlp_extractor.value_net.0.weight":
            if src.shape[0] != target.shape[0] or (int(src.shape[1]) != int(old_obs_dim)
                                                   or int(target.shape[1]) != int(new_obs_dim)):
                raise ValueError(
                    f"兼容迁移失败：critic 首层 {tuple(src.shape)} → {tuple(target.shape)}"
                    f"（期望 (H,{old_obs_dim}) → (H,{new_obs_dim})）")
            wider = th.zeros_like(target)
            # 按 actor_index 放回旧列：h=1 时它就是前 old_obs_dim 列；h>1 时
            # 特权块夹在每帧末尾，必须用下标映射（与 restore_vecnormalize_compat 同口径）。
            wider[:, idx] = src.to(target.device)
            merged[key] = wider
            expanded = True
            continue
        raise ValueError(f"兼容迁移失败：{key} 形状 {tuple(src.shape)} → {tuple(target.shape)}")
    model.policy.load_state_dict(merged, strict=True)
    return f"{old_obs_dim}->{new_obs_dim}" if expanded else "exact"


def restore_vecnormalize_compat(venv: VecEnv, path: str | Path, *,
                                old_obs_dim: int, new_obs_dim: int,
                                actor_index: Sequence[int] | None = None
                                ) -> VecNormalize | None:
    """把旧观测维的 ``*_vecnormalize.pkl`` 统计扩到新观测维。

    旧权重是在 ``old_obs_dim`` 维观测上归一化训练的；新观测更宽（多出 critic
    特权块）。做法：旧统计按 ``actor_index`` 放回对应位置，新增维取均值 0 /
    方差 1（= 不放大，等训练期自己统计）。奖励统计（``ret_rms``）与维度无关，原样搬。
    """
    p = Path(path)
    if not p.is_file():
        return None
    vec = _find_vecnormalize(venv)
    if vec is None:
        return None
    # 不能走 VecNormalize.load(str(p), vec.venv)：它会校验「pkl 的观测空间」与
    # 「venv 的观测空间」形状一致，而这里恰恰是 53 维 pkl → 57 维 venv。
    # 直接反 pickle 取统计量（与 robotdog.policy.ObsNorm.from_path 同一套路），
    # 并立刻把需要的字段拷出来，避免 SB3 __getattr__ 递归。
    with open(p, "rb") as fh:
        loaded = pickle.load(fh)
    try:
        obs_rms = loaded.obs_rms
        ret_rms = loaded.ret_rms
        clip_obs = float(getattr(loaded, "clip_obs", 10.0))
        clip_reward = float(getattr(loaded, "clip_reward", 10.0))
        epsilon = float(getattr(loaded, "epsilon", 1e-8))
    finally:
        del loaded
    old_mean = np.asarray(obs_rms.mean, dtype=np.float64)
    old_var = np.asarray(obs_rms.var, dtype=np.float64)
    if old_mean.shape[0] != int(old_obs_dim):
        raise ValueError(f"观测统计维度不匹配：pkl 是 {old_mean.shape[0]} 维，"
                         f"期望旧权重 {old_obs_dim} 维")
    idx = (list(range(int(old_obs_dim))) if actor_index is None
           else [int(i) for i in actor_index])
    if len(idx) != int(old_obs_dim) or max(idx) >= int(new_obs_dim):
        raise ValueError(f"actor_index 与观测维度不匹配：len={len(idx)}, "
                         f"old={old_obs_dim}, new={new_obs_dim}")
    new_mean = np.zeros(int(new_obs_dim), dtype=np.float64)
    new_var = np.ones(int(new_obs_dim), dtype=np.float64)
    new_mean[idx] = old_mean
    new_var[idx] = old_var
    vec.obs_rms.mean = new_mean
    vec.obs_rms.var = new_var
    # ⚠ ``count`` 必须一起恢复，否则是**假 warm-start**（§23.34 同型）：
    # RunningMeanStd 的更新是 ``mean += delta * batch / (count + batch)``，
    # 新建 VecNormalize 的 count 只有 epsilon=1e-4 —— 第一次 reset（8 env）就以
    # 8/(8+1e-4)≈100% 的权重把刚装回的统计整份替换掉，日志却照样打印「装回」。
    # count 缺失/非正时**显式报错**，不静默退化成 0（否则又回到假 warm-start）。
    old_obs_count = getattr(obs_rms, "count", None)
    old_ret_count = getattr(ret_rms, "count", None)
    if old_obs_count is None or float(old_obs_count) <= 0.0:
        raise ValueError("pkl 缺少有效的 obs_rms.count，无法恢复真正的观测统计；"
                         "请确认这是 VecNormalize.save() 的原生产物。")
    if old_ret_count is None or float(old_ret_count) <= 0.0:
        raise ValueError("pkl 缺少有效的 ret_rms.count，无法恢复奖励统计。")
    vec.obs_rms.count = float(old_obs_count)
    vec.ret_rms.mean = np.asarray(ret_rms.mean, dtype=np.float64)
    vec.ret_rms.var = np.asarray(ret_rms.var, dtype=np.float64)
    vec.ret_rms.count = float(old_ret_count)
    vec.clip_obs = clip_obs
    vec.clip_reward = clip_reward
    vec.epsilon = epsilon
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


def warm_select_stages(n_stages: int) -> list[int]:
    """需要为「下一级」挑 warm-start 的级序号（**1-based**）：除最后一级外的所有级。

    训练循环用 ``enumerate(stages, start=1)``；第 i 级训完才为第 i+1 级选起点，
    所以可选的 i 是 ``1..n_stages-1``。最后一级之后没有下一级，**既不回放也不写
    ``warm_select``**（否则 Trace 里会出现一条从未发生的选取，复盘时被当成事实）。
    ``n_stages <= 1`` 时返回空列表（单级训练没有「下一级」可选）。
    """
    return list(range(1, max(int(n_stages), 1)))


def replay_success_rate(
    model_path: str | Path,
    *,
    cfg: EnvConfig | None = None,
    stair: StairSpec | None = None,
    episodes: int = 40,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, float]:
    """确定性回放 ``episodes`` 个回合，统计成功率/摔倒率/平均步数/平均前进。

    训练循环里「同级多个 checkpoint 挑最优起点」用（chain 复刻的 warm-start
    选取规则，见 ``doc/introduce_tactile.md`` §9）。走 ``load_policy`` 的同一套
    恢复路径（配置、网络结构、观测归一化），不另搭评估管线。

    口径与 ``play.py --mode eval --sampled-command`` 一致：不钉指令，沿用
    ``reset()`` 采样的指令流；策略确定性；回合上限 = episode_length_s / control_dt。
    """
    from robotdog.env import Go2StairsEnv
    from robotdog.policy import env_config_for_model, stair_from_height

    ckpt = Path(model_path)
    env_cfg = cfg or env_config_for_model(ckpt)
    spec = stair if stair is not None else stair_from_height(0.0)
    model, norm = load_policy(ckpt, cfg=env_cfg, stair=spec, device=device)
    env = Go2StairsEnv(stair=spec, cfg=env_cfg, seed=int(seed))
    max_steps = int(env_cfg.episode_length_s / env_cfg.control_dt)
    n_eps = max(1, int(episodes))
    succ = fall = 0
    total_steps = 0
    total_dist = 0.0
    try:
        for ep in range(n_eps):
            obs, _ = env.reset(seed=int(seed) + ep)
            x0 = float(env.base_pos[0])
            done = trunc = False
            n = 0
            info: dict = {}
            while not (done or trunc) and n < max_steps:
                x = obs if norm is None else norm.normalize(obs)
                action, _ = model.predict(x, deterministic=True)
                obs, _r, done, trunc, info = env.step(action)
                n += 1
            ok = bool(info.get("is_success", False))
            succ += int(ok)
            fall += int(bool(done) and not ok)
            total_steps += n
            total_dist += float(env.base_pos[0]) - x0
    finally:
        env.close()
        # ``load_policy`` 内部建了一个 VecEnv 并挂在 model 上；不关会**每候选泄漏
        # 一个**（16 级 × ~25 候选 × 100 回合可到 OOM，而 OOM 会伪装成 MJCF
        # 编译失败 —— experience.md 已记过两次同类事故）。
        try:
            model.get_env().close()
        except Exception:  # pragma: no cover - 防御：老 checkpoint 可能没有 env
            pass
    return {
        "episodes": float(n_eps),
        "success_rate": succ / n_eps,
        "fall_rate": fall / n_eps,
        "mean_steps": total_steps / n_eps,
        "mean_distance": total_dist / n_eps,
    }

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
    "resolve_vec_env",
    "policy_state_dict",
    "replay_success_rate",
    "warm_select_stages",
    "restore_vecnormalize",
    "restore_vecnormalize_compat",
    "save_vecnormalize",
    "set_parameters_compat",
]

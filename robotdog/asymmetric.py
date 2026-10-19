"""非对称 actor-critic：critic 读全量观测，actor 只读属于自己的那些维。

为什么需要
----------
本仓是**盲走**环境：策略部署到真机时只能依赖编码器 / IMU / 本体感觉估计 /
上层指令，不能有视觉、也不能有足端接触传感器（信息来源清单见 `doc/env.md`，
信息源审计见 `experience.md` §11）。但训练期的 **critic** 可以白拿仿真器真值
（足端接触、地形高度……）来降低价值估计方差 —— 这就是非对称 actor-critic：
actor 决定部署时的输出，critic 只在训练时存在。**特权信息只进 value 网络，
不改变 actor 的输入维数，因此不破坏盲走契约**（「真值只进奖励/终止，不进观测」
那条硬边界讲的是 actor 的观测；critic 的特权输入是 RL 的标准做法）。

实现方式（SB3 2.9 的内部结构）
----------------------------
`ActorCriticPolicy` 在 `share_features_extractor=True`（默认）时：
`features = FlattenExtractor(obs)` 是**完整**观测向量，然后
`forward` / `evaluate_actions` / `get_distribution` / `predict_values`
全部经 `self.mlp_extractor.forward_actor` / `forward_critic` 分叉。
因此**不需要重写任何 policy 方法**，只要换掉 `mlp_extractor`：

* `AsymmetricMlpExtractor.forward_actor` 先 `index_select` 只留 actor 维
  （帧优先条带切分，支持 `history_len > 1`；见 `ObsConfig.actor_index`）；
* `AsymmetricMlpExtractor.forward_critic` 用全部特征（含特权块）。

前向兼容
--------
`AsymmetricActorCriticPolicy` **只在观测里确有特权块时才被构造**
（`robotdog/training.py` 的 `build_ppo(actor_obs_index=None)`）。默认
`obs.privileged_foot_contact=false` ⇒ 走原来的 `MlpPolicy`，观测、网络、
checkpoint 与历史权重逐位不变。切分下标作为非持久 buffer 保存，不进
`state_dict`，所以 `set_parameters` 只搬真正的网络权重。
"""

from __future__ import annotations

from typing import Sequence

import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import get_device


def _build_mlp(input_dim: int, hidden_dims: Sequence[int],
               activation_fn: type[nn.Module]) -> tuple[nn.Sequential, int]:
    """与 SB3 `MlpExtractor` 同构的小工具：Linear + activation 逐层堆叠。

    返回 `(Sequential, 最后一层宽度)`；`hidden_dims` 为空时是 Identity，
    宽度等于 `input_dim`（SB3 同样如此）。
    """
    layers: list[nn.Module] = []
    last = int(input_dim)
    for width in hidden_dims:
        layers.append(nn.Linear(last, int(width)))
        layers.append(activation_fn())
        last = int(width)
    return nn.Sequential(*layers), last


class AsymmetricMlpExtractor(nn.Module):
    """actor 只看 `actor_index` 指定的特征维，critic 看全部特征。

    镜像 SB3 `MlpExtractor` 的公开形状（`policy_net` / `value_net` /
    `latent_dim_pi` / `latent_dim_vf` / `forward_actor` / `forward_critic`），
    这样 `ActorCriticPolicy._build` 里对 latent 维、正交初始化、state_dict 的处理
    全部照旧。
    """

    def __init__(
        self,
        feature_dim: int,
        actor_index: Sequence[int],
        net_arch: list[int] | dict[str, list[int]],
        activation_fn: type[nn.Module],
        device: th.device | str = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        index = th.as_tensor([int(i) for i in actor_index], dtype=th.long)
        # 非持久 buffer：不进 state_dict（否则 set_parameters 会因多出的键报错），
        # 但会随 module.to(device) 一起搬。
        self.register_buffer("actor_index", index, persistent=False)
        self.actor_feature_dim = int(index.numel())
        if isinstance(net_arch, dict):
            pi_dims = list(net_arch.get("pi", []))
            vf_dims = list(net_arch.get("vf", []))
        else:
            pi_dims = list(net_arch)
            vf_dims = list(net_arch)
        policy_net, last_pi = _build_mlp(self.actor_feature_dim, pi_dims, activation_fn)
        value_net, last_vf = _build_mlp(int(feature_dim), vf_dims, activation_fn)
        self.latent_dim_pi = last_pi
        self.latent_dim_vf = last_vf
        self.policy_net = policy_net.to(device)
        self.value_net = value_net.to(device)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """actor 只吃切出来的那些维（特权块永远进不来）。"""
        return self.policy_net(features.index_select(-1, self.actor_index))

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        """critic 吃全部特征（含特权块）。"""
        return self.value_net(features)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    """`ActorCriticPolicy` + 非对称 `mlp_extractor`。

    `actor_obs_index` 是 actor 在**整帧堆叠观测**里的下标（`ObsConfig.actor_index`）。
    为 `None` 或覆盖全部维时退回 SB3 默认行为（对称 AC），所以同一份代码既能
    跑旧配置也能跑新配置。
    """

    def __init__(self, *args, actor_obs_index: Sequence[int] | None = None,
                 **kwargs) -> None:
        # 必须在 super().__init__ 之前存好：BasePolicy.__init__ 会立刻调 _build，
        # 而 _build 会调 _build_mlp_extractor。
        self._actor_obs_index = (None if actor_obs_index is None
                                else tuple(int(i) for i in actor_obs_index))
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        index = self._actor_obs_index
        if index is None:
            # 对称 AC（默认路径）：调用方只在确有特权块时才传下标，None 走原路径。
            super()._build_mlp_extractor()
            return
        if len(index) >= int(self.features_dim):
            # 不能静默退回对称 AC：那会把本应只给 critic 的特权维喂给 actor
            # （盲走契约破了且没有任何报错）。传成「全量」一定是接线/配置错误。
            raise ValueError(
                f"actor_obs_index 覆盖了全部 {self.features_dim} 维观测（len={len(index)}）："
                f"这等于把 critic 的特权块也喂给 actor。请检查 "
                f"ObsConfig.privileged_foot_contact 与 build_ppo(actor_obs_index=...) 的接线。"
            )
        self.mlp_extractor = AsymmetricMlpExtractor(
            self.features_dim,
            index,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )


# 注：切分下标**只**经 `build_ppo` 的 ``policy_kwargs["actor_obs_index"]`` 落进
# checkpoint 的 ``data``（policy_kwargs 会被 SB3 一起保存）；加载侧（``load_policy``）
# 从 ``_env.json`` 的 ``ObsConfig`` 重新推导下标，不依赖 pickle 的策略对象 ——
# 所以这里不需要覆写任何构造参数钩子。


__all__ = ["AsymmetricActorCriticPolicy", "AsymmetricMlpExtractor"]

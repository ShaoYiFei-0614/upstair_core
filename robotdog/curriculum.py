"""第 6 步: 课程学习 (curriculum learning) + 训练回调。

用户规格 (m00001): 先平地训练 -> 3~8 cm 台阶 -> 8~15 cm -> 15~22 cm, 每阶段
成功率高就升难度。

* 每个阶段是一个 :class:`StageSpec`, 持有该阶段的地形和"毕业条件"(目标成功率 +
  最少回合数 + 时间步预算)。
* 阶段之间**串行**: 一个阶段 ``model.learn()`` 训到毕业或预算耗尽,
  ``CurriculumCallback`` 通过 ``_on_step`` 返回 ``False`` 提前结束, 再由
  :func:`stage_iterator` 切换。切换**不在 rollout 中间发生** —— 换地形要重编译
  MJCF, 中途换会让当次 rollout 的观测/状态不一致。
* 成功判据来自 info 的 ``is_success`` (见 ``Go2StairsEnv._success_reached``)。
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from robotdog.terrain import StairSpec, demo_stairs_spec, flat_spec

# --------------------------------------------------------------------------- #
# Stage definition
# --------------------------------------------------------------------------- #


def deterministic_success(model, spec: StairSpec, cfg: Any = None, *,
                          n_episodes: int = 10, seed: int = 12345) -> float | None:
    """在 ``spec`` 上跑**确定性**策略, 返回成功率。

    ``EpisodeStats.success_rate`` 统计的是 ``learn()`` 产出的 rollout, 而那些用
    **采样**动作 (PPO 每步给 12 个动作维度加 ``log_std`` ≈ 0.137)。那个数字衡量的是
    「探索噪声能不能活过楼梯」, 不是「策略能不能爬上去」, 会大幅低估交付策略。

    实测 (``course15``): 课程以 **10%** 关掉了 ``medium3`` (7.8 cm × 8), 而同一个
    checkpoint 确定性跑 15 回合是 **33%** —— 差距全来自动作噪声。没有这个第二数字,
    课程会把一个几乎解决的阶段当成翻不过去的墙, 把整个预算耗在上面。

    测不出来时返回 ``None``, 使测量失败永远不会打断训练。
    """
    # `get_env()` lives on a real SB3 learner; the unit tests drive `_on_step()`
    # with a hand-rolled stub that has no such method, so this too must be
    # defensive -- otherwise the measurement crashes the very callback it is
    # supposed to protect.  Asked *before* building the scene: no point paying a
    # model compile just to discover we cannot query the observation normaliser.
    try:
        venv = model.get_env()
    except Exception:  # pragma: no cover - defensive, never fatal
        return None
    obs_rms = getattr(venv, "obs_rms", None)
    clip_obs = float(getattr(venv, "clip_obs", 10.0))
    eps = float(getattr(venv, "epsilon", 1e-8))

    try:
        from robotdog.env import Go2StairsEnv

        env = Go2StairsEnv(stair=spec, cfg=cfg, seed=seed)
    except Exception:  # pragma: no cover - defensive, never fatal
        return None

    max_steps = int(env.cfg.episode_length_s / env.cfg.control_dt)
    wins = 0
    try:
        for ep in range(max(1, int(n_episodes))):
            obs, _ = env.reset(seed=seed + ep)
            terminated = truncated = False
            info: dict[str, Any] = {}
            n = 0
            while not (terminated or truncated) and n < max_steps:
                x = np.asarray(obs, dtype=np.float64)
                if obs_rms is not None:
                    x = (x - obs_rms.mean) / np.sqrt(obs_rms.var + eps)
                    x = np.clip(x, -clip_obs, clip_obs)
                action, _ = model.predict(x.astype(np.float32), deterministic=True)
                obs, _r, terminated, truncated, info = env.step(action)
                n += 1
            wins += int(bool(info.get("is_success", False)))
    except Exception:  # pragma: no cover - defensive, never fatal
        return None
    finally:
        env.close()
    return wins / max(1, int(n_episodes))


@dataclass
class StageSpec:
    """One rung of the curriculum ladder."""

    name: str
    label: str  # human-readable, Chinese
    spec: StairSpec
    target_success: float  # graduate when the rolling success rate exceeds this
    min_episodes: int  # ... but never before this many finished episodes
    max_timesteps: int  # hard budget; advancing anyway is opt-in
    # ... and never before this many environment steps have been consumed.
    #
    # WHY THIS EXISTS (measured, see README §3.3): with a strong open-loop gait
    # prior the *untrained* policy already clears the success target, so a stage
    # built only on ``min_episodes`` + ``target_success`` can graduate after a
    # single rollout. PPO needs ``n_steps * n_envs`` steps per update, so the
    # stage ends after 0~1 gradient updates and the network never leaves its
    # initialization -- which is exactly why the probed policy output was ~0.011
    # (i.e. "do nothing, let the prior walk"). A step floor guarantees that PPO
    # actually trains before the rung is declared passed.
    min_timesteps: int = 0
    description: str = ""

    def summary(self) -> str:
        s = self.spec
        extra = (f" 最少步数={self.min_timesteps:,}"
                 if self.min_timesteps > 0 else "")
        return (
            f"{self.name:<7} {self.label:<12} 台阶高={s.step_height * 100:5.1f}cm "
            f"踏面={s.tread * 100:5.1f}cm 级数={s.n_up:<2d} 顶端={s.crest_height:5.2f}m "
            f"| 目标成功率={self.target_success:.0%} 最少回合={self.min_episodes} "
            f"预算={self.max_timesteps:,}{extra}"
        )


def default_curriculum(step_heights: Sequence[float] = (0.05, 0.12, 0.19),
                       tread: float = 0.50, n_up: int = 8) -> list[StageSpec]:
    """The four stages from the spec, with fixed representative step heights.

    The user asked for bands (3~8 / 8~15 / 15~22 cm); we use the band midpoint
    because the terrain is recompiled per stage (``set_stair`` rebuilds the
    MJCF model, ~250 ms), so per-episode height randomization is not viable.
    """
    h_lo, h_mid, h_hi = step_heights
    return [
        StageSpec(
            name="flat",
            label="平地",
            # step_height=0.0 只是如实标注「平地没有台阶」。StairSpec 的默认值是
            # 0.05（为了让 demo_stairs_spec 的默认值统一），但对 n_up=0 的平地
            # 留 0.05 会让人误以为平地段有 5cm 台阶，也让 level 报表出现假高度。
            spec=flat_spec(n_up=0, n_down=0, step_height=0.0, tread=tread,
                           width=2.0, start_x=1.50),
            target_success=0.90,
            min_episodes=30,
            max_timesteps=250_000,
            description="平地行走, 建立稳定的步态与前进速度",
        ),
        StageSpec(
            name="easy",
            label="3~8cm 台阶",
            spec=demo_stairs_spec(step_height=h_lo, tread=tread, n_up=n_up,
                                  top_landing=1.20),
            target_success=0.80,
            min_episodes=40,
            max_timesteps=300_000,
            description="5 cm 台阶 (3~8 cm 区间中点), 学会抬腿跨越",
        ),
        StageSpec(
            name="medium",
            label="8~15cm 台阶",
            spec=demo_stairs_spec(step_height=h_mid, tread=tread, n_up=n_up,
                                  top_landing=1.20),
            target_success=0.65,
            min_episodes=40,
            max_timesteps=350_000,
            description="12 cm 台阶 (8~15 cm 区间中点)",
        ),
        StageSpec(
            name="hard",
            label="15~22cm 台阶",
            spec=demo_stairs_spec(step_height=h_hi, tread=tread, n_up=n_up,
                                  top_landing=1.20),
            target_success=0.50,
            min_episodes=40,
            max_timesteps=400_000,
            description="19 cm 台阶 (15~22 cm 区间中点)",
        ),
    ]


def mixed_stage_specs(spec: StairSpec, n_envs: int, lo_cm: float,
                      hi_cm: float) -> list[StairSpec]:
    """每个子环境一个 spec, 台阶高度在 ``[lo, hi]`` 上铺开。

    **为什么需要它 (实测)**: 把 8.0 -> 8.5 -> 9.0 -> 9.5 -> 10.0 cm 一级一级地爬,
    会让每一级成为 PPO 梯度唯一见到的地形, 策略**灾难性遗忘**已越过的台阶;
    每一级只在它训练的高度上最好, 且全都比 course18 在 7.5~8.0 cm 上更差。让一个
    阶段内各子环境处于不同高度, 能把整条带都放进每个梯度批次, 消除遗忘。

    **刻意不用**每回合随机化: ``set_stair`` 每次重编译 MJCF (~250 ms/子环境), 每个
    ``reset`` 都重编译会比训练本身更耗时。高度线性铺开且含中点, 避免批次被两端主导。
    """
    n_envs = max(1, int(n_envs))
    lo = min(float(lo_cm), float(hi_cm)) / 100.0
    hi = max(float(lo_cm), float(hi_cm)) / 100.0
    if n_envs == 1 or hi <= lo:
        return [replace(spec, step_height=hi)]
    span = hi - lo
    return [
        replace(spec, step_height=round(lo + span * i / (n_envs - 1), 6))
        for i in range(n_envs)
    ]


def mixed_grid_specs(spec: StairSpec, n_envs: int, *,
                     heights_cm: tuple[float, float],
                     treads: tuple[float, float]) -> list[StairSpec]:
    """每个子环境一个 spec, 铺在**台阶高 × 踏面**网格上。

    把 :func:`mixed_stage_specs` 扩展到第二个难度轴。踏面轴的实测依据
    (``course21`` final, 每格 20 个确定性回合; 通过缩放 ``n_up`` 把上升段固定在
    4.0 m, 免得 20 s 上限混淆结果): 踏面 35 cm -> 0%, 50 cm -> 55%, 65 cm -> 0%,
    80 cm -> 0% (10.0 cm 台阶)。

    ⇒ 更长的踏面**不会**多给机器人调整时间: 65/80 cm 崩到 0%, 35 cm 也崩。策略是围绕
    训练时的 50 cm 调好的, 残差步态与它相位锁定, 所以**改变**踏面是让起跨失步而不是
    给它余量。结论不是「用更长踏面」, 而是「别让策略依赖某一种踏面」: 让各子环境用不同
    踏面, 逼迫起跨对任意到达相位都成立 —— 这正是高立面起跨所缺的性质。

    两轴交叉成网格: 高度轴铺满全范围, 踏面轴占两个子环境 (理由同上), 于是 8 个子环境
    = 4 高度 × 2 踏面。``n_envs == 1`` 取右上角 (最高最长), 与
    :func:`mixed_stage_specs` 一致。
    """
    n_envs = max(1, int(n_envs))
    h_lo, h_hi = sorted(float(v) for v in heights_cm)
    t_lo, t_hi = sorted(float(v) for v in treads)
    h_lo, h_hi = h_lo / 100.0, h_hi / 100.0
    if n_envs == 1:
        return [replace(spec, step_height=h_hi, tread=t_hi)]

    # Split ``n_envs`` into heights x treads.  The **height** axis gets the
    # bulk: step height is the difficulty the band is meant to cover, and
    # ``mixed_stage_specs`` already established that the whole band has to sit
    # inside every gradient batch.  Two treads are enough to decorrelate the
    # gait phase, so unless the caller passed a degenerate tread range we spend
    # exactly two sub-envs on it.
    if t_hi <= t_lo:
        n_t = 1
    elif n_envs >= 4 and n_envs % 2 == 0:
        n_t = 2
    else:
        n_t = 1
    n_h = max(1, n_envs // n_t)

    def spread(lo: float, hi: float, n: int) -> list[float]:
        if n == 1 or hi <= lo:
            return [hi]
        return [round(lo + (hi - lo) * i / (n - 1), 6) for i in range(n)]

    hs = spread(h_lo, h_hi, n_h)
    ts = spread(t_lo, t_hi, n_t)
    specs: list[StairSpec] = []
    for h in hs:
        for t in ts:
            specs.append(replace(spec, step_height=h, tread=t))
    while len(specs) < n_envs:              # leftover envs: reuse the top corner
        specs.append(specs[-1])
    return specs[:n_envs]


def expand_stage(stage: StageSpec, n_levels: int = 4, *,
                 start_fraction: float = 0.4,
                 min_step_height: float = 0.02,
                 start_step_height: float | None = None,
                 n_up_start: int | None = None) -> list[StageSpec]:
    """把一个台阶阶段拆成**上升**的阶梯 (每级一个台阶高)。

    规格要的是区间 (3~8 / 8~15 / 15~22 cm) 且「成功率高就升难度」。直接上 5 cm 是很
    硬的第一级: 平地调好的先验只把最低足抬 ~20 mm, 所以 50 mm 立面就是墙。拆成多级
    先给策略能清掉的地形, 再抬立面。

    ``start_fraction`` 跳过最低那一级 (除非仍高于 ``min_step_height``); 高度从
    ``stage.spec.step_height * start_fraction`` 线性铺到满高, 最后一级**恰好**是
    ``stage.spec`` 以保持原合同。``max_timesteps`` 均分, 余数给最后一级。

    ``start_step_height`` (绝对米数) 覆盖第一级的 ``start_fraction``。这个覆盖有实际
    意义, 因为开环楼梯小跑对台阶高**不单调**: 2 cm -> 40%, 3 cm -> 40%, **4 cm ->
    100%**, 5 cm -> 20%, 6 cm -> 0%。固定 183 ms 的步态周期在 ~4 cm 处与 50 cm 踏面
    共振, 所以从 2 cm 起步的阶梯会从**先验自身能力之下**开始, 白费第一级。

    ``n_up_start`` 加上第二个独立难度轴: 立面的**数量**, 从 ``n_up_start`` 线性升到
    ``stage.spec.n_up``。这很关键, 因为终止成功奖励对整段楼梯是全有全无的 —— 8 级、每级
    存活 0.85 时成功率已经掉到 0.27, 正是前四次训练撞到的平台 (先验 25%, PPO 75k 步后
    25~32%)。从 2 级起步把同样的每级能力变成 ~72% 成功率, 于是策略能拿到密集的 +20
    终止奖励, 而不是稀有的那一个。
    """
    n_levels = max(1, int(n_levels))
    if n_levels == 1 or stage.spec.n_up == 0:
        return [stage]
    full = float(stage.spec.step_height)
    lo = max(float(min_step_height), full * float(start_fraction))
    if start_step_height is not None:
        lo = max(float(min_step_height), min(float(start_step_height), full))
    heights = [lo + (full - lo) * i / (n_levels - 1) for i in range(n_levels)]
    heights[-1] = full
    full_n_up = int(stage.spec.n_up)
    if n_up_start is None:
        n_ups = [full_n_up] * n_levels
    else:
        n_lo = max(1, min(int(n_up_start), full_n_up))
        n_ups = [int(round(n_lo + (full_n_up - n_lo) * i / (n_levels - 1)))
                 for i in range(n_levels)]
        n_ups[-1] = full_n_up
    per = max(1, stage.max_timesteps // n_levels)
    levels: list[StageSpec] = []
    for i, h in enumerate(heights):
        budget = stage.max_timesteps - per * (n_levels - 1) if i == n_levels - 1 else per
        levels.append(StageSpec(
            name=f"{stage.name}{i + 1}",
            label=f"{stage.label.rsplit(' ', 1)[0]} {h * 100:.1f}cm",
            spec=replace(stage.spec, step_height=h, n_up=n_ups[i]),
            # Early rungs must be cleared comfortably before the riser grows.
            target_success=max(stage.target_success, 0.80),
            min_episodes=max(8, stage.min_episodes // 2),
            max_timesteps=budget,
            description=(f"{stage.description} | 子级 {i + 1}/{n_levels}: "
                         f"台阶高 {h * 100:.1f}cm x {n_ups[i]} 级"),
        ))
    return levels


def expand_curriculum(stages: Sequence[StageSpec], n_levels: int = 4,
                      **kw) -> list[StageSpec]:
    """对每个台阶阶段套用 :func:`expand_stage` (平地原样通过)。

    给了 ``start_step_height`` 时它只钉住**第一个**台阶阶段的最低一级; 之后的台阶阶段
    从上一阶段的满高起步, 而不是重复它。否则 ``--stage easy,medium --sub-start-cm 4``
    会把 medium 展开成 4/6.7/9.3/12 cm, 白费第一块预算去重学 easy 已经解决的级。

    ``n_up_start`` 同理: 只有第一个台阶阶段做立面**数量**的爬坡, 到第二个台阶阶段时策略
    已经见过完整长度的楼梯了。
    """
    out: list[StageSpec] = []
    start = kw.get("start_step_height")
    n_up_start = kw.get("n_up_start")
    prev_full: float | None = None
    seen_stair = False
    for st in stages:
        stage_kw = dict(kw)
        if st.spec.n_up > 0:
            if start is not None:
                stage_kw["start_step_height"] = start if prev_full is None else prev_full
                prev_full = float(st.spec.step_height)
            if n_up_start is not None and seen_stair:
                stage_kw["n_up_start"] = None
            seen_stair = True
        out.extend(expand_stage(st, n_levels, **stage_kw))
    return out


# --------------------------------------------------------------------------- #
# Per-episode statistics
# --------------------------------------------------------------------------- #


class EpisodeStats:
    """Rolling window over finished episodes."""

    def __init__(self, window: int = 50) -> None:
        self.window = window
        self.success: deque[float] = deque(maxlen=window)
        self.fall: deque[float] = deque(maxlen=window)
        self.ret: deque[float] = deque(maxlen=window)
        self.length: deque[float] = deque(maxlen=window)
        self.n_total = 0

    def add(self, *, success: bool, fallen: bool, ret: float, length: float) -> None:
        self.success.append(1.0 if success else 0.0)
        self.fall.append(1.0 if fallen else 0.0)
        self.ret.append(float(ret))
        self.length.append(float(length))
        self.n_total += 1

    @property
    def n_window(self) -> int:
        return len(self.success)

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.success)) if self.success else 0.0

    @property
    def fall_rate(self) -> float:
        return float(np.mean(self.fall)) if self.fall else 0.0

    @property
    def mean_return(self) -> float:
        if not self.ret:
            return 0.0
        vals = np.asarray(self.ret, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if vals.size else 0.0

    @property
    def mean_length(self) -> float:
        if not self.length:
            return 0.0
        vals = np.asarray(self.length, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        return float(np.mean(vals)) if vals.size else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "episodes_total": self.n_total,
            "episodes_window": self.n_window,
            "success_rate": round(self.success_rate, 4),
            "fall_rate": round(self.fall_rate, 4),
            "mean_return": round(self.mean_return, 3),
            "mean_length": round(self.mean_length, 1),
        }


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #


class CurriculumCallback(BaseCallback):
    """Track success rate for the current stage and stop ``learn()`` on graduation."""

    def __init__(
        self,
        stage: StageSpec,
        *,
        window: int = 50,
        advance_on_budget: bool = True,
        print_every_episodes: int = 20,
        verbose: int = 1,
        cfg: Any = None,
        det_episodes: int = 10,
        stage_index: int = 0,
        randomization_on: bool = False,
        dr_ramp_steps: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.stage = stage
        self.stats = EpisodeStats(window=window)
        self.advance_on_budget = advance_on_budget
        self.print_every_episodes = print_every_episodes
        # Deterministic end-of-stage measurement (see ``deterministic_success``).
        self.cfg = cfg
        self.det_episodes = int(det_episodes)
        self.stage_index = int(stage_index)
        self.success_det: float | None = None
        # Domain-randomization severity ramp (step 8).  ``randomization_on`` says
        # the driver has already switched the knobs on for this stage; the ramp
        # then walks ``randomize_scale`` 0 -> 1 so the policy first solves the
        # stage on the nominal model and only then meets the perturbations.
        # Without this the full range lands on step one and the stage never
        # converges: measured, un-ramped DR training plateaued at 26% sampled
        # success over 368k steps against a 65% target.
        self.randomization_on = bool(randomization_on)
        self.dr_ramp_steps = int(dr_ramp_steps)
        self._dr_scale = 0.0
        # Results, read by the training driver after learn() returns.
        self.graduated = False
        self.stop_reason = "running"
        self.start_timesteps = 0
        self._floor_warned = False
        self._t0 = time.time()

    # ------------------------------------------------------------ life cycle
    def _on_training_start(self) -> None:
        self.start_timesteps = self.model.num_timesteps
        self._floor_warned = False
        self._t0 = time.time()
        # Start the stage on the nominal model and let the ramp introduce the
        # perturbations (``dr_ramp_steps < 0`` disables the ramp on purpose).
        if self.randomization_on:
            self._dr_scale = 1.0 if self.dr_ramp_steps < 0 else 0.0
            self.training_env.env_method(
                "set_domain_randomization", scale=self._dr_scale
            )
        if self.verbose:
            print(f"\n[课程] >>> 阶段 '{self.stage.name}' ({self.stage.label}) 开始")
            print(f"[课程]     {self.stage.description}")
            print(f"[课程]     {self.stage.summary()}")
            if self.randomization_on:
                span = (self._dr_ramp_span()
                        if self.dr_ramp_steps >= 0 else 0)
                print(f"[课程]     域随机化 ramp: 严重度 0.0 -> 1.0 "
                      f"({'立即全量' if span == 0 else f'{span:,} 步'})")

    def _dr_ramp_span(self) -> int:
        """Timesteps over which ``randomize_scale`` walks 0 -> 1.

        Defaults to the stage's minimum budget: the ramp then finishes exactly
        when the stage becomes *allowed* to graduate, so severity is only ever
        raised on a policy that has already met the stage's floor.
        """
        if self.dr_ramp_steps > 0:
            return self.dr_ramp_steps
        return max(1, int(self.stage.min_timesteps))

    def _on_step(self) -> bool:
        dones = np.asarray(self.locals.get("dones", []), dtype=bool).ravel()
        infos = self.locals.get("infos", [])
        n_envs = len(dones)

        for i in range(n_envs):
            info = infos[i] if i < len(infos) else {}
            if not dones[i]:
                continue
            success = bool(info.get("is_success", False))
            fallen = bool(info.get("fallen", False))
            # VecMonitor supplies the authoritative episode return/length; the
            # fallbacks only matter if the monitor is absent.
            ep = info.get("episode") or {}
            ret = float(ep.get("r", float("nan")))
            length = float(ep.get("l", float("nan")))
            self.stats.add(success=success, fallen=fallen, ret=ret, length=length)
            # ``print_every_episodes <= 0`` means "never print the periodic line"
            # (guard against 0 so the modulo cannot raise ZeroDivisionError).
            if (
                self.verbose
                and self.print_every_episodes > 0
                and self.stats.n_total % self.print_every_episodes == 0
            ):
                el = time.time() - self._t0
                steps = self.num_timesteps - self.start_timesteps
                print(
                    f"[课程] {self.stage.name:<6} 回合={self.stats.n_total:>4d} "
                    f"成功率={self.stats.success_rate:6.1%} "
                    f"摔倒率={self.stats.fall_rate:6.1%} "
                    f"均回报={self.stats.mean_return:8.2f} "
                    f"均步数={self.stats.mean_length:6.1f} "
                    f"| steps={steps:>8,} {steps / max(el, 1e-9):7.0f} steps/s",
                    flush=True,
                )

        if self.stats.n_window:
            self.logger.record("curriculum/success_rate", self.stats.success_rate)
            self.logger.record("curriculum/fall_rate", self.stats.fall_rate)
            self.logger.record("curriculum/mean_return", self.stats.mean_return)
            self.logger.record("curriculum/mean_ep_length", self.stats.mean_length)
            self.logger.record("curriculum/episodes", self.stats.n_total)
            self.logger.record(
                "curriculum/stage_steps",
                self.num_timesteps - self.start_timesteps,
            )

        steps = self.num_timesteps - self.start_timesteps

        # Domain-randomization severity ramp.  ``_dr_scale`` is monotonically
        # non-decreasing and saturates at 1.0, so re-issuing it only on change
        # keeps this to one env_method per step and never lowers severity mid
        # stage (lowering it would undo the robustness just learned).
        if self.randomization_on and self.dr_ramp_steps >= 0:
            span = self._dr_ramp_span()
            scale = float(min(1.0, steps / span))
            if scale > self._dr_scale:
                self._dr_scale = scale
                self.training_env.env_method(
                    "set_domain_randomization", scale=scale
                )
            self.logger.record("curriculum/dr_scale", self._dr_scale)

        steps_ok = steps >= self.stage.min_timesteps
        ready = (
            self.stats.n_total >= self.stage.min_episodes
            and self.stats.success_rate >= self.stage.target_success
            and steps_ok
        )
        # 只在「成功率已达标但这个阶段还不许毕业」时提示一次, 免得日志刷屏。
        if (not steps_ok and self.verbose and not self._floor_warned
                and self.stats.n_total >= self.stage.min_episodes
                and self.stats.success_rate >= self.stage.target_success):
            self._floor_warned = True
            print(
                f"[课程] {self.stage.name}: 成功率 {self.stats.success_rate:.1%} 已达标, "
                f"但最少训练步数 {self.stage.min_timesteps:,} 未满 "
                f"(当前 {steps:,}) -- 继续训练以让 PPO 真正学到东西",
                flush=True,
            )
        if ready:
            self.graduated = True
            self.stop_reason = (
                f"毕业: 成功率 {self.stats.success_rate:.1%} >= "
                f"目标 {self.stage.target_success:.0%} (共 {self.stats.n_total} 回合)"
            )
        elif steps >= self.stage.max_timesteps:
            if self.advance_on_budget:
                self.graduated = False
                self.stop_reason = (
                    f"预算耗尽({self.stage.max_timesteps:,} 步): 成功率 "
                    f"{self.stats.success_rate:.1%} < 目标 {self.stage.target_success:.0%}"
                    f" -- 仍推进 (advance_on_budget=True)"
                )
            else:
                self.graduated = False
                self.stop_reason = (
                    f"预算耗尽({self.stage.max_timesteps:,} 步): 成功率 "
                    f"{self.stats.success_rate:.1%} 未达标, 停留在本阶段"
                )

        if self.stop_reason != "running":
            # ⭐ 把「部署时的成功率」也量出来。``EpisodeStats.success_rate`` 统计的是
            # ``learn()`` 采样出来的**随机**动作 (12 个动作维上 log_std≈0.137), 它
            # 会大幅低估真正部署的策略。实测 course15: 课程在 medium3 (7.8cm x 8)
            # 结束时报 10%, 而同一个 checkpoint 用确定性动作跑 15 回合是 33% —— 差
            # 的全是动作噪声。没有这个数字, 课程会把「其实已经会了」的阶段当成墙,
            # 一路把预算烧在它上面。
            self.success_det = deterministic_success(
                self.model, self.stage.spec, self.cfg,
                n_episodes=self.det_episodes, seed=12345 + self.stage_index,
            )
            if self.verbose and self.success_det is not None:
                print(f"[课程] {self.stage.name}: 确定性成功率 {self.success_det:.1%}"
                      f" (随机动作下 {self.stats.success_rate:.1%}, "
                      f"{self.det_episodes} 回合)", flush=True)
            if self.verbose:
                print(f"[课程] <<< 阶段 '{self.stage.name}' 结束: {self.stop_reason}",
                      flush=True)
            return False
        return True

    # ------------------------------------------------------------- reporting
    def result(self) -> dict[str, Any]:
        # The terrain *shape* is recorded next to the height: a stage is
        # defined by (step_height, n_up), and without n_up a later
        # ``render_checkpoint.py`` / eval run cannot rebuild the same terrain
        # from summary.json alone.
        return {
            "stage": self.stage.name,
            "label": self.stage.label,
            "step_height": self.stage.spec.step_height,
            "n_up": int(self.stage.spec.n_up),
            "tread": float(self.stage.spec.tread),
            "crest_height": float(self.stage.spec.crest_height),
            "target_success": self.stage.target_success,
            "success_rate_deterministic": (
                None if self.success_det is None else round(self.success_det, 4)
            ),
            "graduated": self.graduated,
            "stop_reason": self.stop_reason,
            "timesteps": int(self.num_timesteps - self.start_timesteps),
            **self.stats.as_dict(),
        }


class PeriodicCheckpointCallback(BaseCallback):
    """Save checkpoints by *total* env steps (unambiguous vs save_freq semantics).

    Also mirrors ``VecNormalize`` statistics when present, so a checkpoint is
    loadable on its own. Keeps only the newest ``keep`` numbered checkpoints.
    """

    def __init__(
        self,
        save_dir: str | Path,
        save_every: int,
        *,
        name_prefix: str = "ckpt",
        keep: int = 5,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = int(save_every)
        self.name_prefix = name_prefix
        self.keep = int(keep)
        self._last_save = -1
        self.saved: list[str] = []

    def _on_training_start(self) -> None:
        self._last_save = self.model.num_timesteps
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save < self.save_every:
            return True
        self._last_save = self.num_timesteps
        path = self.save_dir / f"{self.name_prefix}_{self.num_timesteps}_steps"
        self.model.save(str(path))
        self._save_vecnormalize(path)
        self.saved.append(str(path) + ".zip")
        if self.verbose:
            print(f"[检查点] 已保存 {path.name}.zip", flush=True)
        self._prune()
        return True

    def _save_vecnormalize(self, path: Path) -> None:
        vec = getattr(self.model, "get_vec_normalize_env", lambda: None)()
        if vec is not None:
            vec.save(str(path.with_suffix("")) + "_vecnormalize.pkl")

    def _prune(self) -> None:
        """Delete only checkpoints this run created, keeping the newest ``keep``."""
        if self.keep <= 0:
            return
        existing = sorted(
            self.save_dir.glob(f"{self.name_prefix}_*_steps.zip"),
            key=lambda p: int(p.stem.split("_")[-2]),
        )
        for old in existing[: max(0, len(existing) - self.keep)]:
            for partner in (
                old,
                old.with_name(old.stem + "_vecnormalize.pkl"),
            ):
                try:
                    partner.unlink()
                except FileNotFoundError:
                    pass


class BestModelCallback(BaseCallback):
    """Save a ``best_<stage>`` checkpoint whenever the rolling success rate improves.

    ``floor`` lets a driver carry the best rate across several curriculum stages:
    a fresh instance starts with ``best = -1.0``, so without a floor the first
    success of any *later* (harder) stage would overwrite the file -- and a 3%
    success rate on 19 cm stairs would replace a 100% flat policy.  Pass the
    run-wide best as ``floor`` to keep the file meaningful.
    """

    def __init__(self, save_path: str | Path, monitor: CurriculumCallback,
                 *, floor: float = -1.0, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.monitor = monitor
        self.floor = float(floor)
        self.best = -1.0

    def _on_step(self) -> bool:
        s = self.monitor.stats
        if s.n_total < self.monitor.stage.min_episodes:
            return True
        threshold = max(self.best, self.floor)
        if s.success_rate > threshold + 1e-9:
            self.best = s.success_rate
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.save_path))
            vec = getattr(self.model, "get_vec_normalize_env", lambda: None)()
            if vec is not None:
                vec.save(str(self.save_path.with_suffix("")) + "_vecnormalize.pkl")
            if self.verbose:
                print(f"[最佳] 成功率 {self.best:.1%} -> 保存 {self.save_path.name}",
                      flush=True)
        return True


class WallClockLimitCallback(BaseCallback):
    """Abort ``learn()`` after a wall-clock budget (safety net for long runs)."""

    def __init__(self, max_seconds: float, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.max_seconds = float(max_seconds)
        self._t0 = 0.0

    def _on_training_start(self) -> None:
        self._t0 = time.time()

    def _on_step(self) -> bool:
        if time.time() - self._t0 > self.max_seconds:
            if self.verbose:
                print(f"[限时] 超过 {self.max_seconds:.0f}s, 停止本阶段训练", flush=True)
            return False
        return True


# --------------------------------------------------------------------------- #
# Driver helper
# --------------------------------------------------------------------------- #


def select_stages(all_stages: Sequence[StageSpec], names: Sequence[str] | str) -> list[StageSpec]:
    """Pick stages by name, preserving curriculum order.

    ``names`` may be a sequence of stage names or a comma-separated string
    (``"flat,easy1"``).  A bare string used to be iterated character by
    character -- ``"flat,easy"`` then raised ``未知阶段 [',', 'a', 'e', ...]``,
    which is a confusing way to report a one-character typo.
    """
    if isinstance(names, str):
        names = [part.strip() for part in names.split(",") if part.strip()]
    if not names or "all" in names:
        return list(all_stages)
    wanted = set(names)
    picked = [s for s in all_stages if s.name in wanted]
    missing = wanted - {s.name for s in picked}
    if missing:
        raise ValueError(
            f"未知阶段 {sorted(missing)}; 可选 {[s.name for s in all_stages]}"
        )
    return picked


def write_curriculum_report(path: str | Path, results: list[dict[str, Any]],
                            extra: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stages": results}
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

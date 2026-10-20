"""训练盲走楼梯策略。

按 ``config.yaml`` 的 ``curriculum`` 段串行训练：默认是
**平地 → 3 → 6 → 9 → 12 → 15 cm**。每一级只对应一种台阶几何，训满该级的步数预算后
保存权重，下一级从上一级的权重 warm-start（``PPO.set_parameters``），所以策略逐级迁移。

用法::

    # 全部超参在 config.yaml 里，脚本只认运行名与配置文件
    .venv\\Scripts\\python.exe scripts\\train.py
    .venv\\Scripts\\python.exe scripts\\train.py --run-name course2

产物落在 ``runs/<名字>/``：

* ``model_final.zip`` / ``model_final_vecnormalize.pkl`` —— 最后一级的权重与观测统计
* ``checkpoints/<阶段>_final.zip`` —— 每一级的权重（也带 ``_vecnormalize.pkl``）
* ``summary.json`` —— 课程定义、每级实测结果、完整 ``EnvConfig``（供 play 恢复）
* ``tensorboard/`` —— PPO 标量

**观测归一化必须跟权重一起用**：``*_vecnormalize.pkl`` 与 ``*.zip`` 是成对的，
play 会按同名自动配对。
"""

from __future__ import annotations

import shutil
import sys
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from robotdog.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    config_to_env,
    config_to_ppo,
    config_to_run,
    config_to_stages,
    load_config,
    n_envs as cfg_n_envs,
    vec_env as cfg_vec_env,
)
from robotdog.curriculum import (  # noqa: E402
    EpisodeStats,
    RewardTermsMeter,
    StageResult,
    StageSpec,
    format_curriculum,
    format_seconds,
    write_summary,
)
from robotdog.env import EnvConfig  # noqa: E402
from robotdog.policy import env_json_path, observation_dim, save_env_json  # noqa: E402
from robotdog.tools import console  # noqa: E402
from robotdog.training import (  # noqa: E402
    RunDir,
    build_ppo,
    make_vec_env,
    resolve_vec_env,
    restore_vecnormalize,
    restore_vecnormalize_compat,
    save_vecnormalize,
    set_parameters_compat,
)

#: 只打印进度，不训练的最小步数（避免把 0 步的配置跑成死循环）。
MIN_STAGE_STEPS = 1


def warm_norm_path(warm_start: str | Path) -> Path:
    """warm-start 权重旁边那份观测统计（``<同目录同名前缀>_vecnormalize.pkl``）。"""
    p = Path(warm_start)
    return p.with_name(p.stem + "_vecnormalize.pkl")


def kw_call(venv, method: str, **kwargs):
    """用**关键字**调用子环境方法（``venv.env_method`` 的薄封装）。

    ``env_method`` 的位置实参按形参顺序绑定 —— 凡「多个同类型可选参数」的
    子环境方法一律走这里，让参数名对不上变成 ``TypeError`` 而不是静默传错值。
    """
    return venv.env_method(method, **kwargs)


# --------------------------------------------------------------------------- #
# 回调
# --------------------------------------------------------------------------- #


class ProgressCallback(BaseCallback):
    """有新回合结束时（且距上次打印至少 ``log_every`` 次更新）打印一行进度。

    ``reward_keys`` 非空时，额外按同一窗口打印「每步奖励分量均值」，并把同样的
    均值追加到 ``terms_path``（默认 ``runs/<run>/reward_terms.jsonl``）：
    ``{"step": int, "stage": str, "terms": {name: float}}``。这是训练期的逐项明细，
    与 ``play.py`` 的回合级聚合互补；空列表 = 不记录（训练行为不变）。
    """

    def __init__(self, *, log_every: int = 20, window: int = 50,
                 reward_keys: Sequence[str] = (), terms_path: Path | None = None,
                 stage: str = "", verbose: int = 1) -> None:
        super().__init__(verbose)
        self.log_every = max(1, int(log_every))
        self.stats = EpisodeStats(window=window)
        self.terms = RewardTermsMeter(keys=tuple(reward_keys))
        self.terms_path = Path(terms_path) if terms_path is not None else None
        self.stage = str(stage)
        self._last_episodes = 0
        self._last_print = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        self.stats.add_infos(infos)
        self.terms.add_infos(infos)
        new_episode = self.stats.n_total != self._last_episodes
        due = self.n_calls - self._last_print >= self.log_every
        if due and new_episode:
            self._last_episodes = self.stats.n_total
            self._last_print = self.n_calls
            print(f"    [{self.num_timesteps:>9,} 步] {self.stats.format_line()}",
                  flush=True)
            means = self.terms.drain()
            if means:
                summary = "  ".join(f"{k}={v:+.3f}" for k, v in means.items())
                print(f"    [奖励/步] {summary}", flush=True)
                if self.terms_path is not None:
                    self.terms_path.parent.mkdir(parents=True, exist_ok=True)
                    record = {"step": int(self.num_timesteps),
                              "stage": self.stage,
                              "terms": {k: round(float(v), 6)
                                        for k, v in means.items()}}
                    with self.terms_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True


class CheckpointCallback(BaseCallback):
    """每 ``every`` 步存 ``checkpoints/<阶段>/<阶段>_<步数>_steps.zip``，
    连带 ``_vecnormalize.pkl`` 与 ``_env.json``。

    ``_env.json`` 必须写：它记录「这个权重当时跑在什么配置下」，而
    ``gait_prior`` / ``action_scale`` 这类字段不进观测，任何按维度反推的方案
    都恢复不了它们（experience.md §20.7）。
    """

    def __init__(self, directory: Path, stage: str, *, every: int,
                 env_cfg: EnvConfig | None = None, step_height: float | None = None,
                 verbose: int = 0) -> None:
        super().__init__(verbose)
        self.directory = Path(directory)
        self.stage = stage
        self.every = max(1, int(every))
        self.env_cfg = env_cfg
        self.step_height = step_height
        self._last = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last < self.every:
            return True
        self._last = self.num_timesteps
        self.directory.mkdir(parents=True, exist_ok=True)
        base = self.directory / f"{self.stage}_{self.num_timesteps}_steps"
        self.model.save(str(base))
        save_vecnormalize(self.model, base)
        if self.env_cfg is not None:
            # ⚠ **这里写的是 ``live_cfg``（含本级 ``reward_overrides``），而
            # `<阶段>_final_env.json`（下面 ``save_env_json(target, self.cfg, …)``）
            # 写的是 ``config.yaml`` 的**基线**。同一个 run 目录里两份 ``_env.json``
            # 会对奖励字段给出**不同的值**，看起来像"覆盖没生效"或"中途被改了"。
            # 两边都是**故意**的：``_env.json`` 按设计只记基线（下一级 warm-start
            # 要"先叠基线再叠本级键"，见 ``env.py:set_reward_overrides``），
            # 而逐 checkpoint 的这份用来事后查"那一刻环境到底长什么样"。
            # ⇒ **复现某一级的奖励口径，唯一权威是 ``plan.json``**（train.py 课程循环）。
            save_env_json(base, self.env_cfg, step_height=self.step_height,
                          stage=self.stage)
        return True


# --------------------------------------------------------------------------- #
# 训练者
# --------------------------------------------------------------------------- #


class Trainer:
    """实现 :class:`robotdog.curriculum.StageTrainer` 协议。

    所有超参来自 ``config.yaml``，阶段特有的步数与学习率来自 :class:`StageSpec`。
    """

    def __init__(self, cfg: EnvConfig, rundir: RunDir, *, run: dict[str, Any],
                 ppo: dict[str, Any], n_envs: int, vec_env: str = "auto") -> None:
        self.cfg = cfg
        self.rundir = rundir
        self.run = run
        self.ppo = ppo
        self.n_envs = n_envs
        self.vec_env = vec_env
        self.last_stats = EpisodeStats(window=200)
        # ⭐ config.yaml 的奖励基线（没叠任何课程覆盖的那一份）：
        # make_env 每级开始前把它抄回子环境，保证覆盖是「从基线叠」。
        self._reward_baseline = copy.deepcopy(cfg.reward)
        # 「阶段名 → 该级实际生效的指令量程与奖励覆盖」，训练结束后写到
        # ``runs/<run>/plan.json``。见 ``make_env`` 里的说明。
        self.plans: dict[str, dict[str, Any]] = {}

    # ---- 每个阶段一个独立的向量环境（地形烘焙在编译后的 MJCF 里）----------
    def make_env(self, stage: StageSpec, *, warm_start: str | None = None):
        # ⭐ 三条纪律（违反的方式都是静默出错）：
        # ① 每级深拷贝 cfg —— 下发全是就地改写，共享 cfg 会让设置跨级累积；
        # ② 奖励覆盖先叠基线再叠本级键 —— 否则上一级覆盖过的键残留到这一级；
        # ③ 指令量程无条件下发 —— 量程为 0 是「不练这个方向」的有意义设置，
        #    守卫会让该级继承上一级的范围。且必须走关键字（见 kw_call）。
        stage_cfg = copy.deepcopy(self.cfg)
        venv = make_vec_env(
            stage.spec,
            n_envs=self.n_envs,
            cfg=stage_cfg,
            seed=int(self.run["seed"]),
            norm_reward=bool(self.ppo.get("norm_reward", True)),
            vec_env=self.vec_env,
        )
        # 指令课程：无条件下发这一级的采样范围（观测那 3 维恒定，PPO 结构不用重建）。
        kw_call(venv, "set_commands",
                vy_max=stage.command_vy_max,
                yaw_max=stage.command_yaw_max,
                vx_min=stage.command_vx[0],
                vx_max=stage.command_vx[1])
        # 奖励课程：先叠基线、再叠本级覆盖（见 make_env 注释 ②）。
        venv.env_method("set_reward_overrides",
                        dict(asdict(self._reward_baseline)))
        if stage.reward_overrides:
            venv.env_method("set_reward_overrides", dict(stage.reward_overrides))
        # 记录这一级实际生效的设置：_env.json / summary.json 里只有 config.yaml
        # 的基线，复现某一级必须读这份实施记录（plan.json）。
        self.plans[stage.name] = {
            "label": stage.label,
            "step_height": float(stage.step_height),
            "timesteps": int(stage.timesteps),
            "command_vx": [float(v) for v in stage.command_vx],
            "command_vy_max": float(stage.command_vy_max),
            "command_yaw_max": float(stage.command_yaw_max),
            "reward_overrides": {k: float(v)
                                 for k, v in dict(stage.reward_overrides).items()},
        }
        return venv

    def train_stage(self, stage: StageSpec, venv, *, warm_start: str | None = None,
                    warm_norm: str = "auto",
                    tensorboard_log: str | Path | None = None, **_: Any):
        arch = tuple(self.ppo.get("net_arch", (512, 256)))
        model = build_ppo(
            venv,
            seed=int(self.run["seed"]),
            learning_rate=stage.learning_rate,
            n_steps=int(self.ppo.get("n_steps", 256)),
            batch_size=int(self.ppo.get("batch_size", 512)),
            n_epochs=int(self.ppo.get("n_epochs", 5)),
            gamma=float(self.ppo.get("gamma", 0.99)),
            gae_lambda=float(self.ppo.get("gae_lambda", 0.95)),
            clip_range=float(self.ppo.get("clip_range", 0.2)),
            ent_coef=float(self.ppo.get("ent_coef", 0.002)),
            vf_coef=float(self.ppo.get("vf_coef", 0.5)),
            max_grad_norm=float(self.ppo.get("max_grad_norm", 1.0)),
            target_kl=self.ppo.get("target_kl", 0.03),
            log_std_init=float(self.ppo.get("log_std_init", -2.0)),
            net_arch=arch,
            # 非对称 AC：观测里有特权块时 actor 只看 actor 维（critic 看全量）。
            # 默认 cfg 没有特权块 ⇒ None ⇒ 仍是 MlpPolicy，与历史逐位一致。
            actor_obs_index=(self.cfg.obs.actor_index
                             if self.cfg.obs.privileged_frame_size > 0 else None),
            tensorboard_log=tensorboard_log,
            device=str(self.ppo.get("device", "cpu")),
        )
        if warm_start:
            # 观测维变了（对称 AC 53 维 → 非对称 AC 57 维）时必须走兼容迁移：
            # actor 维不变、原样搬；critic 首层左侧旧列搬、新增特权列补零
            # （初始 value 与旧权重逐位相同，新输入的影响由训练学）。
            # 正常同维时行为与旧实现完全一致。
            new_dim = int(venv.observation_space.shape[0])
            old_dim = observation_dim(warm_start)
            if old_dim is not None and int(old_dim) != new_dim:
                note = set_parameters_compat(
                    model, warm_start, old_obs_dim=int(old_dim),
                    new_obs_dim=new_dim,
                    actor_index=(self.cfg.obs.actor_index
                                 if self.cfg.obs.privileged_frame_size > 0 else None))
                print(f"    [warm-start] 兼容迁移 {Path(warm_start).name}："
                      f"观测 {note} 维（actor 不变，critic 首层补零）",
                      flush=True)
            else:
                model.set_parameters(str(warm_start))
                print(f"    [warm-start] 从 {Path(warm_start).name} 继承权重",
                      flush=True)
            # ⭐ 观测归一化统计必须跟权重一起过来，否则 warm-start 是**假的**：
            #    新建的 venv 里 VecNormalize 是全 0/1 的初始统计，而权重是在
            #    **归一化之后**的观测上训出来的 ⇒ 前几十万步全花在重新适应输入
            #    分布上。实测（2026-10-17，R1~R4 的 A/B 起步）：六臂全部
            #    摔倒率 100%、回合长度 ~150 步，而同一份权重直接回放能走满 1000 步。
            #    统计缺失时只警告不中止（旧行为可用 ``--warm-norm off`` 显式要回）。
            if warm_norm != "off":
                npath = warm_norm_path(warm_start)
                if npath.is_file():
                    if old_dim is not None and int(old_dim) != new_dim:
                        ok = restore_vecnormalize_compat(
                            venv, npath, old_obs_dim=int(old_dim),
                            new_obs_dim=new_dim,
                            actor_index=self.cfg.obs.actor_index) is not None
                    else:
                        ok = restore_vecnormalize(venv, npath) is not None
                    print(f"    [warm-start] {'装回' if ok else '⚠ 装不回'}"
                          f"观测统计 {npath.name}", flush=True)
                elif warm_norm == "on":
                    raise SystemExit(f"--warm-norm on，但找不到 {npath}")
                else:
                    print(f"    [warm-start] ⚠ 没有 {npath.name}，观测统计从零开始",
                          flush=True)

        reward_keys = tuple(self.run.get("log_reward_terms") or ())
        callbacks = [ProgressCallback(
            log_every=int(self.run["log_every"]),
            reward_keys=reward_keys,
            terms_path=(self.rundir.root / "reward_terms.jsonl"
                        if reward_keys else None),
            stage=stage.name,
        )]
        every = int(self.run["save_every"])
        if every > 0:
            # 中间检查点带 _env.json：从子环境读回这一级实际生效的配置
            # （已叠好量程与覆盖），而不是 config.yaml 的基线。
            try:
                live_cfg = venv.get_attr("cfg")[0]
            except Exception:  # noqa: BLE001 - 取不到就不写，不影响训练
                live_cfg = None
            callbacks.append(
                CheckpointCallback(self.rundir.checkpoints / stage.name, stage.name,
                                   every=every, env_cfg=live_cfg,
                                   step_height=stage.step_height)
            )
        t0 = time.time()
        model.learn(total_timesteps=stage.timesteps, callback=callbacks,
                    reset_num_timesteps=True)
        sps = stage.timesteps / max(1e-6, time.time() - t0)

        # 阶段产物：checkpoints/<阶段>_final.{zip,pkl,env.json}
        target = self.rundir.checkpoints / f"{stage.name}_final"
        self.rundir.save_model(model, target, venv)
        save_env_json(target, self.cfg, step_height=stage.step_height,
                      stage=stage.name)
        self.last_stats = callbacks[0].stats
        print(f"    训练完成 {stage.timesteps:,} 步  {sps:.0f} 步/秒", flush=True)
        return model, self.last_stats


# --------------------------------------------------------------------------- #
# 命令行
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None):
    import argparse

    ap = argparse.ArgumentParser(
        description="按 config.yaml 的固定课程训练盲走楼梯策略",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run-name", default=None,
                    help="输出目录名；默认用 config.yaml 的 run.name")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                    help="配置文件路径")
    ap.add_argument("--warm-start", default=None,
                    help="第一级从哪个 checkpoint 继承权重（跨 run 续训用；"
                         "例如从 runs/<旧 run>/checkpoints/stair9_final.zip 出发只训更高台阶）")
    ap.add_argument("--warm-norm", default="auto", choices=("auto", "on", "off"),
                    help="warm-start 时是否把权重旁边那份 *_vecnormalize.pkl 的观测统计"
                         "一起装回：auto = 有就装（默认）、on = 必须有、off = 不装"
                         "（回到 2026-10-17 之前的旧行为）")
    return ap.parse_args(argv)


def _result(stage: StageSpec, stats: EpisodeStats, wall: float, path: Path,
            cfg: EnvConfig):
    return StageResult(
        name=stage.name,
        label=stage.label,
        step_height=stage.step_height,
        timesteps=int(stage.timesteps),
        mean_return=float(stats.mean_return),
        success_rate=float(stats.success_rate),
        fall_rate=float(stats.fall_rate),
        wall_seconds=float(wall),
        model_path=str(path),
        mean_length=float(stats.mean_length),
        episodes=int(stats.n_total),
        obs_size=int(cfg.obs.size),
    )


def main(argv: Sequence[str] | None = None) -> int:
    console.setup_utf8_console()
    args = parse_args(argv)
    raw = load_config(args.config)

    run = config_to_run(raw)
    ppo = config_to_ppo(raw)
    cfg = config_to_env(raw)
    stages = config_to_stages(raw)
    name = args.run_name or str(run["name"])
    n_envs = cfg_n_envs(raw)
    vec_env = cfg_vec_env(raw)
    vec_env_impl = resolve_vec_env(n_envs, vec_env)

    for stage in stages:
        if stage.timesteps < MIN_STAGE_STEPS:
            raise SystemExit(f"阶段 {stage.name} 的预算太小：{stage.timesteps}")

    rundir = RunDir(name)
    if rundir.root.exists():
        raise SystemExit(f"输出目录已存在：{rundir.root}\n换个 --run-name，或先删掉它")
    rundir.ensure()

    print(f"[配置] {Path(args.config)}")
    print(f"[运行] {rundir.root}  (seed={run['seed']}, n_envs={n_envs}, "
          f"vec_env={vec_env} → {vec_env_impl})")
    print(f"[观测] 每帧 {cfg.obs.frame_size} 维 × history {cfg.obs.history_len} "
          f"= {cfg.obs.size} 维")
    if cfg.obs.privileged_frame_size > 0:
        # 非对称 AC：actor 的输入维数不含特权块（足端-地形接触只给 critic）。
        print(f"[观测] 非对称 AC: actor {cfg.obs.actor_size} 维 / critic "
              f"{cfg.obs.size} 维（critic 多 "
              f"{cfg.obs.privileged_frame_size * cfg.obs.history_len} 维特权：足端-地形接触）")
    if cfg.action_mode == "gait_params":
        print(f"[动作] gait_params 模式（12 维步态参数：前/后对各 amp_th/amp_ca/"
              f"off_th/off_ca/hip_amp/hip_off → sin 结构 + trot 相位）"
              f"  平地步频={cfg.gait_freq_flat:g} Hz  楼梯={cfg.gait_freq_stair:g} Hz"
              f"  楼梯步幅×{cfg.stair_stride_boost:g}"
              f"  幅值上限={cfg.param_amp_max:g}/偏置上限={cfg.param_off_max:g}")
    else:
        print(f"[动作] residual 模式, action_scale={cfg.action_scale} rad, "
              f"步态先验={'开' if cfg.gait_prior else '关'}")
    cmd = cfg.command
    print(f"[指令] 基线 vx={cmd.vx:g} m/s；实际三轴范围由指令课程按级下发"
          f"（观测里的那 3 维恒在）")
    if cfg.gait_prior:
        print(f"[步态结构] {cfg.gait.summary()}")
    print(f"[奖励] forward_vel={cfg.reward.forward_vel} "
          f"foot_lift={cfg.reward.foot_lift} success={cfg.reward.success} "
          f"termination={cfg.reward.termination}")
    print(f"[PPO] 网络={list(ppo.get('net_arch', (512, 256)))} "
          f"n_steps={ppo.get('n_steps')} batch={ppo.get('batch_size')} "
          f"epochs={ppo.get('n_epochs')} target_kl={ppo.get('target_kl')} "
          f"log_std_init={ppo.get('log_std_init')} "
          f"norm_reward={ppo.get('norm_reward')}")
    print()
    print(format_curriculum(stages))
    print(flush=True)

    trainer = Trainer(cfg, rundir, run=run, ppo=ppo, n_envs=n_envs,
                      vec_env=vec_env)
    results = []
    warm: str | None = getattr(args, "warm_start", None)
    if warm:
        print(f"[课程] 第一级从外部 checkpoint 继承权重: {warm}", flush=True)
    t_start = time.time()
    for i, stage in enumerate(stages, start=1):
        print(f"[课程] >>> 第 {i}/{len(stages)} 级 '{stage.name}' ({stage.label}) "
              f"{stage.summary()}", flush=True)
        venv = trainer.make_env(stage, warm_start=warm)
        t0 = time.time()
        try:
            _model, stats = trainer.train_stage(
                stage, venv, warm_start=warm,
                warm_norm=str(getattr(args, "warm_norm", "auto")),
                tensorboard_log=str(rundir.tensorboard),
            )
        finally:
            venv.close()
        wall = time.time() - t0
        path = rundir.checkpoints / f"{stage.name}_final.zip"

        results.append(_result(stage, stats, wall, path, cfg))
        warm = str(path)
        print(f"[课程] <<< '{stage.name}' 成功率={stats.success_rate:.1%} "
              f"摔倒率={stats.fall_rate:.1%} 均回报={stats.mean_return:.1f} "
              f"用时={format_seconds(wall)}", flush=True)

    # 末级权重另存为 model_final，方便 play 直接找（连同它的归一化统计与地形记录）
    if results and Path(results[-1].model_path).is_file():
        last = Path(results[-1].model_path)
        copies = [
            (last, Path(str(rundir.final_model) + ".zip")),
            (last.with_name(last.stem + "_vecnormalize.pkl"), rundir.vecnormalize),
            (env_json_path(last), env_json_path(rundir.final_model)),
        ]
        for src, dst in copies:
            if src.is_file():
                shutil.copy2(src, dst)

    final_model = results[-1].model_path if results else ""
    # 课程的实施记录：每一级实际生效的指令量程 / 奖励覆盖。``env_config`` 与
    # ``<阶段>_env.json`` 都是 ``config.yaml`` 的基线，复现某一级时必须读这个。
    plan_path = rundir.root / "plan.json"
    plan_path.write_text(
        json.dumps(trainer.plans, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    payload: dict[str, Any] = {
        "env_config": asdict(cfg),
        "obs_config": asdict(cfg.obs),
        "reward_config": asdict(cfg.reward),
        "final_model": final_model,
        "wall_seconds": round(time.time() - t_start, 1),
    }
    summary = write_summary(rundir.summary, stages, results, payload)

    print()
    print(format_curriculum(stages, results))
    print(f"\n产物: {rundir.root}")
    print(f"  模型: {rundir.root / 'model_final'}.zip")
    print(f"  汇总: {summary}")
    print("\n下一步:")
    print(f"  .venv\\Scripts\\python.exe scripts\\play.py --run {name} --mode eval")
    print(f"  .venv\\Scripts\\python.exe scripts\\play.py --run {name} --mode video")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

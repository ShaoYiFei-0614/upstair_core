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
)
from robotdog.curriculum import (  # noqa: E402
    EpisodeStats,
    StageSpec,
    format_curriculum,
    write_summary,
)
from robotdog.env import EnvConfig  # noqa: E402
from robotdog.policy import env_json_path, save_env_json  # noqa: E402
from robotdog.tools import console  # noqa: E402
from robotdog.training import (  # noqa: E402
    RunDir,
    build_ppo,
    format_duration,
    make_vec_env,
    save_vecnormalize,
)

#: 只打印进度，不训练的最小步数（避免把 0 步的配置跑成死循环）。
MIN_STAGE_STEPS = 1


def kw_call(venv, method: str, **kwargs):
    """用**关键字**调用子环境方法（``venv.env_method`` 的薄封装）。

    ``env_method`` 是 ``(*method_args, **method_kwargs)`` 原样透传
    （``stable_baselines3/common/vec_env/dummy_vec_env.py:127``），位置实参按
    **形参顺序**绑定。凡是「多个同类型可选参数」的子环境方法，调用点一律走这里，
    让「参数名对不上」变成 ``TypeError`` 而不是静默传错值。
    """
    return venv.env_method(method, **kwargs)


# --------------------------------------------------------------------------- #
# 回调
# --------------------------------------------------------------------------- #


class ProgressCallback(BaseCallback):
    """有新回合结束时（且距上次打印至少 ``log_every`` 次更新）打印一行进度。"""

    def __init__(self, *, log_every: int = 20, window: int = 50,
                 verbose: int = 1) -> None:
        super().__init__(verbose)
        self.log_every = max(1, int(log_every))
        self.stats = EpisodeStats(window=window)
        self._last_episodes = 0
        self._last_print = 0

    def _on_step(self) -> bool:
        self.stats.add_infos(self.locals.get("infos", []))
        new_episode = self.stats.n_total != self._last_episodes
        due = self.n_calls - self._last_print >= self.log_every
        if due and new_episode:
            self._last_episodes = self.stats.n_total
            self._last_print = self.n_calls
            print(f"    [{self.num_timesteps:>9,} 步] {self.stats.format_line()}",
                  flush=True)
        return True


class CheckpointCallback(BaseCallback):
    """每 ``every`` 步存一份 ``checkpoints/<阶段>/<阶段>_<步数>_steps.zip``。"""

    def __init__(self, directory: Path, stage: str, *, every: int,
                 verbose: int = 0) -> None:
        super().__init__(verbose)
        self.directory = Path(directory)
        self.stage = stage
        self.every = max(1, int(every))
        self._last = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last < self.every:
            return True
        self._last = self.num_timesteps
        self.directory.mkdir(parents=True, exist_ok=True)
        base = self.directory / f"{self.stage}_{self.num_timesteps}_steps"
        self.model.save(str(base))
        save_vecnormalize(self.model, base)
        return True


# --------------------------------------------------------------------------- #
# 训练者
# --------------------------------------------------------------------------- #


class Trainer:
    """实现 :class:`robotdog.curriculum.StageTrainer` 协议。

    所有超参来自 ``config.yaml``，阶段特有的步数与学习率来自 :class:`StageSpec`。
    """

    def __init__(self, cfg: EnvConfig, rundir: RunDir, *, run: dict[str, Any],
                 ppo: dict[str, Any], n_envs: int) -> None:
        self.cfg = cfg
        self.rundir = rundir
        self.run = run
        self.ppo = ppo
        self.n_envs = n_envs
        self.last_stats = EpisodeStats(window=200)
        # ⭐ ``config.yaml`` 的奖励基线（**没叠任何课程覆盖**的那一份）。
        # ``make_env`` 每一级开始前把它抄回子环境，保证奖励覆盖是「从基线叠」
        # 而不是「在上一次覆盖之上再叠」。这个陷阱在没有深拷贝的年代不可见
        # （因为那时所有级共享同一个被改烂的 cfg），加上深拷贝之后才暴露出来。
        self._reward_baseline = copy.deepcopy(cfg.reward)

    # ---- 每个阶段一个独立的向量环境（地形烘焙在编译后的 MJCF 里）----------
    def make_env(self, stage: StageSpec, *, warm_start: str | None = None):
        # ⭐ **每一级必须是自己的 ``EnvConfig`` 副本。**
        #
        # ``make_vec_env`` 把传进去的 cfg **原样绑**到每个子环境上，不做拷贝，
        # 而下面那一串下发全是**就地改写**：``set_commands`` 写
        # ``cfg.command.*``、``set_reward_overrides`` 用 ``setattr`` 写
        # ``cfg.reward.*``。直接传 ``self.cfg`` 会让这些设置**跨阶段累积**，
        # 顺序敏感。深拷贝一份，让每一级都从 ``config.yaml`` 的基线出发。
        stage_cfg = copy.deepcopy(self.cfg)
        venv = make_vec_env(
            stage.spec,
            n_envs=self.n_envs,
            cfg=stage_cfg,
            seed=int(self.run["seed"]),
            norm_reward=bool(self.ppo.get("norm_reward", True)),
        )
        # 指令课程：把这一级的指令**采样范围**下发给每一个子环境。阶段内不再改动
        # —— 观测的那 3 维在整个课程里恒定，PPO 的网络与 RolloutBuffer 才不用重建。
        #
        # ⚠ 这里**不能**用 ``if stage.command_vy_max:`` 之类的守卫：量程为 0 是
        # **有意义的设置**（这一级不练这个方向），跳过下发只会让这一级继承上一级
        # 的范围。无条件下发，且**必须走关键字**（见 ``kw_call``）。
        kw_call(venv, "set_commands",
                vy_max=stage.command_vy_max,
                yaw_max=stage.command_yaw_max,
                vx_min=stage.command_vx[0],
                vx_max=stage.command_vx[1])
        # 奖励课程：这一级的奖励覆盖（``curriculum.reward_overrides``）叠到
        # ``reward:`` 段的基线值上。常见的用法是「台阶越高，越需要抬腿试探」，
        # 于是逐级放大 ``foot_lift`` / ``probe_lift`` 这类成形项。
        #
        # ``set_reward_overrides`` 是 ``setattr`` 就地改，所以必须先显式把
        # 基线抄回去 —— 否则上一级叠上去的键会残留，而这一级没覆盖到的键
        # 就永远是上一级的值（``stage_cfg`` 的深拷贝只保证「不污染 self.cfg」，
        # 保证不了「每一级都从基线出发」）。
        venv.env_method("set_reward_overrides",
                        dict(asdict(self._reward_baseline)))
        if stage.reward_overrides:
            venv.env_method("set_reward_overrides", dict(stage.reward_overrides))
        return venv

    def train_stage(self, stage: StageSpec, venv, *, warm_start: str | None = None,
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
            tensorboard_log=tensorboard_log,
            device=str(self.ppo.get("device", "cpu")),
        )
        if warm_start:
            model.set_parameters(str(warm_start))
            print(f"    [warm-start] 从 {Path(warm_start).name} 继承权重", flush=True)

        callbacks = [ProgressCallback(log_every=int(self.run["log_every"]))]
        every = int(self.run["save_every"])
        if every > 0:
            callbacks.append(
                CheckpointCallback(self.rundir.checkpoints / stage.name, stage.name,
                                   every=every)
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
    return ap.parse_args(argv)


def _result(stage: StageSpec, stats: EpisodeStats, wall: float, path: Path,
            cfg: EnvConfig):
    from robotdog.curriculum import StageResult

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

    for stage in stages:
        if stage.timesteps < MIN_STAGE_STEPS:
            raise SystemExit(f"阶段 {stage.name} 的预算太小：{stage.timesteps}")

    rundir = RunDir(name)
    if rundir.root.exists():
        raise SystemExit(f"输出目录已存在：{rundir.root}\n换个 --run-name，或先删掉它")
    rundir.ensure()

    print(f"[配置] {Path(args.config)}")
    print(f"[运行] {rundir.root}  (seed={run['seed']}, n_envs={n_envs})")
    print(f"[观测] 每帧 {cfg.obs.frame_size} 维 × history {cfg.obs.history_len} "
          f"= {cfg.obs.size} 维")
    print(f"[动作] 残差模式, action_scale={cfg.action_scale} rad, "
          f"步态先验={'开' if cfg.gait_prior else '关'}")
    cmd = cfg.command
    print(f"[指令] 基线 vx={cmd.vx:g} m/s；实际三轴范围由指令课程按级下发"
          f"（观测里的那 3 维恒在）")
    if cfg.gait_prior:
        print(f"[步态] {cfg.gait.summary()}")
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

    trainer = Trainer(cfg, rundir, run=run, ppo=ppo, n_envs=n_envs)
    results = []
    warm: str | None = None
    t_start = time.time()
    for i, stage in enumerate(stages, start=1):
        print(f"[课程] >>> 第 {i}/{len(stages)} 级 '{stage.name}' ({stage.label}) "
              f"{stage.summary()}", flush=True)
        venv = trainer.make_env(stage, warm_start=warm)
        t0 = time.time()
        try:
            _model, stats = trainer.train_stage(
                stage, venv, warm_start=warm,
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
              f"用时={format_duration(wall)}", flush=True)

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
    print(f"  .venv\\Scripts\\python.exe scripts\\play.py --run {name} --mode gif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

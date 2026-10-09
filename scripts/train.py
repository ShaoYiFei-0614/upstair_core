"""训练盲走楼梯策略。

按 :mod:`robotdog.curriculum` 的固定课程串行训练：**平地 → 3 → 6 → 9 → 12 → 15 cm**。
每一级做一个地形、训完保存，下一级从上一级的权重 warm-start（``PPO.set_parameters``），
所以策略会逐级迁移。

用法::

    # 完整课程（默认）
    .venv\\Scripts\\python.exe scripts\\train.py --run-name course1

    # 只训某一级起（调试用）
    .venv\\Scripts\\python.exe scripts\\train.py --run-name dbg --from-stage stair3

    # 从头短跑一遍（每个阶段 20k 步）
    .venv\\Scripts\\python.exe scripts\\train.py --run-name smoke --total-timesteps 120000

产物落在 ``runs/<名字>/``：

* ``model_final.zip`` / ``model_final_vecnormalize.pkl`` —— 最后一级的权重与观测统计
* ``checkpoints/<阶段>_final.zip`` —— 每一级的权重（也带 ``_vecnormalize.pkl``）
* ``summary.json`` —— 课程定义、每级实测结果、完整 ``EnvConfig``（供 play 恢复）
* ``tensorboard/`` —— PPO 标量

**观测归一化必须跟权重一起用**：``*_vecnormalize.pkl`` 与 ``*.zip`` 是成对的，
play 会按同名自动配对。

⚠ ``--resume`` 只会恢复权重与进度，**不会**应用 ``--net-arch`` / ``--learning-rate`` /
``--target-kl``（PPO 构造时已固定），要改这些必须从头训。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from robotdog.curriculum import (  # noqa: E402
    EpisodeStats,
    StageSpec,
    default_curriculum,
    format_curriculum,
    write_summary,
)
from robotdog.env import EnvConfig  # noqa: E402
from robotdog.gait import load_gait  # noqa: E402
from robotdog.tools import paths  # noqa: E402
from robotdog.training import (  # noqa: E402
    DEFAULT_NET_ARCH,
    RunDir,
    build_ppo,
    format_duration,
    make_vec_env,
)


# --------------------------------------------------------------------------- #
# 回调
# --------------------------------------------------------------------------- #


class ProgressCallback(BaseCallback):
    """有新回合结束时（且距上次打印至少 ``log_interval`` 次更新）打印一行进度。"""

    def __init__(self, *, log_interval: int = 20, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.log_interval = max(1, int(log_interval))
        self.stats = EpisodeStats(window=50)
        self._last_episodes = 0
        self._last_print = 0

    def _on_step(self) -> bool:
        self.stats.add_infos(self.locals.get("infos", []))
        new_episode = self.stats.n_total != self._last_episodes
        due = self.n_calls - self._last_print >= self.log_interval
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
        from robotdog.training import save_vecnormalize

        save_vecnormalize(self.model, base)
        return True


# --------------------------------------------------------------------------- #
# 训练者
# --------------------------------------------------------------------------- #


class Trainer:
    """实现 :class:`robotdog.curriculum.StageTrainer` 协议。"""

    def __init__(self, cfg: EnvConfig, rundir: RunDir, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.rundir = rundir
        self.args = args
        self.last_stats = EpisodeStats(window=200)
        self.last_venv = None

    # ---- 每个阶段一个独立的向量环境（地形烘焙在编译后的 MJCF 里） ----------
    def make_env(self, stage: StageSpec, *, warm_start: str | None = None):
        return make_vec_env(
            stage.spec,
            n_envs=self.args.n_envs,
            cfg=self.cfg,
            seed=self.args.seed,
            norm_reward=bool(self.args.norm_reward),
        )

    def train_stage(self, stage: StageSpec, venv, *, warm_start: str | None = None,
                    tensorboard_log: str | Path | None = None, **_: Any):
        model = build_ppo(
            venv,
            seed=self.args.seed,
            learning_rate=stage.learning_rate,
            n_steps=self.args.n_steps,
            batch_size=self.args.batch_size,
            n_epochs=self.args.n_epochs,
            target_kl=self.args.target_kl,
            log_std_init=self.args.log_std_init,
            net_arch=tuple(self.args.net_arch),
            tensorboard_log=tensorboard_log,
            device=self.args.device,
        )
        if warm_start:
            model.set_parameters(str(warm_start))
            print(f"    [warm-start] 从 {Path(warm_start).name} 继承权重", flush=True)

        callbacks = [ProgressCallback(log_interval=self.args.log_interval)]
        if self.args.save_every > 0:
            callbacks.append(
                CheckpointCallback(self.rundir.checkpoints / stage.name, stage.name,
                                   every=self.args.save_every)
            )
        t0 = time.time()
        model.learn(total_timesteps=stage.timesteps, callback=callbacks,
                    reset_num_timesteps=True)
        sps = stage.timesteps / max(1e-6, time.time() - t0)

        # 阶段产物：checkpoints/<阶段>_final.{zip,pkl}
        target = self.rundir.checkpoints / f"{stage.name}_final"
        self.rundir.save_model(model, target, venv)
        self.last_stats = callbacks[0].stats
        self.last_venv = venv
        print(f"    训练完成 {stage.timesteps:,} 步  {sps:.0f} 步/秒", flush=True)
        return model, self.last_stats


# --------------------------------------------------------------------------- #
# 配置装配
# --------------------------------------------------------------------------- #


def build_config(args: argparse.Namespace) -> EnvConfig:
    cfg = EnvConfig()
    cfg.episode_length_s = args.episode_length_s
    cfg.action_scale = args.action_scale
    cfg.command_vx = args.command_vx
    cfg.terminate_contact_steps = args.terminate_contact_steps
    cfg.obs.base_lin_vel_body = bool(args.obs_base_lin_vel_body)
    cfg.obs.base_height = bool(args.obs_base_height)
    cfg.obs.gait_phase = bool(args.obs_gait_phase)
    cfg.obs.step_height = bool(args.obs_step_height)
    if args.gait_prior:
        cfg.gait_prior = True
        cfg.gait = load_gait(args.gait_file)
    return cfg


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="按固定课程训练盲走楼梯策略",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run-name", default="course1")
    ap.add_argument("--total-timesteps", type=int, default=None,
                    help="全部阶段的总步数；不给则用每级默认预算")
    ap.add_argument("--from-stage", default=None,
                    help="从课程的某一级开始（跳过之前的级）")
    ap.add_argument("--resume", default=None, help="从某个 .zip 权重继续训练")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")

    # PPO
    ap.add_argument("--net-arch", type=int, nargs=2, default=list(DEFAULT_NET_ARCH))
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-epochs", type=int, default=5)
    ap.add_argument("--target-kl", type=float, default=0.03)
    ap.add_argument("--log-std-init", type=float, default=-2.0)
    ap.add_argument("--norm-reward", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="奖励归一化；从零训练时它是最强的杠杆，"
                         "但 warm-start 一个未归一化的权重时应当关掉")

    # 地形与步态
    ap.add_argument("--tread", type=float, default=0.50)
    ap.add_argument("--n-up", type=int, default=8)
    ap.add_argument("--gait-prior", action=argparse.BooleanOptionalAction,
                    default=True, help="在 trot 开环先验之上学习残差")
    ap.add_argument("--gait-file", default=str(paths.ARTIFACTS_DIR / "trot_stairs.json"))

    # 环境
    ap.add_argument("--episode-length-s", type=float, default=20.0)
    ap.add_argument("--action-scale", type=float, default=0.30)
    ap.add_argument("--command-vx", type=float, default=0.55)
    ap.add_argument("--terminate-contact-steps", type=int, default=10)
    ap.add_argument("--obs-base-lin-vel-body", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--obs-base-height", action=argparse.BooleanOptionalAction,
                    default=False)
    ap.add_argument("--obs-gait-phase", action=argparse.BooleanOptionalAction,
                    default=False)
    ap.add_argument("--obs-step-height", action=argparse.BooleanOptionalAction,
                    default=True)

    # 日志与存档
    ap.add_argument("--log-interval", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=50000)
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    stages = default_curriculum(
        tread=args.tread, n_up=args.n_up, total_timesteps=args.total_timesteps
    )
    if args.from_stage:
        names = [s.name for s in stages]
        if args.from_stage not in names:
            raise SystemExit(f"--from-stage 必须是 {names} 之一")
        stages = stages[names.index(args.from_stage):]

    rundir = RunDir(args.run_name)
    print(f"[运行] {rundir.root}")
    print(f"[观测] 每帧 {cfg.obs.frame_size} 维 × history {cfg.obs.history_len} "
          f"= {cfg.obs.size} 维")
    print(f"[动作] 残差模式, action_scale={cfg.action_scale} rad, "
          f"步态先验={'开' if cfg.gait_prior else '关'}")
    if cfg.gait_prior:
        print(f"[步态] {cfg.gait.summary()}")
    print(f"[奖励] forward_vel={cfg.reward.forward_vel} "
          f"foot_lift={cfg.reward.foot_lift} success={cfg.reward.success} "
          f"termination={cfg.reward.termination}")
    print(f"[PPO] 网络={list(args.net_arch)} n_steps={args.n_steps} "
          f"batch={args.batch_size} epochs={args.n_epochs} "
          f"target_kl={args.target_kl} log_std_init={args.log_std_init}")
    print()
    print(format_curriculum(stages))
    print()

    trainer = Trainer(cfg, rundir, args)
    results = []
    warm = args.resume
    t_start = time.time()
    for i, stage in enumerate(stages, start=1):
        print(f"[课程] >>> 第 {i}/{len(stages)} 级 '{stage.name}' ({stage.label}) "
              f"{stage.summary()}", flush=True)
        venv = trainer.make_env(stage, warm_start=warm)
        t0 = time.time()
        try:
            model, stats = trainer.train_stage(
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

    # 末级权重另存为 model_final，方便 play 直接找
    if results and Path(results[-1].model_path).is_file():
        import shutil

        shutil.copy2(results[-1].model_path, Path(str(rundir.final_model) + ".zip"))
        norm = Path(results[-1].model_path).with_name(
            Path(results[-1].model_path).stem + "_vecnormalize.pkl"
        )
        if norm.is_file():
            shutil.copy2(norm, Path(str(rundir.final_model) + "_vecnormalize.pkl"))

    final_model = results[-1].model_path if results else ""
    payload: dict[str, Any] = {
        "env_config": asdict(cfg),
        "obs_config": asdict(cfg.obs),
        "reward_config": asdict(cfg.reward),
        "final_model": final_model,
        "wall_seconds": round(time.time() - t_start, 1),
    }
    summary = rundir.write_summary(payload)
    write_summary(summary, stages, results, payload)

    print()
    print(format_curriculum(stages, results))
    print(f"\n产物: {rundir.root}")
    print(f"  模型: {rundir.root / 'model_final'}.zip")
    print(f"  汇总: {summary}")
    print("\n下一步:")
    print(f"  .venv\\Scripts\\python.exe scripts\\play.py --run {args.run_name} "
          f"--mode eval --episodes 100")
    print(f"  .venv\\Scripts\\python.exe scripts\\play.py --run {args.run_name} "
          f"--mode gif")
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())

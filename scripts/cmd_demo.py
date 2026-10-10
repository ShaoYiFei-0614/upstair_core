"""指令跟踪演示：给策略喂一串**分段恒定**的横向/旋转指令，边跑边看它跟不跟。

与 ``scripts/teleop.py`` 的区别：teleop 是「控制器闭环看着地形打方向」（遥感操控），
本脚本是「开环给指令」，专门用来**量跟踪能力**并出 GIF 给用户看：

* 每 ``--hold-steps`` 个控制步换一条指令，段内恒定、段间跳变；
* 每段统计**实测均值的符号/量级**，最后出一张「指令 vs 实测」表，配
  ``斜率``（实测对指令的线性回归）与 ``相关系数``；

用法::

    # 原地横移 + 原地偏航（要用手上带原地阶段的权重）
    .venv\\Scripts\\python.exe scripts\\cmd_demo.py --run cmdtrack --phase inplace --gif

    # 边走边横移
    .venv\\Scripts\\python.exe scripts\\cmd_demo.py --run cmdtrack --phase walk --gif

⚠ 权重必须**是带指令课程训出来的**（``curriculum.command_vy_max`` 非零的阶段），
否则观测维数或指令范围对不上。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from play import (  # noqa: E402
    CAMERA_GIF,
    FRAME_HEIGHT,
    FRAME_STRIDE,
    FRAME_WIDTH,
    open_runner,
    resolve_target,
    write_gif,
)
from robotdog.config import DEFAULT_CONFIG_PATH, device, load_config  # noqa: E402
from robotdog.tools import console, paths  # noqa: E402

#: 分段指令表：``(标签, vx, vy, wz)``。每段按顺序跑 ``--hold-steps`` 步。
#: 数值都落在课程量程内（vy ±0.40 / wz ±0.70），否则量到的是外推行为。
INPLACE_PHASES: tuple[tuple[str, float, float, float], ...] = (
    ("站住",      0.00,  0.00,  0.00),
    ("右移 +0.35", 0.00,  0.35,  0.00),
    ("左移 -0.35", 0.00, -0.35,  0.00),
    ("站住",      0.00,  0.00,  0.00),
    ("右转 -0.60", 0.00,  0.00, -0.60),
    ("左转 +0.60", 0.00,  0.00,  0.60),
    ("站住",      0.00,  0.00,  0.00),
    ("斜走 +0.30/-0.45", 0.00, 0.30, -0.45),
    ("斜走 -0.30/+0.45", 0.00, -0.30, 0.45),
)

WALK_PHASES: tuple[tuple[str, float, float, float], ...] = (
    ("直走 0.45",     0.45,  0.00,  0.00),
    ("走+右移 0.35",  0.45,  0.35,  0.00),
    ("走+左移 -0.35", 0.45, -0.35,  0.00),
    ("直走 0.45",     0.45,  0.00,  0.00),
    ("走+右转 -0.60", 0.45,  0.00, -0.60),
    ("走+左转 +0.60", 0.45,  0.00,  0.60),
    ("直走 0.45",     0.45,  0.00,  0.00),
)


@dataclass
class PhaseStat:
    """一段恒定指令的跟踪结果。"""

    label: str
    steps: int
    cmd_vy: float
    cmd_wz: float
    mean_vy: float
    mean_wz: float

    @property
    def err_vy(self) -> float:
        return abs(self.mean_vy - self.cmd_vy)

    @property
    def err_wz(self) -> float:
        return abs(self.mean_wz - self.cmd_wz)


def slope_and_r(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """实测对指令的最小二乘斜率与皮尔逊相关系数。样本太少时返回 ``(0, 0)``。"""
    if len(xs) < 3 or len(set(xs)) < 2:
        return 0.0, 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    slope = float(np.polyfit(x, y, 1)[0])
    sx, sy = float(x.std()), float(y.std())
    if sx == 0.0 or sy == 0.0:
        return slope, 0.0
    return slope, float(np.corrcoef(x, y)[0, 1])


def run_demo(env, runner, phases, *, seed: int, hold_steps: int, capture: bool,
             camera: str, width: int, height: int) -> tuple[list[PhaseStat], list[np.ndarray], dict]:
    """跑完整个分段指令表，返回逐段统计、GIF 帧与逐段原始序列。"""
    obs, _info = env.reset(seed=seed)
    frames: list[np.ndarray] = []
    stats: list[PhaseStat] = []
    trace: dict[str, list[float]] = {"cmd_vy": [], "vy": [], "cmd_wz": [], "wz": []}
    step = 0
    for label, vx, vy, wz in phases:
        # 指令在段内恒定、段间跳变 —— 这正是「指哪走哪」要考的东西。
        env.set_command_now(vy=vy, yaw_rate=wz, vx=vx)
        # set_command_now 只刷新环境内部的那一帧，手里的 obs 还带旧指令，必须重取。
        obs = env._observation()
        acc_vy: list[float] = []
        acc_wz: list[float] = []
        for _ in range(hold_steps):
            obs, _r, terminated, truncated, info = env.step(runner.predict(obs))
            step += 1
            acc_vy.append(float(info["lateral_vel"]))
            acc_wz.append(float(info["yaw_rate"]))
            trace["cmd_vy"].append(vy)
            trace["vy"].append(float(info["lateral_vel"]))
            trace["cmd_wz"].append(wz)
            trace["wz"].append(float(info["yaw_rate"]))
            if capture and step % FRAME_STRIDE == 0:
                frames.append(env.render(camera, width=width, height=height).copy())
            if terminated or truncated:
                # 摔倒了就重开一回合，但**继续**按原计划换指令，便于看趋势。
                obs, _info = env.reset(seed=seed + step)
                env.set_command_now(vy=vy, yaw_rate=wz, vx=vx)
                obs = env._observation()
        stats.append(PhaseStat(
            label=label, steps=hold_steps, cmd_vy=vy, cmd_wz=wz,
            mean_vy=float(np.mean(acc_vy)), mean_wz=float(np.mean(acc_wz)),
        ))
    return stats, frames, trace


def main(argv: list[str] | None = None) -> int:
    console.setup_utf8_console()
    parser = argparse.ArgumentParser(description="指令跟踪演示与量化")
    parser.add_argument("--run", required=True, help="运行名 / 目录 / .zip")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--phase", choices=("inplace", "walk"), default="inplace",
                        help="inplace = 原地指令表；walk = 边走边跟指令表")
    parser.add_argument("--hold-steps", type=int, default=75,
                        help="每段持续的控制步数（默认 75 = 1.5 s）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args(argv)
    raw = load_config(args.config)  # 早失败：配置有问题就不要等到环境建好才发现

    target = resolve_target(args.run)
    from robotdog.tools.terrain import flat_spec

    # step_height 必须显式给 0：否则 StairSpec 会留 0.05 的默认值，
    # 观测的 step_height 槽会按「5cm 台阶」报数（地形本身没有台阶块）。
    stair = flat_spec(tread=0.50, width=2.0, start_x=1.50, step_height=0.0)
    runner, env, _venv = open_runner(target, stair=stair, device=device(raw))

    phases = INPLACE_PHASES if args.phase == "inplace" else WALK_PHASES
    # 把指令量程放开到演示表的最大幅度，否则 `set_command_now` 的指令会被
    # 采样范围夹住（课程范围之外的指令策略没见过，量到的也不是课程内性能）。
    vy_limit = max(abs(p[2]) for p in phases)
    wz_limit = max(abs(p[3]) for p in phases)
    env.set_commands(vy_max=vy_limit, yaw_max=wz_limit)

    print(f"[指令演示] {target.label}  平地  指令表={args.phase}  "
          f"每段 {args.hold_steps} 步（{args.hold_steps * env.cfg.control_dt:.2f} s）  "
          f"量程 vy±{vy_limit:g} wz±{wz_limit:g}")
    stats, frames, trace = run_demo(
        env, runner, phases, seed=args.seed, hold_steps=args.hold_steps,
        capture=args.gif, camera=CAMERA_GIF,
        width=FRAME_WIDTH, height=FRAME_HEIGHT,
    )

    print(f"{'指令段':<18}{'指令 vy':>9}{'实测 vy':>9}{'指令 wz':>9}{'实测 wz':>9}")
    for s in stats:
        print(f"{s.label:<18}{s.cmd_vy:>+9.3f}{s.mean_vy:>+9.3f}"
              f"{s.cmd_wz:>+9.3f}{s.mean_wz:>+9.3f}")

    sl_vy, r_vy = slope_and_r(trace["cmd_vy"], trace["vy"])
    sl_wz, r_wz = slope_and_r(trace["cmd_wz"], trace["wz"])
    print()
    print(f"横向：斜率 {sl_vy:+.3f}（1.0 = 完全跟上）  相关系数 {r_vy:+.3f}  "
          f"平均|误差| {np.mean(np.abs(np.array(trace['vy']) - np.array(trace['cmd_vy']))):.3f} m/s")
    print(f"偏航：斜率 {sl_wz:+.3f}（1.0 = 完全跟上）  相关系数 {r_wz:+.3f}  "
          f"平均|误差| {np.mean(np.abs(np.array(trace['wz']) - np.array(trace['cmd_wz']))):.3f} rad/s")

    if args.gif and frames:
        out = paths.OUTPUT_GIFS_DIR / f"{target.label}_cmd_{args.phase}.gif"
        write_gif(out, frames)
        print(f"  GIF: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""用外部控制器遥控策略 —— 模拟人在遥感操控。

这个脚本演示的是 :class:`robotdog.env.Go2StairsEnv` 的**外部指令接口**：策略
只负责跟踪指令，指令由外部控制器给出。控制器**可以**读地形与机器狗当前位置
（这正是「遥感操控」的含义：人是看着场景在打方向），所以它比策略多知道很多
信息，不违反盲走约束 —— 盲走约束只限制**策略的观测**。

两个接口的分工：

* ``env.set_command_now(vy=..., yaw_rate=..., vx=...)`` —— 每个控制步改写**当前
  指令**，控制器的高频入口（本脚本每个控制步都调一次）。
* ``env.set_commands(vy_max=..., yaw_max=...)`` —— 改写**采样范围**，课程在阶段
  边界调用；这里只用来把范围放开到控制器的输出上限。

用法::

    .venv\\Scripts\\python.exe scripts\\teleop.py --run course1
    .venv\\Scripts\\python.exe scripts\\teleop.py --run course1 --height 9 --gif
    .venv\\Scripts\\python.exe scripts\\teleop.py --run flatcmd --hold 0.0 0.30 0.0

``--hold vx vy wz`` 是最直接的演示：把指令钉死在一条上跑满一回合，用来量策略
跟踪某个**指定**指令的能力（照 ``StairController`` 更能看出「遥感操控」的样子）。
没有 ``--hold`` 时用 ``StairController`` —— 沿楼梯中线爬升的横移 + 对准。

⚠ 权重必须**是带指令课程训出来的**（``config.yaml`` 的
``curriculum.command_vy_max`` / ``command_yaw_max`` 非零的那些阶段）。用旧权重
（观测 46 维）会直接报形状不匹配，这是有意的：那批权重不认识指令观测。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robotdog.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    config_to_stages,
    device,
    load_config,
)
from robotdog.policy import stair_from_height  # noqa: E402
from robotdog.tools import console, paths  # noqa: E402

from play import (  # noqa: E402
    CAMERA_GIF,
    FRAME_HEIGHT,
    FRAME_STRIDE,
    FRAME_WIDTH,
    open_runner,
    read_json_or,
    resolve_target,
    write_gif,
)
from robotdog.env import Go2StairsEnv  # noqa: E402


# --------------------------------------------------------------------------- #
# 外部控制器
# --------------------------------------------------------------------------- #


@dataclass
class GaitCommand:
    """一条指令 ``[vx, vy, wz]``，带可读字段。"""

    vx: float = 0.55
    vy: float = 0.0
    wz: float = 0.0


def flat_spec_for(width: float = 2.0, start_x: float = 1.50):
    """一块只有平地的地形（``--hold`` 演示用）。"""
    from robotdog.tools.terrain import flat_spec

    # 显式给 step_height：flat_spec 不改它的话会留 0.05 的默认值，
    # 而观测的 step_height 槽读的就是它（地形本身没有台阶块）。
    return flat_spec(tread=0.50, width=width, start_x=start_x, step_height=0.0)


class StairController:
    """看着地形与机器狗位置打方向的控制器。

    策略：**爬升阶段**（机身在楼梯 x 区间内）以恒定前向速度为主，用偏航把机身
    对准楼梯中线（``y = width/2``）、用横移把机身推回中线；**接近楼梯之前**先
    对准入口；**越过顶端**之后收住指令让它站稳。
    """

    def __init__(self, stair, *, vx: float = 0.55, vy_gain: float = 0.9,
                 vy_limit: float, yaw_gain: float = 1.6, yaw_limit: float,
                 align_dist: float = 1.2) -> None:
        self.stair = stair
        self.vx = vx
        self.vy_gain = vy_gain
        self.vy_limit = vy_limit
        self.yaw_gain = yaw_gain
        self.yaw_limit = yaw_limit
        self.align_dist = align_dist

    def _clip(self, value: float, limit: float) -> float:
        return float(np.clip(value, -limit, limit))

    def __call__(self, env: Go2StairsEnv, step: int) -> GaitCommand:
        x = float(env.base_pos[0])
        y = float(env.base_pos[1])
        yaw = float(env.yaw)
        stair = self.stair
        # 楼梯中线：地形在 y 上以 center_y 为中心、宽 width（见 StairSpec.to_xml）。
        y_target = stair.center_y

        if x < stair.start_x - self.align_dist:
            # 远距离：直着走过去，不改横向。
            return GaitCommand(vx=self.vx, vy=0.0, wz=0.0)

        if x > stair.up_end_x:
            # 已经越过顶端：收住前进，站稳。
            return GaitCommand(vx=0.0, vy=0.0, wz=0.0)

        # 1) 先把机身「朝 +x」摆正：偏航误差 = -yaw 折到 [-pi, pi]。
        # 2) 再朝中线压一点：目标偏航角 = -gain * (y - y_target)。
        desired_yaw = -0.5 * np.arctan2(y - y_target, 1.0)
        yaw_err = np.arctan2(np.sin(desired_yaw - yaw), np.cos(desired_yaw - yaw))
        # 前向速度按航向误差打折，免得斜着去撞立面。
        vx = self.vx * float(np.clip(np.cos(yaw_err), 0.0, 1.0))
        return GaitCommand(
            vx=vx,
            vy=self._clip(-self.vy_gain * (y - y_target), self.vy_limit),
            wz=self._clip(self.yaw_gain * yaw_err, self.yaw_limit),
        )


class HoldController:
    """把一条固定指令钉死跑满一回合 —— 量「跟踪某个指定指令」的能力。

    这是最干净的指令跟踪判据：指令不随状态变化，所以回合统计里
    「横向末位 / 偏航末位 / 平均指令」的偏差就是纯粹的跟踪误差。
    """

    def __init__(self, *, vx: float, vy: float, wz: float) -> None:
        self.cmd = GaitCommand(vx=vx, vy=vy, wz=wz)

    def __call__(self, env: Go2StairsEnv, step: int) -> GaitCommand:
        return self.cmd


# --------------------------------------------------------------------------- #
# 回合
# --------------------------------------------------------------------------- #


def run_teleop(env: Go2StairsEnv, runner, controller, *, seed: int,
               capture: bool, camera: str, width: int, height: int
               ) -> tuple[dict, list[np.ndarray]]:
    """跑一个被遥控的回合，返回统计与（可选的）帧。"""
    obs, _ = env.reset(seed=seed)
    x0 = float(env.base_pos[0])
    y0 = float(env.base_pos[1])
    yaw0 = float(env.yaw)
    z0 = float(env.base_pos[2])
    max_steps = int(env.cfg.episode_length_s / env.cfg.control_dt)
    frames: list[np.ndarray] = []
    info: dict = {}
    terminated = truncated = False
    step = 0
    sum_cmd = np.zeros(3)
    abs_vy_err: list[float] = []
    abs_wz_err: list[float] = []

    while not (terminated or truncated) and step < max_steps:
        cmd = controller(env, step)
        env.set_command_now(vx=cmd.vx, vy=cmd.vy, yaw_rate=cmd.wz, hold=True)
        # set_command_now 只刷新环境内部「已算好还没被取走」的那一帧，
        # 调用方手里的 obs 还是上一步的指令，必须重取。
        obs = env._observation()
        sum_cmd += np.array([cmd.vx, cmd.vy, cmd.wz])
        obs, _r, terminated, truncated, info = env.step(runner.predict(obs))
        step += 1
        # 跟踪误差：info 里的 lateral_vel / yaw_rate 是**实测值**（传感器口径），
        # 与 cmd 的差就是策略没跟上的部分。
        abs_vy_err.append(abs(float(info.get("lateral_vel", 0.0)) - cmd.vy))
        abs_wz_err.append(abs(float(info.get("yaw_rate", 0.0)) - cmd.wz))
        if capture and step % FRAME_STRIDE == 0:
            frames.append(env.render(camera, width=width, height=height).copy())

    dt = step * env.cfg.control_dt
    stats = {
        "steps": step,
        "distance": round(float(env.base_pos[0]) - x0, 3),
        "height_gain": round(float(env.base_pos[2]) - z0, 3),
        "speed": round((float(env.base_pos[0]) - x0) / dt, 3) if dt > 0 else 0.0,
        "final_y": round(float(env.base_pos[1]), 3),
        "final_yaw": round(float(env.yaw), 3),
        "mean_cmd": [round(float(v), 3) for v in sum_cmd / max(1, step)],
        # 横向位移总量与偏航角总量，比「末位」更能说明指令有没有被执行
        "drift_y": round(float(env.base_pos[1]) - y0, 3),
        "turn_total": round(float(env.yaw) - yaw0, 3),
        "mean_abs_vy_err": round(float(np.mean(abs_vy_err)), 4),
        "mean_abs_wz_err": round(float(np.mean(abs_wz_err)), 4),
        "success": bool(info.get("is_success", False)),
        "fell": bool(terminated) and not bool(info.get("is_success", False)),
        "timeout": bool(truncated) and not bool(info.get("is_success", False)),
    }
    return stats, frames


#: 三级来源都查不到时的演示默认量程 ``(vy_max, yaw_max)``。
DEFAULT_DEMO_LIMITS = (0.25, 0.35)


def _stage_limits(path: Path, height: float) -> tuple[float, float] | None:
    """从 ``plan.json`` / ``summary.json`` 里按台阶高取 ``(vy_max, yaw_max)``。"""
    data = read_json_or(path, None)
    if data is None:
        return None
    rows = data.get("stages") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        rows = list(data.values()) if isinstance(data, dict) else []  # plan.json
    for row in rows:
        if not isinstance(row, dict) or "command_vy_max" not in row:
            continue
        if abs(float(row.get("step_height", -1.0)) - height) < 1e-9:
            return float(row["command_vy_max"]), float(row.get("command_yaw_max", 0.0))
    return None


def command_limits(stair, raw: dict, model_path) -> tuple[float, float, str]:
    """本回合驱动的那一级的 ``(vy_max, yaw_max, 来源)``。

    优先级：``plan.json``（训练**实际下发**的值）→ ``summary.json``（训练记录）
    → 当前 ``config.yaml`` 的同名级 → 演示默认值。权重自带的 ``_env.json`` 只记
    ``config.yaml`` 的基线（``vy_max``/``yaw_max`` 常常是 0），不能当训练量程用。
    """
    height = round(float(stair.step_height), 6)
    root = Path(model_path).parent
    if root.name == "checkpoints":  # 权重在 checkpoints/ 下，运行目录是上一层
        root = root.parent
    for name in ("plan.json", "summary.json"):
        got = _stage_limits(root / name, height)
        if got is not None:
            return got[0], got[1], name
    for stage in config_to_stages(raw):
        if abs(round(float(stage.step_height), 6) - height) < 1e-9:
            return (float(stage.command_vy_max), float(stage.command_yaw_max),
                    "config.yaml 同名级")
    print("⚠ 该阶段的课程量程无从查证：plan.json / summary.json 里没有这一级，")
    print("  当前 config.yaml 里也没有同名级。")
    return DEFAULT_DEMO_LIMITS[0], DEFAULT_DEMO_LIMITS[1], "演示默认（无记录）"


def main(argv: list[str] | None = None) -> int:
    console.setup_utf8_console()
    parser = argparse.ArgumentParser(description="外部控制器遥控策略（遥感操控）")
    parser.add_argument("--run", required=True, help="运行名 / 目录 / .zip")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=float, default=None,
                        help="台阶高 [cm]，默认用权重自带记录")
    parser.add_argument("--flat", action="store_true",
                        help="换成平地地形（测指令跟踪时用，默认跟着 --hold 走）")
    parser.add_argument("--gif", action="store_true", help="渲染成 GIF")
    parser.add_argument("--hold", type=float, nargs=3, default=None,
                        metavar=("VX", "VY", "WZ"),
                        help="钉死一条指令跑满回合（同时默认切到平地地形）")
    args = parser.parse_args(argv)
    raw = load_config(args.config)  # 早失败：配置有问题就不要等到环境建好才发现

    target = resolve_target(args.run)
    cfg = target.cfg
    use_flat = args.flat or (args.hold is not None and args.height is None)
    if use_flat:
        stair = flat_spec_for()
        terrain_label = "平地"
    else:
        stair = stair_from_height(args.height) or _stair_from_model(target)
        if stair is None:
            raise SystemExit("无法确定地形：请用 --height 指定台阶高 [cm]，或加 --flat")
        terrain_label = f"台阶 {stair.step_height * 100:.1f}cm × {stair.n_up} 级"

    vy_limit, yaw_limit, limit_src = command_limits(stair, raw, target.model_path)
    dev = device(raw)
    runner, env, _venv = open_runner(target, stair=stair, device=dev)

    if args.hold is not None:
        controller = HoldController(vx=args.hold[0], vy=args.hold[1],
                                    wz=args.hold[2])
    else:
        controller = StairController(
            stair, vx=cfg.command.vx,
            vy_limit=vy_limit, yaw_limit=yaw_limit,
        )

    print(f"[遥操作] {target.label}  {terrain_label}  "
          f"指令上界 vy±{vy_limit:g} wz±{yaw_limit:g}（来源 {limit_src}）")
    frames: list[np.ndarray] = []
    for ep in range(args.episodes):
        stats, ep_frames = run_teleop(
            env, runner, controller, seed=args.seed + ep, capture=args.gif,
            camera=CAMERA_GIF, width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
        )
        frames.extend(ep_frames)
        print(f"  回合 {ep + 1}/{args.episodes}  "
              f"前进 {stats['distance']:+.2f}m  速度 {stats['speed']:.2f}m/s  "
              f"横移总量 {stats['drift_y']:+.2f}m  偏航总量 {stats['turn_total']:+.2f}rad")
        print(f"              平均指令 {stats['mean_cmd']}  "
              f"横向误差(均) {stats['mean_abs_vy_err']:.3f}m/s  "
              f"偏航误差(均) {stats['mean_abs_wz_err']:.3f}rad/s  "
              f"{'成功' if stats['success'] else ('摔倒' if stats['fell'] else '超时')}")

    if args.gif and frames:
        out = paths.OUTPUT_GIFS_DIR / f"{target.label}_teleop.gif"
        write_gif(out, frames)
        print(f"  GIF: {out}")
    return 0


def _stair_from_model(target):
    """用权重自带的台阶高造地形（与 ``play.py`` 的行为一致）。"""
    from robotdog.policy import step_height_from_model

    h = step_height_from_model(target.model_path)
    if h is None:
        print("⚠ 权重没有自带地形记录（老权重），退回 3 cm")
        h = 0.03
    return stair_from_height(h * 100.0)


if __name__ == "__main__":
    raise SystemExit(main())

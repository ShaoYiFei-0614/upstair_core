"""第七步: 训练脚本 —— 课程学习 + PPO + 检查点。

用法
----
# 完整课程 (平地 -> 5cm -> 12cm -> 19cm), 8 个 CPU 环境
.venv\\Scripts\\python.exe scripts\\train.py --run-name go2_curriculum --n-envs 8

# 只训平地
.venv\\Scripts\\python.exe scripts\\train.py --stage flat --run-name go2_flat

# 冒烟测试 (几十秒, 验证流水线能跑通)
.venv\\Scripts\\python.exe scripts\\train.py --smoke --run-name smoke

# 从检查点继续
.venv\\Scripts\\python.exe scripts\\train.py --resume runs\\go2_curriculum\\model_final --stage medium

# 打开域随机化 (第八步)
.venv\\Scripts\\python.exe scripts\\train.py --randomize --randomize-from easy

# 残差 RL: 在搜索到的开环 trot 先验上学习 (推荐的训练方式, 理由见 robotdog/gait.py)
.venv\\Scripts\\python.exe scripts\\train.py --gait-prior --run-name go2_gait --n-envs 8

怎么验证
--------
1. 脚本会打印每个阶段的成功率/摔倒率/步数, 并在结束时打印汇总表。
2. ``runs/<名字>/summary.json`` 有机器可读的每阶段结果。
3. ``runs/<名字>/model_final.zip`` + ``vecnormalize.pkl`` 是最终策略。
4. ``runs/<名字>/checkpoints/`` 里是定期检查点。
5. tensorboard: ``.venv\\Scripts\\python.exe -m tensorboard.main --logdir runs\\<名字>\\tensorboard``
6. 用 ``scripts\\eval.py`` 在**没见过的**楼梯上评估 (第九步)。

注意 (沙箱限制)
---------------
本机禁止 Windows 命名管道, ``SubprocVecEnv`` 会报 ``PermissionError: [WinError 5]``。
所以 ``--n-envs 8`` 实际是 ``DummyVecEnv(8)``: 单进程里顺序推进 8 个环境。
实测端到端 ≈ 560 steps/s (8 环境 + PPO), 100 万步 ≈ 30 分钟。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, fields, is_dataclass, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robotdog.curriculum import (  # noqa: E402
    BestModelCallback,
    CurriculumCallback,
    PeriodicCheckpointCallback,
    StageSpec,
    WallClockLimitCallback,
    default_curriculum,
    expand_curriculum,
    mixed_stage_specs,
    mixed_grid_specs,
    select_stages,
    write_curriculum_report,
)
from robotdog.env import EnvConfig  # noqa: E402
from robotdog.gait import GaitParams, load_gait  # noqa: E402
from robotdog.paths import LOGS_DIR  # noqa: E402
from robotdog.training import (  # noqa: E402
    RunDir,
    _find_vecnormalize,
    build_ppo,
    format_duration,
    load_ppo,
    make_vec_env,
    restore_vecnormalize,
    set_terrain,
    set_terrain_per_env,
)
from stable_baselines3.common.callbacks import CallbackList  # noqa: E402


# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Go2 上下楼梯课程学习训练 (PPO)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-name", default=None,
                   help="运行目录名 (默认 runs/go2_<时间戳>)")
    p.add_argument("--stage", default="all",
                   help="阶段: all | flat,easy,medium,hard (逗号分隔)")
    p.add_argument("--n-envs", type=int, default=8,
                   help="环境数 (DummyVecEnv, 单进程顺序推进; 默认 8)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--resume", default=None,
                   help="从已有模型 (.zip 路径, 不带扩展名也可) 继续训练")
    p.add_argument("--total-timesteps", type=int, default=None,
                   help="覆盖整个课程的总步数上限 (按阶段预算等比缩放)")
    p.add_argument("--stage-timesteps", type=int, default=None,
                   help="覆盖每个阶段的步数预算")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="整个运行的墙钟时间上限 (秒), 到点停止")
    p.add_argument("--n-steps", type=int, default=256, help="PPO 每环境 rollout 步数")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--target-kl", type=float, default=0.025,
                   help="PPO 提前停止的 KL 阈值; 调大可用满 n_epochs (实测默认值常在第 0~1 轮就停)")
    p.add_argument("--ent-coef", type=float, default=0.002)
    p.add_argument("--log-std-init", type=float, default=-0.7,
                   help="动作分布初始 log 标准差 (默认 0.0 即 std=1, 对 residual 动作过大)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    p.add_argument("--save-every", type=int, default=50_000,
                   help="每多少总步保存一次检查点")
    p.add_argument("--keep-checkpoints", type=int, default=5)
    p.add_argument("--log-interval", type=int, default=5,
                   help="每多少个 rollout 打印一次 PPO 统计 (rollout/ep_rew_mean 等)")
    p.add_argument("--window", type=int, default=50,
                   help="成功率滑动窗口的回合数")
    p.add_argument("--det-episodes", type=int, default=10, metavar="N",
                   help="每个阶段结束时, 额外用**确定性**动作跑 N 回合量一次成功率 "
                        "(默认 10)。训练时的成功率是采样动作下的, log_std≈0.137 会让它"
                        "大幅低估部署策略 (实测 course15 的 7.8cm 阶段: 课程报 10%%, "
                        "确定性 33%%)。设 0 关闭这个测量。")
    # 第八步
    p.add_argument("--randomize", action="store_true",
                   help="打开全部域随机化 (摩擦/质量/延迟/IMU)")
    p.add_argument("--randomize-from", default="easy",
                   help="从哪个阶段开始启用域随机化 (默认 easy, 即平地不随机化)")
    p.add_argument("--randomize-friction", action="store_true")
    p.add_argument("--randomize-mass", action="store_true")
    p.add_argument("--randomize-latency", action="store_true")
    p.add_argument("--randomize-imu", action="store_true")
    p.add_argument("--dr-ramp-steps", type=int, default=0, metavar="N",
                   help="域随机化严重度的 ramp 长度 (随机化开启时把 randomize_scale "
                        "从 0 线性推到 1)。0=按阶段最少步数 (默认, 即阶段刚允许毕业时 "
                        "刚好满量), N>0=指定步数, N<0=不 ramp 立即全量。"
                        "实测: 不 ramp 时全量随机化 368k 步只到 26%% 采样成功率")
    # ---- residual RL over an open-loop gait ------------------------------- #
    p.add_argument("--gait-prior", action="store_true",
                   help="启用开环步态先验 (残差 RL): "
                        "target = home + clip(gait_gain*gait.radians(t), -1, 1) "
                        "+ action_scale*action  (先验幅度与策略余量刻意解耦)。"
                        "从零训练的 PPO 在 8 个 CPU 环境上无法发现步态 (实测 "
                        "120k/150k 步成功率恒为 0)，而搜索到的 trot 单独就能以 "
                        "1.70 m/s 走且 100%% 存活。")
    p.add_argument("--gait-file", type=Path, default=None,
                   help="步态参数 JSON: 既接受 scripts/search_trot.py 的结果文件, "
                        "也接受 GaitParams.save() 写出的纯参数文件。缺省用内置 trot。")
    p.add_argument("--gait-gain", type=float, default=1.0,
                   help="先验幅度缩放 (<1 给策略留出更多修正余地)")
    p.add_argument("--gait-phase-obs", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="在观测里附加 2 维步态时钟 (sin/cos)。先验开着时几乎必需，"
                        "否则策略无法判断自己处在周期中的哪个相位。")
    p.add_argument("--gait-phase-randomize", action="store_true",
                   help="每次 reset 随机步态起始相位，防止策略过拟合固定相位。")
    # 观测扩展 (用户规格里没有，但上楼梯时是必需的)
    p.add_argument("--obs-base-height", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="观测里附加 1 维『机身离当前地形的高度』。")
    p.add_argument("--obs-foot-height", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="观测里附加 4 维『每只脚离其正下方地形的高度』。"
                        "这是策略判断『前面有没有台阶』的唯一几何信号 —— "
                        "严格按用户规格的观测是**看不到楼梯的**，而规格又禁止接触"
                        "传感器。它不是接触传感器，是足端运动学高度。")
    # --- step 11: 触觉 / 测距 / 直线 -------------------------------- #
    p.add_argument("--obs-touch", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="观测里附加 9 维足端触觉: 4 个按体重归一化的法向接触力 "
                        "+ 4 个打滑比 (切向/法向) + 1 个『有脚在蹭立面』标志。 "
                        "这是规格里被排除的接触信息，用来学『踩实/打滑/撞台阶』。")
    p.add_argument("--obs-stair-probe", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="观测里附加 3 维前向测距: 前方障碍高度、离传感器距离、"
                        "是否命中。与 foot_height 不同，它**在脚到达台阶之前**就能"
                        "看到台阶，是『检测到楼梯就抬腿』的预判信号。")
    p.add_argument("--obs-drift", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="观测里附加 2 维『相对**楼梯中线**的横向偏移』与『相对"
                        "**楼梯轴线** (yaw=0) 的航向误差』。不给这个信号，策略"
                        "无法观测到自己在往侧面漂；奖励里的 straight_line / "
                        "heading / lateral_vel / yaw_rate 四项也全都无从优化。")
    p.add_argument("--obs-base-lin-vel-body", action=argparse.BooleanOptionalAction,
                   default=False, dest="obs_base_lin_vel_body",
                   help="观测里附加 3 维**机体系**机身线速度 (qvel[0:3] 左乘 R^T)。"
                        "奖励里 lateral_vel / lateral_vel_track / yaw_rate 都是 "
                        "vy、wz 的函数，而默认观测里 **vy 完全不可见** —— 策略被"
                        "它看不见的量打分，只能靠关节速度和 10 帧历史反推，这是"
                        "『走不直怎么也修不好』的结构性原因。世界系版本 (base_lin_vel)"
                        "会随机身 yaw 改变含义，不适合做前进命令跟踪。")
    p.add_argument("--obs-foot-lift-gap", action=argparse.BooleanOptionalAction,
                   default=False, dest="obs_foot_lift_gap",
                   help="观测里附加 4 维『每只脚**上一轮摆动**离越过台阶还差多少』"
                        "（按 lift_goal 归一化, clip 到 [0,1], 0=已经抬够）。用"
                        "**摆动峰值**结算，所以支撑相也携带『这只脚上一轮尝试的"
                        "结果』。")
    p.add_argument("--obs-foot-lift-deficit", action=argparse.BooleanOptionalAction,
                   default=False, dest="obs_foot_lift_deficit",
                   help="观测里附加 4 维『每只脚**此刻**还要再抬高多少才能越过它"
                        "将踩上去的那一级』（按 lift_goal 归一化, clip 到 [0,1]）。"
                        "参照面是**落脚面**（脚当前位置与落脚区间内最高的踏面），"
                        "见 --cfg reward.foot_placement_reach。与上一条的分工是"
                        "『当下 / 回顾』：只有 gap 时策略要等落地才知道失败原因，"
                        "只有 deficit 时支撑相读到的是与刚做过的动作无关的数。")
    # --- 奖励归一化 -------------------------------------------------- #
    p.add_argument("--norm-reward", action=argparse.BooleanOptionalAction,
                   default=False, dest="norm_reward",
                   help="让 VecNormalize 同时归一化奖励（默认关闭，与原行为一致）。"
                        "课程在阶段之间切换地形时，在 5cm 上学到的奖励尺度到了 "
                        "11cm 会漂移，所以默认关闭。开启后 critic 目标更可学："
                        "3cm 台阶 150k 步实测成功率 0.474->0.539、残差 mean|a| "
                        "0.070->0.080；但这是**温和的稳定器而非解药**，两侧在 "
                        "11cm 上都没有成功，所以不要指望它单独把台阶爬通。")
    # --- 终止条件 ---------------------------------------------------- #
    p.add_argument("--terminate-contact", default="trunk",
                   choices=["any", "trunk", "none"], metavar="{any,trunk,none}",
                   help="哪些『非足端 geom 碰地形』算致命。trunk (默认) = 只有"
                        "**躯干**撞地形才终止，小腿蹭立面只罚分不掐回合；"
                        "any = 旧行为（任何非足端 geom 一碰就结束）；"
                        "none = 接触从不终止。"
                        "实测 (course12, 每种地形 30 回合): 10cm×8 上策略平均只"
                        "走 62.6 步、最远 1.42m，而首级立面在 1.50m —— 它在脚还"
                        "没踩到任何一级踏面之前就被自己的小腿判死，所以「高台阶"
                        "完全没有试错机会」。")
    # 其他
    p.add_argument("--mixed-heights", default=None, metavar="LO,HI",
                   help="同阶段混合地形: 每个楼梯阶段的 N 个子环境各自拿一个"
                        "从 LO 到 HI (cm) 线性排开的台阶高度, 整个高度带都会"
                        "出现在每一次 PPO 梯度更新里。这是为了解决逐级课程"
                        "的灾难性遗忘: course19 每一档只在自己训练的那一级最强"
                        "(m3 在 9.0cm 65%%、m4 在 9.5cm 45%%、m5 在 10.0cm 20%%)"
                        "而 7.5~8.0cm 反而比 course18 差 (35/65%% vs 85/75%%)。"
                        "建议与 --sub-steps 1 合用。")
    p.add_argument("--mixed-treads", default=None, metavar="LO,HI",
                   help="同阶段混合地形 (第二个难度轴): 与 --mixed-heights 交叉"
                        "成一个「台阶高度 x 踏面深度」网格, 每个子环境拿一格。"
                        "实测 (.tmp/step2b_probe.py, course21 final, 20 回合, "
                        "爬升总距离恒为 4.0m 以排除 20s 时限): 踏面 35/50/65/80cm "
                        "在 10.0cm 上分别是 0%%/55%%/0%%/5%% —— **加长踏面不能让"
                        "机器狗每一级多停留一会儿**, 反而把 50cm 训练出来的步态"
                        "相位彻底打乱。真正需要的是让「登上一级」这件事不再依赖"
                        "某一个踏面/相位, 所以把踏面也放进训练分布里。"
                        "建议 0.35,0.65。")
    p.add_argument("--cfg", action="append", default=[],
                   metavar="KEY=VALUE", help="覆盖 EnvConfig 字段 (可重复)")
    p.add_argument("--smoke", action="store_true",
                   help="冒烟模式: 极小预算, 只验证流水线")
    p.add_argument("--no-advance-on-budget", action="store_true",
                   help="阶段预算耗尽且未达标时**不**推进 (默认推进)")
    p.add_argument("--sub-steps", type=int, default=1, metavar="N",
                   help="把每个楼梯阶段再拆成 N 个台阶高度递增的梯级 (默认 1=不拆)。"
                        "从 max(2cm, 40%%高度) 线性升到该阶段的完整高度。")
    p.add_argument("--sub-start-cm", type=float, default=None, metavar="H",
                   help="子课程第一级的台阶高度 (cm)。不给则用 40%% 规则。"
                        "楼梯先验在 4cm 上 100%% 成功、2~3cm 反而只有 40%%, 所以起手"
                        "建议 --sub-start-cm 4。")
    p.add_argument("--noise-start-frac", type=float, default=0.0, metavar="F",
                   help="出生噪声课程: 第一个阶段只用 F 倍的 reset_*_noise, 线性升到"
                        "最后一个阶段的 1.0 倍 (默认 0=不做噪声课程)。"
                        "理由: 实测开环楼梯 trot 在 4cm 上零噪声 100%% 成功, 但加上"
                        "训练默认的 yaw/joint 噪声就掉到 25%% —— 它是刀刃上的谐振, "
                        "一开始就要求策略在 0.12m 出生偏移下爬楼是学不动的。")
    p.add_argument("--sub-n-up", type=int, default=None, metavar="K",
                   help="子课程第一级的台阶**级数** (默认不拆级数, 一直用完整级数)。"
                        "从 K 级线性升到该阶段的完整级数。理由: 终局成功奖励是"
                        "「爬完整段楼梯」的全有全无信号; 8 级 x 单级存活 0.85 = 27%%, "
                        "正是前四次训练卡住的平台。从 2 级起手能把同一份单级能力变成"
                        "~72%% 成功率, 策略才会收到密集的 +20 终局奖励。")
    p.add_argument("--stage-top-cm", type=float, default=None, metavar="H",
                   help="把所有楼梯阶段的**最高**台阶高度压到 H cm。"
                        "内置课程里 medium 的顶层是 12cm, 但实测策略的前沿只有 ~8cm, "
                        "于是 --sub-steps 拆出来的梯级有一半落在无人能过的 9~12cm 上、"
                        "白烧预算。压顶之后梯级才会集中在真正的前沿上。")
    p.add_argument("--min-stage-frac", type=float, default=0.5, metavar="F",
                   help="每个阶段必须先跑掉 F 倍预算才允许毕业 (默认 0.5)。"
                        "理由 (实测): 有强先验时**未训练**的策略就已经满足成功率目标, "
                        "于是阶段在 1 个 rollout 内就毕业 —— PPO 每次更新要 "
                        "n_steps*n_envs 步, easy1 只跑了 1,824 步(0 次更新)、"
                        "easy4 只跑了 3,456 步(1 次更新), 网络停在初始化, 策略输出"
                        "恒为 ~0.011(即「什么都不做, 让先验走」)。设 0 可恢复旧行为。")
    return p.parse_args(argv)


def _cfg_target(cfg: EnvConfig, key: str):
    """把 ``--cfg`` 的键解析成 ``(宿主对象, 属性名)``，支持点号进入嵌套 dataclass。

    例如 ``reward.alive=0.3`` -> ``(cfg.reward, "alive")``；
    ``obs.base_height=true`` -> ``(cfg.obs, "base_height")``。
    """
    parts = [p.strip() for p in key.split(".") if p.strip()]
    if not parts:
        raise SystemExit(f"--cfg 的键不能为空: {key!r}")
    owner = cfg
    for part in parts[:-1]:
        if not hasattr(owner, part):
            raise SystemExit(
                f"未知 EnvConfig 字段 {key!r}: {'.'.join(parts[:-1])} 里没有 {part!r}"
            )
        owner = getattr(owner, part)
    attr = parts[-1]
    if not hasattr(owner, attr) or callable(getattr(owner, attr)):
        raise SystemExit(
            f"未知 EnvConfig 字段 {key!r}; {type(owner).__name__} 的可用字段: "
            f"{[f.name for f in fields(owner)] if is_dataclass(owner) else sorted(vars(owner))}"
        )
    return owner, attr


def apply_cfg_overrides(cfg: EnvConfig, pairs: list[str]) -> None:
    """``--cfg kp=[40,40,60]`` 之类; 用 dataclass 字段类型做安全转换。

    键支持点号路径进入嵌套 dataclass: ``--cfg reward.alive=0.3``、
    ``--cfg obs.base_height=true``、``--cfg gait.freq=5.5``。
    """
    for item in pairs:
        if "=" not in item:
            raise SystemExit(f"--cfg 需要 KEY=VALUE 形式, 收到 {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        owner, attr = _cfg_target(cfg, key)
        cur = getattr(owner, attr)
        try:
            if isinstance(cur, bool):
                value = raw.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(cur, int) and not isinstance(cur, bool):
                value = int(raw)
            elif isinstance(cur, float):
                value = float(raw)
            elif isinstance(cur, tuple):
                value = tuple(float(v) for v in raw.strip("()[]").split(","))
            else:
                value = raw
        except ValueError as exc:
            raise SystemExit(f"无法解析 {key}={raw!r}: {exc}") from exc
        setattr(owner, attr, value)
        print(f"[配置] EnvConfig.{key} = {value!r}")


def _quote_arg(a: str) -> str:
    """Quote a single argv element for a Windows shell, only when needed."""
    if a == "" or any(ch in a for ch in ' \t"'):
        return '"' + a.replace('"', '\\"') + '"'
    return a


def reproduce_command(argv: list[str] | None) -> str:
    """The exact command line that produced this run.

    This used to be hand-curated from five arguments (``--run-name``,
    ``--n-envs``, ``--seed``, ``--stage``), which meant the ``reproduce`` field
    of ``summary.json`` looked authoritative but silently dropped every flag
    that actually shapes the model -- ``--stage-top-cm``, ``--mixed-heights``,
    ``--resume``, the ``--obs-*`` switches, ``--gait-prior`` and all ``--cfg``
    overrides.  Re-running it produced a *different* robot.  Echo the real argv
    instead, so the recorded command is honest by construction.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    return " ".join([sys.executable, "scripts/train.py"]
                    + [_quote_arg(a) for a in args])


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    t_start = time.time()

    run_name = args.run_name or f"go2_{time.strftime('%Y%m%d_%H%M%S')}"
    run = RunDir(run_name)

    print("=" * 78)
    print(f"Go2 上下楼梯强化学习训练   run = {run_name}")
    print("=" * 78)
    print(f"运行目录        : {run.root}")
    print(f"环境数          : {args.n_envs}  (DummyVecEnv, 单进程顺序推进)")
    print(f"设备            : {args.device}")
    print(f"随机种子        : {args.seed}")
    print(f"多进程          : 不可用 (沙箱禁止命名管道) -> 已退化为单进程")
    print("=" * 78)

    # ---------------------------------------------------------------- config
    cfg = EnvConfig()
    if args.gait_prior:
        gait = load_gait(args.gait_file) if args.gait_file else GaitParams.default_trot()
        cfg.gait_prior = True
        cfg.gait = gait
        cfg.gait_gain = args.gait_gain
        cfg.gait_phase_randomize = args.gait_phase_randomize
        cfg.obs.gait_phase = bool(args.gait_phase_obs)
        print(f"[步态先验] 已启用 residual RL: {gait.summary()}")
        print(f"[步态先验] gain={cfg.gait_gain:.2f}  先验幅度={cfg.gait.action_scale:.2f} rad  "
              f"策略余量 action_scale={cfg.action_scale:.2f} rad  "
              f"观测步态时钟={cfg.obs.gait_phase} (每帧 {cfg.obs.frame_size} 维)")
        print(f"[步态先验] 两者刻意分开: 先验幅度改了不会顺带放大策略的探索噪声")

    # 观测扩展：默认关闭以严格匹配用户规格；上楼梯时必须打开 foot_height，
    # 否则策略在几何上无法知道前方有台阶。
    cfg.obs.base_height = bool(args.obs_base_height)
    cfg.obs.foot_height = bool(args.obs_foot_height)
    cfg.obs.touch = bool(args.obs_touch)
    cfg.obs.stair_probe = bool(args.obs_stair_probe)
    cfg.obs.drift = bool(args.obs_drift)
    cfg.obs.base_lin_vel_body = bool(args.obs_base_lin_vel_body)
    cfg.obs.foot_lift_gap = bool(args.obs_foot_lift_gap)
    cfg.obs.foot_lift_deficit = bool(args.obs_foot_lift_deficit)
    cfg.terminate_contact = str(args.terminate_contact)
    if cfg.obs.foot_height or cfg.obs.base_height:
        print(f"[观测] 已扩展: base_height={cfg.obs.base_height} "
              f"foot_height={cfg.obs.foot_height} -> 每帧 {cfg.obs.frame_size} 维, "
              f"history={cfg.obs.history_len} -> 共 {cfg.obs.size} 维")
        print("[观测] 注意: 这不属于用户规格里的 5 个块, 但规格禁止接触传感器, "
              "而『足端离地高度』是策略判断台阶位置的唯一几何信号。")
    if cfg.obs.touch or cfg.obs.stair_probe or cfg.obs.drift:
        print(f"[观测/触觉] 已扩展: touch={cfg.obs.touch} (9 维足端法向力+打滑+蹭立面), "
              f"stair_probe={cfg.obs.stair_probe} (3 维前向测距), "
              f"drift={cfg.obs.drift} (2 维横向偏移+航向误差)")
        print(f"[观测/触觉] -> 每帧 {cfg.obs.frame_size} 维, history={cfg.obs.history_len}, "
              f"共 {cfg.obs.size} 维")
    if cfg.obs.base_lin_vel_body:
        print(f"[观测/直线] 已加入 3 维机体系线速度 (vx, vy, vz) -> 每帧 "
              f"{cfg.obs.frame_size} 维, 共 {cfg.obs.size} 维。奖励里的 "
              f"lateral_vel / lateral_vel_track / yaw_rate 现在都可观测。")
    print(f"[终止] 地形接触终止策略 = {cfg.terminate_contact!r} "
          f"(trunk=只有躯干撞地才终止; any=任何非足端 geom 碰地都终止; none=不终止)。"
          f"腿蹭地形每步罚 {cfg.reward.leg_collision:.2f}。")
    if cfg.reward.stair_probe_lift != 0.0 and not cfg.obs.stair_probe:
        print("[观测/触觉] ⚠ 奖励项 stair_probe_lift 开着, 但观测块 stair_probe "
              "关着: 策略看不到『前方有台阶』这个触发量, 该项对它只是噪声, "
              "建议同时加 --obs-stair-probe。")

    apply_cfg_overrides(cfg, args.cfg)

    if cfg.terminate_contact != "none" and cfg.terminate_contact_steps <= 1:
        print("[终止/建议] ⚠ terminate_contact_steps=1: 躯干碰到地形就立刻终止。"
              "实测 9~10 cm 台阶上, 狗抬头爬第一级立面时胸口会蹭到立面, 此时机身"
              "高度 0.26 m、|pitch| 12~16°、按几何判据根本没摔, 单步终止却把回合"
              "从 ~290 步砍到 ~190 步。爬 8 cm 以上建议加 "
              "--cfg terminate_contact_steps=10 (0.2 s)。")

    stages = select_stages(default_curriculum(), [s.strip() for s in
                                                 args.stage.split(",")])
    # ``--stage-top-cm``: 把每个楼梯阶段的顶层高度压下来, 让子课程梯级集中在
    # 策略真正的前沿上 (medium 的顶层是 12cm, 而实测前沿只有 ~8cm)。
    if args.stage_top_cm is not None:
        top = max(0.0, float(args.stage_top_cm) / 100.0)
        for s in stages:
            if s.spec.n_up > 0 and s.spec.step_height > top:
                old_cm = s.spec.step_height * 100
                s.spec = replace(s.spec, step_height=top)
                print(f"[课程] --stage-top-cm: {s.name} 顶层 {old_cm:.1f}cm -> "
                      f"{top * 100:.1f}cm")

    if args.smoke:
        for s in stages:
            s.min_episodes = 2
            s.max_timesteps = 256 * 2
            s.target_success = 0.99
        args.n_steps = min(args.n_steps, 32)
        args.batch_size = min(args.batch_size, 64)
        args.n_epochs = 2
        args.save_every = 256
        args.window = 10
        args.log_interval = 1
    if args.stage_timesteps is not None:
        for s in stages:
            s.max_timesteps = args.stage_timesteps
    # 子课程: 把每个楼梯阶段再拆成「台阶高度递增」的若干梯级。用户第六步要求
    # 「每阶段成功率高就升难度」——单个 5 cm 台阶对只会抬 20 mm 的先验来说是
    # 一堵墙, 先给 2 cm 再逐级升到 5 cm 才有可学的信号。
    if args.sub_steps > 1:
        coarse = list(stages)
        stages = expand_curriculum(
            stages, args.sub_steps,
            start_step_height=(None if args.sub_start_cm is None
                               else args.sub_start_cm / 100.0),
            n_up_start=args.sub_n_up,
        )
        for st in coarse:
            if st.spec.n_up > 0:
                sub = [s for s in stages if s.name.startswith(st.name)]
                heights = [round(s.spec.step_height * 100, 1) for s in sub]
                n_ups = [s.spec.n_up for s in sub]
                print(f"[子课程] {st.name}: 台阶高度梯级 {heights} cm, "
                      f"级数梯级 {n_ups} (总预算 {st.max_timesteps:,} 均分)")
    if args.total_timesteps is not None:
        total_default = sum(s.max_timesteps for s in stages)
        scale = args.total_timesteps / max(total_default, 1)
        for s in stages:
            s.max_timesteps = max(256, int(s.max_timesteps * scale))

    # 最少训练步数下限: 有强先验时未训练的策略就已经满足成功率目标, 阶段会在
    # 1 个 rollout 内毕业, PPO 拿不到任何梯度更新。见 StageSpec.min_timesteps。
    if args.min_stage_frac > 0.0:
        for s in stages:
            s.min_timesteps = max(1, int(s.max_timesteps * args.min_stage_frac))
        n_updates = max(1, args.n_steps * args.n_envs)
        floor_steps = min(s.min_timesteps for s in stages)
        print(f"[课程] 最少训练步数下限 = {args.min_stage_frac:.0%} 预算 "
              f"(最紧的一档 {floor_steps:,} 步 ≈ {floor_steps / n_updates:.1f} 次 PPO 更新, "
              f"每次 {n_updates:,} 步)")

    print("\n课程阶段:")
    for i, s in enumerate(stages):
        print(f"  [{i}] {s.summary()}")
    print()

    # ------------------------------------------------------------- vec env(s)
    venv = make_vec_env(stages[0].spec, n_envs=args.n_envs, cfg=cfg,
                        seed=args.seed, normalize=True, monitor=True,
                        norm_reward=bool(args.norm_reward))

    # ---------------------------------------------------------------- model
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.suffix != ".zip":
            resume_path = resume_path.with_suffix(".zip")
        if not resume_path.exists():
            raise SystemExit(f"找不到要恢复的模型: {resume_path}")
        print(f"[恢复] 从 {resume_path} 加载策略权重")
        model = load_ppo(resume_path, venv, device=args.device)
        # PPO.load 只还原网络权重, 观测归一化统计 (VecNormalize) 是独立的
        # 一个 pickle。不还原它, 策略就会拿到按 mean=0/var=1 归一化的输入,
        # 而它训练时看到的是真统计 —— 相当于换了一套输入坐标系, 恢复出来
        # 的策略基本废掉。save_model() 把统计写成 "<模型>_vecnormalize.pkl",
        # 所以这里按同样的命名去找; 找不到就明确警告而不是悄悄继续。
        vn_path = resume_path.with_name(resume_path.stem + "_vecnormalize.pkl")
        if restore_vecnormalize(venv, vn_path):
            vec_norm = _find_vecnormalize(venv)
            count = float(vec_norm.obs_rms.count) if vec_norm is not None else 0.0
            print(f"[恢复] 已还原观测归一化统计 {vn_path.name} "
                  f"(obs_rms count={count:.0f})")
        else:
            print(f"[恢复] ⚠ 没找到或无法加载 {vn_path.name} —— 观测归一化统计"
                  f"会从零重新统计, 恢复出来的策略前若干步表现会明显变差。"
                  f"如果只是想续训同一套观测, 请确认该文件存在。")
        # PPO.load 会把模型里存着的 ``tensorboard_log`` **一并还原**, 而那是
        # 上一个 run 的目录。不覆盖它, 本次训练的曲线就会写进旧 run 里 ——
        # course15 续训 course14 时就真的写进了 runs\course13\tensorboard,
        # 把 course13 的交付曲线污染掉。这里强制改回本次 run 的目录。
        model.tensorboard_log = str(run.tensorboard)
        print(f"[恢复] tensorboard 目录改回本次 run: {run.tensorboard}")
    else:
        model = build_ppo(
            venv,
            seed=args.seed,
            n_steps=args.n_steps,
            batch_size=min(args.batch_size, args.n_steps * args.n_envs),
            n_epochs=args.n_epochs,
            learning_rate=args.learning_rate,
            target_kl=args.target_kl,
            ent_coef=args.ent_coef,
            log_std_init=args.log_std_init,
            gamma=args.gamma,
            net_arch=args.net_arch,
            tensorboard_log=run.tensorboard,
            device=args.device,
        )

    print(f"[模型] 观测维度={venv.observation_space.shape} "
          f"动作维度={venv.action_space.shape} "
          f"网络={args.net_arch} n_steps={args.n_steps} "
          f"batch={model.batch_size} epochs={args.n_epochs}")
    print(f"[模型] 可训练参数量 = "
          f"{sum(p.numel() for p in model.policy.parameters()):,}\n")

    # ------------------------------------------------------------- curricula
    results: list[dict] = []
    randomization_on = False
    # 跨阶段的「最佳」追踪。model_best.zip 只由**毕业过的**阶段写入 (最深的那个),
    # 因为成功率不能跨地形比较: 平地的 100% 并不意味着它会上楼梯。
    best_so_far = -1.0
    best_stage: str | None = None
    best_stage_rate = -1.0
    # 出生噪声课程的标称值 (--cfg 覆盖之后的值), 之后按阶段线性缩放。
    noise_nominal = {
        "xy": float(cfg.reset_xy_noise),
        "yaw": float(cfg.reset_yaw_noise),
        "joint": float(cfg.reset_joint_noise),
    }

    # ---------------------------------------------------------------- summary
    # 先把 summary 的构造抽成闭包: (a) 训练**一开始**就写一份 provisional
    # summary.json, 这样**进行中**的 run 也能被 eval.py / probe_policy_action.py
    # 正确加载（否则它们只能从 .zip 的观测维度反推,--cfg 覆盖过的超参如
    # episode_length_s / action_scale 就丢了); (b) 每个阶段结束后重写一次。
    def build_summary() -> dict:
        total = int(model.num_timesteps)
        wall_now = time.time() - t_start
        return {
            "run_name": run_name,
            "seed": args.seed,
            "n_envs": args.n_envs,
            "device": args.device,
            "vec_env": "DummyVecEnv (单进程; 沙箱禁止命名管道, SubprocVecEnv 不可用)",
            "stages": results,
            "finished": len(results) >= len(stages),
            "total_timesteps": total,
            "wall_seconds": round(wall_now, 1),
            "steps_per_second": round(total / max(wall_now, 1e-9), 1),
            "randomization_enabled": randomization_on,
            "best_stage": best_stage,
            "best_stage_success_rate": (round(best_stage_rate, 4)
                                        if best_stage else None),
            "note_model_best": ("model_best.zip 只保存『毕业过的』阶段中最深的那个; "
                                "各阶段自己的最佳在 artifacts/best_<stage>.zip"),
            # ``gait`` 要留着 (nested dataclass -> dict), 否则恢复时先验会丢,
            # 机器人只会站着不动; ``obs``/``reward`` 有各自的顶层键。
            "env_config": {
                k: (list(v) if isinstance(v, tuple) else v)
                for k, v in asdict(cfg).items() if k not in ("obs", "reward")
            },
            "obs_config": asdict(cfg.obs),
            "reward_config": asdict(cfg.reward),
            "artifacts": {
                "final_model": str(run.final_model) + ".zip",
                "best_model": str(run.best_model) + ".zip",
                "vecnormalize": str(run.final_model) + "_vecnormalize.pkl",
                "checkpoints": str(run.checkpoints),
                "tensorboard": str(run.tensorboard),
            },
            "reproduce": reproduce_command(argv),
        }

    # 训练一开始就落一份 provisional summary.json: 让**进行中**的 run 也能被
    # eval.py / probe_policy_action.py 用训练时的真实 EnvConfig 加载。
    run.write_summary(build_summary())
    print(f"[汇总] 已写入 provisional summary: {run.summary}\n")

    try:
        for idx, stage in enumerate(stages, start=1):
            if args.max_seconds is not None and time.time() - t_start > args.max_seconds:
                print(f"[限时] 已超过 --max-seconds={args.max_seconds:.0f}s, "
                      f"跳过剩余阶段")
                break

            print("-" * 78)
            print(f"[{idx}/{len(stages)}] 切换到阶段 '{stage.name}' ({stage.label})")
            print("-" * 78)

            # ---- 出生噪声课程: 前几个阶段先在小噪声里学, 再线性放大到标称值。
            # 所有 8 个子环境共享同一个 EnvConfig 对象, 所以改这里就是改全部。
            if args.noise_start_frac > 0.0 and len(stages) > 1:
                frac = args.noise_start_frac + (
                    (1.0 - args.noise_start_frac) * (idx - 1) / (len(stages) - 1)
                )
                cfg.reset_xy_noise = noise_nominal["xy"] * frac
                cfg.reset_yaw_noise = noise_nominal["yaw"] * frac
                cfg.reset_joint_noise = noise_nominal["joint"] * frac
                print(f"[噪声课程] 倍率 {frac:.2f} -> xy={cfg.reset_xy_noise:.4f} "
                      f"yaw={cfg.reset_yaw_noise:.4f} joint={cfg.reset_joint_noise:.4f}")

            if args.mixed_heights and stage.spec.n_up > 0:
                lo_hi = [float(v) for v in str(args.mixed_heights).split(",")]
                if args.mixed_treads:
                    tr_lo_hi = [float(v) for v in
                                str(args.mixed_treads).split(",")]
                    mixed = mixed_grid_specs(
                        stage.spec, venv.num_envs,
                        heights_cm=(lo_hi[0], lo_hi[-1]),
                        treads=(tr_lo_hi[0], tr_lo_hi[-1]))
                else:
                    mixed = mixed_stage_specs(
                        stage.spec, venv.num_envs, lo_hi[0], lo_hi[-1])
                set_terrain_per_env(venv, mixed)
                print("[混合地形] 各子环境 "
                      + " / ".join(f"{s.step_height * 100:.1f}cm"
                                   f"@{s.tread * 100:.0f}cm" for s in mixed))
            else:
                set_terrain(venv, stage.spec)
            print(f"[地形] 编译完成: ngeom={venv.get_attr('model')[0].ngeom}, "
                  f"台阶数={len(venv.get_attr('meta')[0].stair_geom_ids)}, "
                  f"顶端高度={stage.spec.crest_height:.2f} m")

            # ---- step 8: turn domain randomization on at the requested stage
            if args.randomize and not randomization_on:
                order = [s.name for s in stages]
                from_idx = (
                    order.index(args.randomize_from)
                    if args.randomize_from in order
                    else 0
                )
                if (idx - 1) >= from_idx:
                    venv.env_method(
                        "set_domain_randomization",
                        randomize_friction=True,
                        randomize_mass=True,
                        randomize_latency=True,
                        randomize_imu=True,
                    )
                    randomization_on = True
                    print(f"[域随机化] 已启用: 摩擦 "
                          f"{cfg.randomize_friction_range[0]}~"
                          f"{cfg.randomize_friction_range[1]}x, 质量 "
                          f"{cfg.randomize_mass_range[0]}~"
                          f"{cfg.randomize_mass_range[1]}x, 控制延迟 0~"
                          f"{cfg.randomize_latency_max} 步, IMU 噪声 sigma="
                          f"{cfg.randomize_imu_sigma}")
            elif not args.randomize and any(
                getattr(args, f"randomize_{k}") for k in
                ("friction", "mass", "latency", "imu")
            ):
                flags = {k: getattr(args, f"randomize_{k}") for k in
                         ("friction", "mass", "latency", "imu")}
                venv.env_method(
                    "set_domain_randomization",
                    **{f"randomize_{k}": v for k, v in flags.items()},
                )
                randomization_on = True
                print(f"[域随机化] 已按开关启用: {flags}")

            curriculum_cb = CurriculumCallback(
                stage, window=args.window,
                advance_on_budget=not args.no_advance_on_budget,
                cfg=cfg, stage_index=0 if args.det_episodes <= 0 else idx,
                det_episodes=args.det_episodes,
                # The callback owns the severity ramp so it can be tied to the
                # stage budget and logged next to the success rate.
                randomization_on=randomization_on,
                dr_ramp_steps=args.dr_ramp_steps,
            )
            best_cb = BestModelCallback(
                run.artifacts / f"best_{stage.name}", curriculum_cb,
                floor=best_so_far,
            )
            callbacks = [
                curriculum_cb,
                PeriodicCheckpointCallback(
                    run.checkpoints / stage.name,
                    save_every=max(args.save_every, model.n_steps * args.n_envs),
                    name_prefix=f"{stage.name}",
                    keep=args.keep_checkpoints,
                ),
                best_cb,
            ]
            if args.max_seconds is not None:
                callbacks.append(
                    WallClockLimitCallback(
                        max(10.0, args.max_seconds - (time.time() - t_start))
                    )
                )

            t_stage = time.time()
            model.learn(
                total_timesteps=stage.max_timesteps,
                callback=CallbackList(callbacks),
                log_interval=args.log_interval,
                tb_log_name="ppo",
                reset_num_timesteps=False,
                progress_bar=False,
            )
            stage_seconds = time.time() - t_stage

            res = curriculum_cb.result()
            res["wall_seconds"] = round(stage_seconds, 1)
            res["steps_per_second"] = round(
                res["timesteps"] / max(stage_seconds, 1e-9), 1
            )
            res["best_success_rate"] = round(max(best_cb.best, 0.0), 4)
            res["best_checkpoint"] = str(run.artifacts / f"best_{stage.name}")
            results.append(res)
            best_so_far = max(best_so_far, best_cb.best)
            if res.get("graduated"):
                best_stage = stage.name
                best_stage_rate = best_cb.best
                run.save_model(model, run.best_model, venv)
                print(f"[最佳] 阶段 '{stage.name}' 毕业 -> model_best.zip 更新为它")
            print(f"[{idx}/{len(stages)}] 阶段 '{stage.name}' 用时 "
                  f"{format_duration(stage_seconds)} "
                  f"({res['steps_per_second']:.0f} steps/s)")
            print(f"           {res['stop_reason']}\n")

            # Per-stage snapshot so a stage can be resumed in isolation.
            run.save_model(model, run.checkpoints / f"{stage.name}_final", venv)
            # 让进行中的 run 也能被 eval.py / probe 加载
            run.write_summary(build_summary())

    except KeyboardInterrupt:
        print("\n[中断] 收到 Ctrl+C, 保存当前模型后退出...")
        run.save_model(model, run.final_model, venv)
        return 130
    finally:
        pass

    # ------------------------------------------------------------------ save
    run.save_model(model, run.final_model, venv)
    summary = build_summary()
    total_steps = summary["total_timesteps"]
    wall = summary["wall_seconds"]

    run.write_summary(summary)
    write_curriculum_report(LOGS_DIR / f"{run_name}_curriculum.json", results,
                            {"run_name": run_name, "total_timesteps": total_steps})

    # ----------------------------------------------------------------- report
    print("=" * 78)
    print(f"训练完成   总步数 = {total_steps:,}   用时 = {format_duration(wall)}   "
          f"平均 = {summary['steps_per_second']:.0f} steps/s")
    print("=" * 78)
    print(f"{'阶段':<8}{'台阶高':>8}{'回合':>7}{'成功率':>9}{'摔倒率':>9}"
          f"{'均回报':>10}{'步数':>11}{'毕业':>6}")
    for r in results:
        print(f"{r['stage']:<8}{r['step_height'] * 100:>7.1f}c{r['episodes_total']:>7d}"
              f"{r['success_rate']:>9.1%}{r['fall_rate']:>9.1%}"
              f"{r['mean_return']:>10.2f}{r['timesteps']:>11,}"
              f"{'是' if r['graduated'] else '否':>6}")
    print()
    print("产物:")
    for key, path in summary["artifacts"].items():
        # model_best.zip 只在有阶段**毕业**时才存在; 直接打印会让用户去找一个
        # 并不存在的文件 (冒烟训练里四个阶段都没毕业时就会这样)。
        if key == "best_model" and not Path(path).exists():
            print(f"  {key:<14} (无: 没有任何阶段毕业; 各阶段最佳见 artifacts/best_*.zip)")
            continue
        print(f"  {key:<14} {path}")
    print(f"  {'summary':<14} {run.summary}")
    print()
    print("下一步验证:")
    print(f"  {sys.executable} scripts/eval.py --run {run.root} --only trained --episodes 20")
    print(f"  {sys.executable} scripts/eval.py --run {run.root} --only unseen --episodes 20 "
          f"--export both")
    print(f"  {sys.executable} scripts/diag_reward.py --run {run.root}")
    print(f"  {sys.executable} scripts/probe_policy_action.py "
          f"--model {run.best_model} --gait-file artifacts\\trot_stairs.json "
          f"--episode-length-s {cfg.episode_length_s}")
    print(f"  {sys.executable} -m tensorboard.main --logdir {run.tensorboard}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

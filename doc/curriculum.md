# `robotdog/curriculum.py` — 课程数据与报表

对应代码：`robotdog/curriculum.py`。课程设计的实测教训见 `experience.md` §4、§17、§21；
本文只讲抽象结构。

## 职责

**只负责「训哪几级、按什么顺序、训多少步」的数据与报表**。真正的 PPO 训练在
`robotdog/training.py` / `scripts/train.py`；本模块没有任何训练循环。

## 核心抽象

### 串行课程哲学

每个阶段只对应**一种**台阶几何（无高度带、阶段内无随机化）；在该级预算上训练、
保存、进入下一级，下一级 warm-start 上一级权重。阶段之间切换地形必须重新编译
MJCF 并 reset，所以切换只发生在两次 `train_stage` 之间，绝不打断 rollout。

### StageSpec：一级的全部参数

地形、预算、学习率，加上**逐级下发的指令量程**（`command_vx` / `command_vy_max` /
`command_yaw_max`）与**奖励覆盖**（`reward_overrides`）。这些逐级值是回放口径的
权威来源 —— `summary.json` / `_env.json` 里只有 `config.yaml` 的基线，拿基线去评
老运行会判出与训练时不同的成败（`experience.md` §17）。

### EpisodeStats：滑窗回合统计

从 `VecMonitor` 写的 `info["episode"]` 吃回合记录（`add_infos` 可无脑批量喂），
维护成功率/摔倒率/回报/步数的滑动窗口，供训练回调打印进度行。

### 报表与汇总

`format_curriculum`（定宽表格）与 `write_summary`（`summary.json`）：课程定义 +
每级结果 + 完整配置。`write_summary` 逐级记录指令量程，中文不转义。

## 踩坑指引

课程「毕业」判据饿死梯度更新、平地预热污染爬楼能力 —— 见 `experience.md`
§4.1、§17。

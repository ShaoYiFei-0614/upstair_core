# Go2 盲走楼梯 · 残差强化学习仓

在一段 **8 级、踏面 50 cm** 的楼梯上训练 Unitree Go2 上楼梯。两个刻意设定：

* **盲走（blind）**。整套代码没有任何触觉、射线或接触力读取。机器人只看本体
  感觉（关节角/角速度、机身姿态与角速度、上一步动作、机身系线速度）加一个
  **台阶高度标量**，靠自己撞出怎么上台阶。
* **残差 RL**。动作是「开环 trot 先验 + PPO 残差」的关节目标角：
  `target = home + gait_gain·gait.radians(t) + action_scale·action`。
  **先验只提供稳定的行走节律，不得成为楼梯的解** —— 爬楼必须由 RL 从奖励梯度里
  学出来（`experience.md` §21）。

课程固定九级（四级平地指令预热 → 3/6/9/12/15 cm），每级在上一级权重上继续训。

## 文档体系

| 文档 | 内容 | 对应代码 |
| --- | --- | --- |
| `README.md`（本文） | 代码结构 | — |
| `doc/<模块>.md` | 每个代码文件的**抽象实现方法** | 与代码文件一一对应 |
| `train.md` | 训练台账（每次训练的改动/结果/GIF） | — |
| `experience.md` | 调试记录（现象 → 根因 → 修法） | — |
| `agent.md` | **改代码之前先读**：同步义务、验证流程、硬约束 | — |

分工原则：**结构看 README、实现看 doc/、踩坑看 experience、规程看 agent**。
代码注释只保留接口契约与一行式设计要点，调试记录一律在 experience.md。

## 目录结构与依赖方向

```
upstair_core/
  README.md  agent.md  experience.md  doc/     文档
  config.yaml                                  ★ 全部超参（唯一入口）
  requirements.txt
  artifacts/                                   步态先验数据（JSON；先验是数据不是代码）
  models/unitree_go2/                          vendored Go2 MJCF（永不修改）
  robotdog/                                    核心包
    env.py         Go2StairsEnv：时序/观测/指令/成功/终止/渲染   → doc/env.md
    reward.py      ★ 全部奖励数学 + 势函数状态机                 → doc/reward.md
    config.py      config.yaml → EnvConfig + 阶段表              → doc/config.md
    curriculum.py  课程数据类与报表                              → doc/curriculum.md
    training.py    向量环境 / PPO / RunDir / 权重加载            → doc/training.md
    policy.py      checkpoint 的配置/结构/归一化恢复             → doc/policy.md
    gait.py        开环先验（数据化、速度参数化）                → doc/gait.md
    go2_model.py   MJCF 装配 / 场景编译 / 元信息                 → doc/go2_model.md
    tools/         工具层（不 import 上层）
      terrain.py   楼梯几何（box + 解析剖面双投影）              → doc/terrain.md
      paths.py     路径常量                                      → doc/paths.md
      console.py   控制台 UTF-8                                  → doc/console.md
    __init__.py / tools/__init__.py   再导出壳（无逻辑，不入 doc/）
  scripts/
    train.py       训练入口（逐级循环）                          → doc/train.md
    play.py        评估 / GIF                                    → doc/play.md
    teleop.py      外部控制器遥控                                → doc/teleop.md
    cmd_demo.py    指令跟踪量化                                  → doc/cmd_demo.md
    download_go2.py 拉取 Go2 模型                                → doc/download_go2.md
  runs/<名字>/     训练产物（gitignored）
  output/gifs/     GIF 交付件
```

依赖方向**单向无环**：`scripts → robotdog → robotdog/tools`。
工具层不知道 PPO、课程或训练循环的存在。

## 数据流

```
config.yaml ──config.py──► EnvConfig + StageSpec[]
                                │
              scripts/train.py  │ 逐级：make_env（深拷贝+下发量程+叠奖励覆盖）
                                ▼
        Go2StairsEnv ◄──► reward.py（compute_reward + 势函数状态机）
              │    ▲
              │    └── gait.py（先验目标角）＋ go2_model.py（MJCF/元信息）
              ▼
        SB3 PPO（training.py 组装，DummyVecEnv 单进程）
                                ▼
        runs/<name>/  checkpoint 三件套（zip + vecnormalize.pkl + _env.json）
                                ▼
        scripts/play.py（eval / GIF，口径从训练记录回灌）
```

## 快速上手

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt ^
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple ^
    --extra-index-url https://download.pytorch.org/whl/cpu

:: 训练（全部超参在 config.yaml）
.venv\Scripts\python.exe scripts\train.py --run-name <名字>

:: 评估 / GIF（权重自动配对 *_vecnormalize.pkl 与 <模型>_env.json 的地形）
.venv\Scripts\python.exe scripts\play.py --run <名字> --mode eval
.venv\Scripts\python.exe scripts\play.py --run <名字> --mode gif --batch <批次>

:: 遥控演示 / 指令跟踪量化
.venv\Scripts\python.exe scripts\teleop.py --run <名字> --flat --hold 0.45 0.3 0.0
.venv\Scripts\python.exe scripts\cmd_demo.py --run <名字> --phase inplace
```

环境要求与安装坑（venv/镜像/编码）见 `agent.md` §5。

## 改代码之后的验证

```bat
.venv\Scripts\python.exe -m py_compile robotdog/*.py robotdog/tools/*.py scripts/*.py
.venv\Scripts\python.exe -X utf8 .tmp\eq_check.py .tmp\eq_new.json   :: 行为等价回归
```

`eq_check.py` 跑 450 步 × 3 场景，与存档 JSON 逐项对比 `reward_terms` —— 碰过
env / reward / gait 的改动**必须**逐位为 0（有意的行为变化除外，那条差异要能解释）。
完整规程见 `agent.md` §6。

## 已知边界

* 能力带约 3–6 cm；9 cm 以上当前翻不过（`experience.md` §21）。
* 单次训练成功率有 ±18 pt 的种子方差，下结论前同 seed 集合至少各跑 3 次（§14）。
* 评估至少 40 回合，10 回合能给出 ±15 pt 的错觉（§5.7）。

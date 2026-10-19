# `robotdog/training.py` — 训练基础设施

对应代码：`robotdog/training.py`。训练侧的实测教训见 `experience.md` §4、§5.9、§7；
本文只讲抽象结构。

## 职责

PPO 训练的四块胶水：向量环境、PPO 构造、运行目录、权重加载。**不含**课程逻辑
（那是 `scripts/train.py` 的逐级循环）。

## 核心抽象

### 向量环境（`make_vec_env`）

```
DummyVecEnv(n_envs) → VecMonitor → VecNormalize
```

两个顺序决策都有理由：

- **单进程 `DummyVecEnv`** 是平台约束（沙箱禁 Windows 命名管道，`SubprocVecEnv`
  不可用；`mj_step` 不释放 GIL，线程池无收益）。
- **`VecMonitor` 放在 `VecNormalize` 里面**：这样 `info["episode"]["r"]` 保持
  未归一化的真实回报，报表读到原始数值。

### PPO 构造（`build_ppo`）

显式列出全部超参（默认值与 `config.yaml` 一致），拒绝 kwargs 透传 —— 显式强转
`int()/float()` 在挡配置文件的类型漂移。两处细节：`log_std_init` 的取值是「探索
噪声不能淹死站立姿态，但残差恒零就学不会爬楼」的折中；SB3 的 mini-batch 警告
在推理场景（1 个环境）无意义，被定向压掉。

**非对称 actor-critic（`actor_obs_index`）**：默认 `None`，就是 SB3 `MlpPolicy`
（actor 与 critic 吃同一份观测）。传非空下标列表时改用
`robotdog/asymmetric.py` 的 `AsymmetricActorCriticPolicy`：actor 网络的第一层只吃
这些下标对应的观测维，critic 吃全量（含 `ObsConfig.privileged_foot_contact` 的
4 维足端-地形接触）。实现只替换 `mlp_extractor`（SB3 的 actor/critic 都从它分叉），
不重写任何 policy 方法；`policy_kwargs` 里带上下标，作为 checkpoint 自描述的一部分。
**默认关时走原路径、策略类与观测逐位不变** —— 前向兼容的判据。

为什么值得：critic 的价值估计可以吃仿真器真值（接触、地形）而 actor 不能（盲走
契约），这是 RL 里标准的「训练期特权信息」；把它接进观测会让部署侧做不到。

### 权重加载（`load_policy`）

**权重离开训练上下文后必须带走两样东西**：观测归一化统计与当时的配置。
`load_policy` 做两件事，缺一不可：

1. 把 `*_vecnormalize.pkl` 的统计量**装回模型自带的 `VecNormalize`**
   （`restore_vecnormalize`）—— 只返回一份拷贝而不装回的话，`model.get_env()`
   路径等于喂未归一化观测，与裸 env 路径给出相反结论（`experience.md` §5.9）。
2. 网络结构从 checkpoint 里读（`policy_arch`，见 `doc/policy.md`），不用默认值建网。

**新增任何评估路径都必须走 `load_policy()`，不要自己搭。**

### 跨观测维 warm-start（对称 AC → 非对称 AC）

当新配置的观测比旧权重更宽（典型：`privileged_foot_contact` 打开，53 → 57 维）时，
权重不能直接 `set_parameters`。`robotdog/training.py` 提供三个配套符号：

- `policy_state_dict(model_path)`：从 SB3 `.zip` 读 `policy.pth`（不建环境）；
- `set_parameters_compat(...)`：只支持**「actor 维不变、critic 首层变宽」**这一种迁移 ——
  actor 原样搬（两边 actor 都只看 `actor_size` 维）；critic 首层**按 `ObsConfig.actor_index`
  把旧列放回、其余新增列补零**（`history_len > 1` 时特权块夹在每帧末尾、actor 维不是
  前缀，必须按下标映射）。补零让**初始 value 与旧权重逐位相同**（新特权输入的影响交给
  训练学）。只支持增宽：窄化、net_arch/actor 维变化、旧权重多余键一律 raise，不静默降级；
- `restore_vecnormalize_compat(...)`：旧 `_vecnormalize.pkl` 的观测统计按
  `ObsConfig.actor_index` 放回新向量、新增维 mean 0 / var 1。**不能走
  `VecNormalize.load`**（它会校验观测空间形状），要直接反 pickle 取 `RunningMeanStd`。
  其中 `mean`/`var`/`count` **三者必须一起恢复**：`count` 不恢复的话，第一次 reset
  （8 env）就以 ~100% 权重把旧统计整份替换 —— 假 warm-start（experience.md §23.34 同型），
  日志却仍打印「装回观测统计」。

`scripts/train.py` 在 `observation_dim(warm_start) != 环境观测维` 时自动走兼容路径，
同维时行为与旧实现完全一致；`--warm-norm off` 仍是「统计从零开始」的显式选择。

### 运行目录（`RunDir`）

`runs/<name>/` 的路径集合。**构造时不建目录**：调用方需要先检查「目录已存在」
来拒绝覆盖上次的 run，eager mkdir 会让这个检查永远为真。各写入点自己
`mkdir(parents=True)`。

## 踩坑指引

并行 run 内存上限、后台任务日志、VecNormalize pickle 的三个坑 —— 见
`experience.md` §5.4–§5.6 与 `agent.md` §5。

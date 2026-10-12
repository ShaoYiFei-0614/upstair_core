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

### 权重加载（`load_policy`）

**权重离开训练上下文后必须带走两样东西**：观测归一化统计与当时的配置。
`load_policy` 做两件事，缺一不可：

1. 把 `*_vecnormalize.pkl` 的统计量**装回模型自带的 `VecNormalize`**
   （`restore_vecnormalize`）—— 只返回一份拷贝而不装回的话，`model.get_env()`
   路径等于喂未归一化观测，与裸 env 路径给出相反结论（`experience.md` §5.9）。
2. 网络结构从 checkpoint 里读（`policy_arch`，见 `doc/policy.md`），不用默认值建网。

**新增任何评估路径都必须走 `load_policy()`，不要自己搭。**

### 运行目录（`RunDir`）

`runs/<name>/` 的路径集合。**构造时不建目录**：调用方需要先检查「目录已存在」
来拒绝覆盖上次的 run，eager mkdir 会让这个检查永远为真。各写入点自己
`mkdir(parents=True)`。

## 踩坑指引

并行 run 内存上限、后台任务日志、VecNormalize pickle 的三个坑 —— 见
`experience.md` §5.4–§5.6 与 `agent.md` §5。

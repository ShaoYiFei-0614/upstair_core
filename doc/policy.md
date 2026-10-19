# `robotdog/policy.py` — 训练产物的读取与恢复

对应代码：`robotdog/policy.py`。配置口径错误的完整案例见 `experience.md` §20.7；
本文只讲抽象结构。

## 职责

回答一个问题：**拿到一个 checkpoint 文件，如何恢复出「能正确运行它」的全部上下文**
—— 网络结构、观测归一化统计、环境配置、地形。

## 核心抽象

### 配置恢复的三层退路（`env_config_for_model`）

```
<模型>_env.json（训练时逐 checkpoint 写的完整配置）
  → 同目录 summary.json（整条课程的最后一份配置）
    → 按观测维度反推观测开关组合（最弱）
```

三层是**能力递减**的：

- `_env.json` 是唯一精确的来源 —— `gait_prior`、`action_scale` 这类**不进观测**
  的字段，任何按维度反推的方案在结构上都不可能恢复。用默认值回放一个在另一套
  配置下训出的权重，会得到**相反的评测结论**（`experience.md` §20.7）。
- `summary.json` 记的是整条课程最后一份配置，中途 checkpoint 用它不精确。
- 维度反推只能还原**可选**观测块组合（`base_lin_vel_body` / `command` 为必选
  恒在，`base_height` 被构造期禁止，均不在枚举内）
  （它不可构造）。反推成功时会打印提示 —— 调用方必须知道自己拿到的是推断值。
- **非对称 AC 训练出的权重无法按维度反推**：`privileged_foot_contact`（critic 的
  特权块）不在枚举里，含它的观测维（如 57）会被判为「无法还原」并中止。这类权重
  必须依赖同目录 `<模型>_env.json`（第一层退路）。

### ObsNorm：归一化统计的安全拷贝

SB3 的 `VecNormalize` 对象反 pickle 后**不能随便访问属性**（多读一个不存在的名字
就 `RecursionError`）。`ObsNorm.from_path` 立刻把 mean/var/epsilon/clip 拷进普通
对象、丢掉原对象，此后只用拷贝。

### checkpoint 自描述（`save_env_json`）

训练时每个 checkpoint 旁写一份 `<模型>_env.json`，`report` 段记下该权重**当时
的地形**（台阶高）。这是必须的：中间检查点在不同台阶上训练，只有最后一级是
15 cm —— 回放时拿「最后一级」当地形是错的。

### 网络结构读取（`policy_arch`）

从 SB3 `.zip` 的 `data` 条目读 `policy_kwargs.net_arch`（普通 JSON，不需要解
pickle）。重建模型必须用**训练时那一套**结构，否则 `set_parameters` 直接
size mismatch（`experience.md` §7.1）。`_zip_data` / `_deserialize` 是对 SB3 两种
存储格式（JSON / 旧版 pickle）的兼容读取，全部按「探测函数」写：任何失败返回
`None`，绝不抛异常。

## 踩坑指引

评测口径必须与训练一致；嵌套 dataclass 过 JSON 往返后要重建 —— 见
`experience.md` §13.5、§20.7。

`env_config_from_summary` 逐个重建嵌套 dataclass（`obs` / `reward` / `command` /
`gait` / **`gait_stair`**）：JSON 往返后它们都是普通 dict，漏一个就会在回放时
以 `AttributeError` 爆掉，或更糟 —— 静默用上默认值、得出相反的评测结论
（§23.32 的 `gait_stair` 漏重建）。**新增嵌套字段时必须同时加这里的一行。**

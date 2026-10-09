# `docs/` —— 模块设计文档

本目录为 `robotdog/` 下的**每一个 `.py`** 提供一份设计文档，**目录结构与
`robotdog/` 一一对应**：`robotdog/` 下共 12 个 `.py`（9 个在顶层，3 个在
`tools/`），因此本目录共 12 份模块文档 + 本索引。

```
docs/
  README.md                 本索引 + 文档规范（就是你正在读的这份）
  robotdog/
    __init__.md             包入口
    config.md               ★ config.yaml → EnvConfig / 课程阶段表
    env.md                  ★ 核心环境（盲走：无触觉、无测距）
    gait.md                 trot 步态先验
    go2_model.md            Go2 MJCF 组装与场景编译
    policy.md               训练产物的加载与观测归一化
    reward.md               ★ 全部奖励数学（RewardConfig + compute_reward）
    training.md             PPO 组装 / 向量环境 / 运行目录
    curriculum.md           六级课程（平地 + 3/6/9/12/15 cm）
    tools/
      __init__.md           工具子包入口
      console.md            UTF-8 控制台
      paths.md              路径常量
      terrain.md            参数化楼梯几何
```

入口脚本在 `scripts/` 下，不属于 `robotdog/` 包，因此没有对应文档；它们的用法
写在仓根 `README.md` §4 与 `train.md`：

* `scripts/train.py` —— 按课程训练，产物落进 `runs/<名字>/`。
* `scripts/play.py` —— 评估现有权重：`--mode gif`（渲染 10 个回合合成 GIF）
  或 `--mode eval`（跑 100 个回合统计成功率）。

## 分层规则

| 层 | 位置 | 允许依赖 | 说明 |
| --- | --- | --- | --- |
| 工具层 | `robotdog/tools/` | 只依赖标准库（`sys` / `pathlib` / `dataclasses`），以及**同层**模块 | **不 import `robotdog.env`**，不知道 PPO / 课程 / 训练循环 |
| 环境层 | `robotdog/env.py`、`go2_model.py`、`gait.py` | 工具层 | 一个 Gymnasium 环境及其模型装配 |
| 训练层 | `robotdog/training.py`、`curriculum.py`、`policy.py` | 环境层 + 工具层 | PPO 组装、课程调度、权重与归一化统计的加载 |

依赖是**单向**的：`训练层 → 环境层 → 工具层`。按这个规则，工具层的模块只有
一条导入路径 `robotdog.tools.<name>`，仓里不存在同名的顶层兼容模块：

```python
from robotdog.tools.terrain import StairSpec
from robotdog.tools import paths
```

是唯一写法，`robotdog.terrain` / `robotdog.paths` 之类的扁平路径**不可用**。
（早期版本用 `sys.modules[__name__] = _mod` 的替身模块保留过旧路径，已删除：
报错 traceback 与 IDE 静态分析都会指到那个没有逻辑的替身文件，代价大于收益。）

## 文档规范

每份模块文档固定 **10 节**，顺序不变（缺哪节就写明「无」并说明原因）：

1. **一句话定位** —— 属于哪一层、解决什么问题。
2. **职责边界** —— 做什么；**明确写出不做什么**。
3. **关键设计决策与理由** —— 重点节。每条决策要写清「为什么这么设计」，
   并**写出被否决的方案及否决原因**。
4. **公开 API** —— 逐个签名 + 语义 + 参数 + 返回值 + 会抛的异常。
   必须与源码逐字一致，不能凭印象写。
5. **内部结构与数据流** —— 关键方法的调用链（用 `A() → B() → C()` 表示）。
6. **关键常量与默认值** —— 逐项给出**真实数值**，不要写「见源码」。
7. **与其它模块的关系** —— 上游/下游依赖图，并解释**为什么方向是这样**。
8. **实测与经验教训** —— 引用仓根的 `experience.md` / `train.md` 的章节号
   （如 `experience.md` §2.1、`train.md` §3.3），**不要重复抄写**那些结论。
9. **已知限制与坑** —— 要能被 grep 到具体符号（函数名、字段名、报错文本）。
10. **示例用法** —— 最小可运行片段。

## 相关文档

* 仓根 `README.md` —— 环境配置与代码实现总览。
* 仓根 `experience.md` —— 关键错误与修正（踩过的坑都在这里）。
* 仓根 `train.md` —— 训练组织与实测复现结论。

## 维护要求

改代码时同步改文档。文档里的 API 签名与默认值必须能在源码里 grep 到；
如果某个符号在源码中已经不存在，文档里就必须删掉它，而不是保留「历史说明」。
历史结论请写进 `experience.md`。

## 本轮重构删掉了哪些文档

`render.md`、`evaluate.md`、`tools/touch.md` 随源码一起删除：`robotdog/render.py`
与 `robotdog/evaluate.py` 不再存在（GIF 能力内联进 `scripts/play.py`，评估就是
`play.py --mode eval`），触觉/测距整套逻辑已删除（本仓的目标是**盲走楼梯**）。

`tools/png.md` 随 `robotdog/tools/png.py` 一起删除：那个 130 行的纯 zlib PNG
容器**连自己内部的 `write_png`/`read_png`/`downscale`/`hstack` 都没有调用者**，
三个历史消费者（`render.py`、`render_checkpoint.py`、`eval.py`）早已删除，
GIF 与拼图改由 `scripts/play.py` 直接用 Pillow 完成。

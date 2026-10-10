# 仓结构与文档纪律

> 对应 `SKILL.md` 阶段 7 / 阶段 10、硬契约 11。素材取自 upstair_core 的目录分层、四份根文档与模块
> 文档规范（`docs/README.md` + `agent.md`）。

## 1. 最小可用目录结构

```
<repo>/
  config.yaml                  # 唯一超参入口（六段：run / curriculum / env / reward / ppo / play）
  README.md  train.md  experience.md  agent.md     # 四份根文档，就这四份
  <pkg>/                       # 主包（只保留一个，不要建平行的第二层）
    __init__.py
    config.py                  # 配置 → dataclass（load_config / config_to_env / config_to_stages）
    <robot>_model.py           # vendored MJCF 组装 + 元信息提取 + 编译
    gait.py                    # 开环步态先验（参数化 + JSON 存取）
    env.py                     # Gymnasium 环境（观测 / 终止 / 成功判据）
    reward.py                  # 无状态纯函数奖励
    curriculum.py              # 课程调度（阶段边界下发指令范围与奖励覆盖）
    training.py                # PPO 组装 + 统一加载路径
    policy.py                  # 权重 / 归一化统计加载
    tools/                     # 工具层：只依赖标准库 + 同层
      terrain.py  paths.py  console.py  ...
  scripts/                     # 入口脚本：只依赖主包
    train.py  play.py  teleop.py  download_<robot>.py
  models/<robot>/              # vendored 上游模型，**永不修改**
  artifacts/                   # 不可重建的产物（搜出来的步态先验等）
  docs/                        # 与主包一一对应的模块文档 + README.md 索引
  runs/  logs/  output/  .generated/  .tmp/    # 产物与临时（前四个 gitignored，output/ 是交付物）
```

## 2. 单向分层（硬契约）

| 层 | 内容 | 允许依赖 |
| --- | --- | --- |
| 入口层 `scripts/` | 训练 / 评估 / 交互演示 / 下载模型 | 主包顶层 |
| 训练层 | `training.py` `curriculum.py` `policy.py` | 环境层 + 工具层 |
| 环境层 | `env.py` `reward.py` `<robot>_model.py` `gait.py` | 工具层 |
| 工具层 `tools/` | 地形 / 路径 / 控制台 | **标准库 + 同层** |

**工具层的硬规则**：不 `import` 环境层，不知道 PPO、课程、训练循环的存在。理由：离线工具必须能
**单独导入**——只算一次地形高度、只读一次路径，不该为此拉起 MuJoCo 与 PyTorch。

两条真实教训：

* 早期为了让扁平导入（`from <pkg>.terrain import ...`）也能用，用
  `sys.modules[__name__] = _mod` 造替身模块。**已删**：traceback 与 IDE 静态分析都会指到一个没有
  逻辑的替身文件，代价大于收益。统一用 `<pkg>.tools.terrain` 这种显式路径。
* 环境层里唯一「无状态」的模块是 `reward.py`：它不 import 环境，只被 `env.py` 导入，环境再把
  `RewardConfig` 重导出给配置层与训练层。**不要**反过来让 `reward.py` import 环境（既成环，又让纯
  函数难测）。

## 3. 超参单一入口

* `config.yaml` 是唯一入口；入口脚本 CLI 只留 `--run-name` / `--config`（评估脚本留 `--run` / `--mode`）。
  **不把超参搬回 CLI**——命令行参数会绕过配置层，让「这次跑了什么」无法从产物里还原。
* 读取时**未知键直接报错并列出可用键**。打错一个字母必须立刻可见，而不是被静默忽略。
* 默认值写在 dataclass 里，yaml 只写**覆盖项**（被注释掉的行表示「与默认相同」）。
* 分层覆盖：任务常量与环境常量放 `env:`，奖励基线放 `reward:`，**按课程级变化的**东西放
  `curriculum:`（每级学习率、指令范围、奖励覆盖、平地段专用成功距离）。这样「基线 + 课程增量」一眼
  能分开，回退某一级不用改基线。
* **每个 checkpoint 必须带自己的环境快照**（`<模型>_env.json`：地形 + 完整配置）。否则评估时无法
  知道这个权重是在什么地形/什么指令范围下训出来的——参考实现里有三个历史权重因为缺这个文件而
  只能靠猜。

## 4. 四份根文档的职责

| 文档 | 写什么 | 不写什么 |
| --- | --- | --- |
| `README.md` | 项目是什么、怎么装、怎么跑、能力带与局限 | 逐次实验流水账 |
| `train.md` | 训练/课程的操作细节与判读方法 | 设计哲学 |
| `experience.md` | **历史账本**：每次实测的现象、判据、数字、否决的方案 | 会过期的「当前状态」 |
| `agent.md` | 给维护者/agent 的硬约束、流程、红线、验证清单 | 教程 |

规则：**根文档只有这四份**；新的实测结论进 `experience.md`（带日期与判据），不要新建第五份。
`experience.md` 是只增不改的账本，所以它允许出现「旧架构名」（例如已经删掉的类与命令行开关）——
读到它们要当成历史记录，**不要照着实现**。

## 5. 模块文档纪律

* `docs/<模块>.md` 与 `<模块>.py` **严格一一对应**（含子包）；`docs/README.md` 是索引 + 分层规则表。
  新增/删除模块要同步索引，否则 5 号门报 `MISSING` / `NOT-IN-INDEX`。
* 每份固定 **10 节、顺序不变**（模板见 `assets/module-doc.template.md`）。某一节没有内容就写明
  「无」并说明原因，不要删节。
* §3 必须写**被否决的方案及原因**；§6 必须写**真实数值**而不是「见源码」；§8 引用经验文档的章节号
  而不要重复抄写。
* 源码里已不存在的符号，文档里**删掉**而不是留「历史说明」；历史写 `experience.md`。
* 引用代码优先写 `路径/文件.py` + **符号名**，行号只在特指某一行时写。

## 6. 代码约定（换仓照带）

* 注释与 docstring 用中文 + 真中文标点。唯一允许的英文：装饰分隔线、`# noqa: ...`、
  形状注释（`# (12,)`）、库名横幅。
* 公开 API 用 `__all__` 显式列出；不用的 import 直接删。
* 模块依赖单向；**禁止新建与主包平行的代码层**。
* 行尾 CRLF（`core.autocrlf=true` 的仓）：整篇 rewrite 写出的是裸 LF，提交前跑一次
  `assets/gates/fix_eol.py`（见 `08-verification.md`）。
* 解释器固定用仓内虚拟环境（`.venv\Scripts\python.exe`），不要依赖全局 Python。

## 7. 校验脚本放哪

放在**仓根下一级目录**（`.tmp/` 或 `tools/dev/`），因为它们用 `parents[1]` 定位仓根。
六支门 + 两个幂等辅助脚本见 `08-verification.md`，原件在 `assets/gates/`。

## 8. 这套结构为什么值钱

1. **一切可被命令验证**：改完跑六支门，人交接给 agent 也一样。
2. **超参单入口 + 每 checkpoint 记配置** ⇒ 任何历史权重都能还原评估条件，不用翻聊天记录。
3. **文档与代码一一对应** ⇒ 「改一处忘另一处」变成一条报错，而不是几个月后的一次困惑。

# `scripts/train.py` — 训练入口

对应代码：`scripts/train.py`。训练结论与调参教训见 `experience.md` §4、§17、§21；
本文只讲抽象结构。

## 职责

按 `config.yaml` 的课程串行训练。CLI 只认 `--run-name` 与 `--config`，其余全部
来自配置文件。

## 核心抽象

### 逐级循环

```
读配置 → 生成阶段表 → 建 RunDir（已存在则拒绝启动）
→ for stage: make_env(stage) → train_stage(stage) → 记录结果 → 末级另存 model_final
```

每一级一个**全新的向量环境**（地形烘焙在编译后的 MJCF 里，换地形 = 重编译，
不能在 rollout 中间做）。

### 每级的环境下发（`Trainer.make_env`）

三条纪律，每条都对应一类静默出错的方式：

1. **每级深拷贝一份 `EnvConfig`**：下发给子环境的设置全是就地改写
   （`set_commands`、`set_reward_overrides`），直接传共享 cfg 会让设置跨级累积、
   顺序敏感。
2. **奖励覆盖「从基线叠」**：先显式抄回 `config.yaml` 的基线，再叠本级的键 ——
   否则上一级覆盖过、这一级没覆盖的键会残留上一级的值。
3. **指令量程无条件下发**：量程为 0 是「这一级不练这个方向」的有意义设置，
   `if stage.command_vy_max:` 之类的守卫会让该级继承上一级的范围。

### 调子环境方法一律走关键字（`kw_call`）

`venv.env_method` 的位置实参按形参顺序绑定 —— 「参数名对不上」要变成
`TypeError` 而不是静默传错值。

### 产物三件套

每级结束存 `<阶段>_final.zip` + `<阶段>_final_vecnormalize.pkl` +
`<阶段>_final_env.json`；中间检查点（`CheckpointCallback`）**同样带 `_env.json`**
—— 它是唯一能把「这个权重当时跑在什么配置下」带出来的东西，且从子环境读回的是
**这一级实际生效**的配置（已叠好量程与覆盖），不是基线。训练另写
`plan.json`：每一级实际下发的量程与奖励覆盖的实施记录。

## 踩坑指引

用另一套配置回放权重会得到相反的评测结论 —— 见 `experience.md` §20.7。

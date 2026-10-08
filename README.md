# Go2 上下楼梯强化学习 · 最小核心仓

这是从完整项目（`D:\项目\RobortDog\upstair`）里抽出来的**最小可复现核心**。它只包含
「从零训练出一只会上楼梯的 Unitree Go2 策略，并把 course22 / course31 / course49
三条记录在案的策略复现出来」所必需的代码、模型和步态先验。

完整仓里的 10 套回归（`scripts/verify.py`）、6 个诊断工具（`scripts/diagnose.py`）、
全部 course 历史（`COURSES.md`）和归档目录 **都不在这里**。

| 这个仓有什么 | 这个仓没有什么 |
| --- | --- |
| `robotdog\`（13 个文件，核心库：MuJoCo 环境 / 地形 / 步态 / 触觉 / 训练 / 评估 / 渲染） | 10 套回归验证、6 个诊断工具 |
| `scripts\`（4 个入口：训练 / 评估 / 渲染 / 下载模型） | 任何 `runs\` 权重（要自己训，见 §4） |
| `tools\`（2 个普查脚本：6 档高度 × 40 回合的配对验收） | `archive\`（177 个一次性探针脚本） |
| `models\unitree_go2\`（MJCF + 网格）、`artifacts\trot_stairs.json`（trot 先验） | course1~course49 的逐步历史（在原仓 `COURSES.md`） |

---

## 1. 能力一句话

在一段 8 级、踏面 50 cm 的楼梯上（`start_x = 1.50`，顶层平台 0.88 m）：

| 高度 | 8.0 cm | 9.0 cm | 9.5 cm | 10.0 cm | 10.5 cm | 11.0 cm | 11.5 cm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 成功率 | 100% | 100% | 90~97% | 87~92% | 60~70% | 10~15% | ~0% |

11.0 cm 以上是**墙**：46 个课程 run、两轮奖励/观测/终止改造之后仍然只有 10~15%，
`course31`（最强者）在 11.5 cm 上是 0%。失败点集中在**第二级立面**。

## 2. 环境

* **Python 3.10+**（实测 3.13.0）；CPU 就够 —— 8 个环境串行跑，实测约 400 向量步/秒
* Windows 10/11 实测；Linux 未测（代码里没有平台相关分支，只是没跑过）
* 依赖（版本就是训练这三条策略时用的版本）：

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt --index-url https://pypi.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cpu
```

`torch==2.14.1+cpu` 的 `+cpu` 后缀来自 PyTorch 官方 CPU 源，所以命令里必须带
`--extra-index-url https://download.pytorch.org/whl/cpu`（清华镜像是主源）。

`models\unitree_go2\`（MJCF + obj 网格，28.9 MiB）和 `artifacts\trot_stairs.json`
（trot 步态先验，11 KB，**唯一不可重建的 artifact**）都已经在仓里，不需要额外下载；
想重新拉一份 Go2 模型可以跑 `scripts\download_go2.py`。

## 3. 目录

```
robotdog\
    paths.py       所有路径常量（工作区 = 本仓根目录）
    terrain.py     楼梯地形（解析式高度场 height_at(x) + MuJoCo 几何体）
    go2_model.py   组装 Go2 MJCF（字符串手术 + 场景编译）
    gait.py        trot 步态先验（GaitParams）
    touch.py       触觉/前视探测（足底法向力、滑移、立面擦碰、前方射线扇）
    env.py         Gymnasium 环境（观测 660 维、奖励、终止、域随机化）★核心
    curriculum.py  课程（平地 → 3~8 → 8~11 cm，含混合高度）
    training.py    PPO 组装 / 运行目录 / VecNormalize / 断点续训
    evaluate.py    评估矩阵 + ONNX / TorchScript 导出
    render.py      GIF 写出（纯 Python + Pillow，不需要 ffmpeg）
    png.py         PNG 写出（zlib）
    console.py     UTF-8 控制台
scripts\
    train.py               训练入口（所有开关都在这里）
    eval.py                评估 / 导出入口
    render_checkpoint.py   渲染 GIF
    download_go2.py        （可选）重新下载 Go2 MJCF
tools\
    ab_pair.py   6 档高度 × 40 回合的配对普查（验收用）
    ab_merge.py  合并多份普查缓存并出表
models\unitree_go2\   Go2 MJCF + 网格
artifacts\trot_stairs.json   trot 先验
```

训练/评估产物都落在本仓根目录下（`runs\`、`logs\`、`output\gifs\`、`.tmp\`），
目录由 `robotdog\paths.py` 自己建。

## 4. 三条复现路线

三条命令都从**本仓根目录**执行。总耗时（i7-12700H，8 环境，纯 CPU）约 1 小时。

### 4.1 course22 —— 从零训练（约 22 分钟，4.45M 环境步）

```bat
.venv\Scripts\python.exe scripts\train.py --run-name course22 --stage medium --stage-top-cm 11.0 --sub-steps 1 --mixed-heights 8.0,11.0 --n-envs 8 --net-arch 512 256 --learning-rate 1.5e-4 --target-kl 0.03 --log-std-init -2.0 --gait-prior --gait-file artifacts\trot_stairs.json --obs-base-lin-vel-body --obs-base-height --obs-foot-height --obs-touch --obs-stair-probe --obs-drift --terminate-contact trunk --log-interval 5 --save-every 50000 --min-stage-frac 1.0 --det-episodes 10 --stage-timesteps 400000 --cfg episode_length_s=20.0 --cfg reward.heading=-30.0 --cfg reward.heading_max=0.35 --cfg terminate_contact_steps=10 --cfg terminate_height_ref=support
```

这条命令就是原项目里 course22 的**完整配方**（逐字来自归档脚本
`archive/scripts/variant_train.py` 的 `COMMON`，注释写着「一个字符都不改」）。
`runs\course22\summary.json` 里自己的 `reproduce` 字段是**旧版有损回显**
（只回显了 5 个参数，丢了 `--obs-*`/`--cfg`/`--mixed-heights`），不要拿它当配方。

期望：`medium` 阶段（8~11 cm 混合高度）在 552k 向量步 / 1339 s 后**毕业**
（滚动成功率 66%，毕业线 65%）。产物：

```
runs\course22\model_final.zip                  ← 最终权重（11.4 MB，942,873 参数）
runs\course22\model_final_vecnormalize.pkl     ← 观测归一化（必须一起用！）
runs\course22\checkpoints\medium_final.zip     ← 阶段快照（= 交付策略的权重）
runs\course22\artifacts\best_medium.zip        ← 阶段内最优
runs\course22\summary.json                     ← 全部超参 + 结果
```

### 4.2 course31 —— 从 course22 续训 40 万步（约 20 分钟）

```bat
.venv\Scripts\python.exe scripts\train.py --run-name course31 --resume runs\course22\checkpoints\medium_final.zip --stage medium --stage-top-cm 11.0 --sub-steps 1 --mixed-heights 8.0,11.0 --n-envs 8 --net-arch 512 256 --learning-rate 1.5e-4 --target-kl 0.03 --log-std-init -2.0 --gait-prior --gait-file artifacts\trot_stairs.json --obs-base-lin-vel-body --obs-base-height --obs-foot-height --obs-touch --obs-stair-probe --obs-drift --terminate-contact trunk --log-interval 5 --save-every 50000 --min-stage-frac 1.0 --det-episodes 10 --stage-timesteps 400000 --cfg episode_length_s=20.0 --cfg reward.heading=-30.0 --cfg reward.heading_max=0.35 --cfg terminate_contact_steps=10 --cfg terminate_height_ref=support --cfg reward.stair_progress_foot_ref=true --cfg reward.stair_progress_min_loaded_feet=2
```

相对 course22 只多了两行 `--cfg`（台阶进度奖励改成「按已加载的足数判参考面」）。
两项都**不改变默认行为之外的东西**：默认 `False` 时整段与历史逐位相同。

⚠ **course31 没有 `model_best.zip`。** 续训后 `best_stage` 是 `null`（没有重新毕业
任何阶段），所以 `model_best` 从来没有被写过 —— 尽管 `summary.json` 的
`artifacts.best_model` 里还写着这个路径。course31 的交付权重是
`runs\course31\artifacts\best_medium.zip`（阶段内最优）。

### 4.3 course49 —— 从 course22 续训 40 万步（约 16 分钟）

```bat
.venv\Scripts\python.exe scripts\train.py --run-name course49 --resume runs\course22\checkpoints\medium_final.zip --stage medium --stage-top-cm 11.0 --sub-steps 1 --mixed-heights 8.0,11.0 --n-envs 8 --net-arch 512 256 --learning-rate 1.5e-4 --target-kl 0.03 --log-std-init -2.0 --gait-prior --gait-file artifacts\trot_stairs.json --obs-base-lin-vel-body --obs-base-height --obs-foot-height --obs-touch --obs-stair-probe --obs-drift --terminate-contact trunk --log-interval 5 --save-every 50000 --min-stage-frac 1.0 --det-episodes 10 --stage-timesteps 400000 --cfg episode_length_s=20.0 --cfg reward.heading=-30.0 --cfg reward.heading_max=0.35 --cfg terminate_contact_steps=10 --cfg terminate_height_ref=support --cfg reward.stair_progress_foot_ref=true --cfg reward.stair_progress_min_loaded_feet=2 --cfg contact_debounce_adaptive=true --cfg contact_debounce_hi_steps=1
```

= course31 + 两行「高台阶上躯干接触去抖收紧到 1 步」的开关。course31 与 course49
**都从 course22 续训，不是前后相承**。course49 的交付权重是
`runs\course49\model_final.zip`。

### 4.4 只想快速看它会不会走

```bat
rem 10 回合、单档 10.0 cm，约 1 分钟
.venv\Scripts\python.exe scripts\eval.py --run course22 --model runs\course22\checkpoints\medium_final.zip --only trained --episodes 10
```

## 5. 评估 / 导出 / 渲染

```bat
rem 评估（trained = 训练时见过的 4 个地形；unseen = 9 个没见过的）
.venv\Scripts\python.exe scripts\eval.py --run course22 --model runs\course22\checkpoints\medium_final.zip --only trained --episodes 20
.venv\Scripts\python.exe scripts\eval.py --run course22 --model runs\course22\checkpoints\medium_final.zip --only unseen --episodes 20

rem 导出 ONNX + TorchScript（归一化已经烘焙进计算图，部署不需要再带 pkl）
.venv\Scripts\python.exe scripts\eval.py --run course31 --model runs\course31\artifacts\best_medium.zip --only trained --episodes 6 --export --export-dir runs\course31\export --export-name policy_course31

rem 渲染 GIF（不带 --no-strip/--no-report 时会额外写 output\stills 与 output\reports）
.venv\Scripts\python.exe scripts\render_checkpoint.py --run course22 --checkpoint runs\course22\checkpoints\medium_final.zip --step-height-cm 10.5 --n-up 8 --episodes 4 --name-tag core --no-strip --no-report
```

`--model` 指到哪个 zip，就自动配同目录的 `<同名>_vecnormalize.pkl`；
渲染完整参数见 `scripts\render_checkpoint.py --help`。

**交付策略的权重来源**（原项目导出时的口径）：

| 导出名 | 权重 | 6 档均值 |
| --- | --- | --- |
| `policy_course22` | `runs\course22\checkpoints\medium_final.zip` | 75.0% |
| `policy_course31` | `runs\course31\artifacts\best_medium.zip` | 77.5% |
| `policy_course49` | `runs\course49\model_final.zip` | 76.7% |

## 6. 验收：40 回合 × 6 档的配对普查

单次 10 回合的确定性评估抖动可达 ±15 pt（原项目踩过：10 回合看是 90%，
40 回合只有 65%）。所以验收统一用 `tools\ab_pair.py`：**40 回合 × 6 档高度、
所有策略共用同一批种子（`SEED + k`）**，所以「同一回合基线成、变体败」可以直接配对计数。

```bat
rem 三条标签一次跑完：3 × 6 × 40 = 720 回合，约 45 分钟 CPU
.venv\Scripts\python.exe tools\ab_pair.py "c22 final=runs/course22/checkpoints/medium_final.zip=runs/course22" "c31 best=runs/course31/artifacts/best_medium.zip=runs/course31" "c49 final=runs/course49/model_final.zip=runs/course49" --episodes 40 --fresh

rem 先冒烟（单档 10 回合），确认管线通了再放开
.venv\Scripts\python.exe tools\ab_pair.py "c22 final=runs/course22/checkpoints/medium_final.zip=runs/course22" --heights 10.5 --episodes 10
```

**期望结果**（原项目实测，确定性策略，6 档 = 8.0 / 9.0 / 9.5 / 10.0 / 10.5 / 11.0 cm）：

| 标签 | 8.0 | 9.0 | 9.5 | 10.0 | 10.5 | 11.0 | 均值 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `c22 final` | 100.0% | 100.0% | 90.0% | 90.0% | 60.0% | 10.0% | 75.0% |
| `c31 best` | 100.0% | 100.0% | 97.5% | 87.5% | 65.0% | 15.0% | 77.5% |
| `c49 final` | 100.0% | 95.0% | 95.0% | 92.5% | 70.0% | 7.5% | 76.7% |

（`c49 best` = `runs\course49\artifacts\best_medium.zip` 是 90/95/100/87.5/70/15 = 76.2%；
三条策略之间的差异**在统计上互不显著**，选谁看你要上限还是看 10.0~10.5 cm 的稳定性。）

⚠ **只承诺落进同一能力带，不承诺逐位复现**：训练是随机过程，重跑出来的数会变，
常态是 ±5~10 pt 的档位波动。同一条命令重训出来的策略如果均值掉到 60% 附近，
先怀疑 `--stage-timesteps` 没跑够 / 中途 Ctrl-C 过，而不是代码坏了。

## 7. 这头狗是怎么训出来的（要点）

* **动作空间**：12 个关节目标角 = trot 步态先验 + PPO 残差 × `action_scale=0.30`，
  底层是 50 Hz 的 PD 控制（`timestep=0.002`、`decimation=10`、`kp=[30,30,45]`、
  `kd=[0.6,0.6,0.9]`）。策略不需要自己学步态，只需要学「怎么改」。
* **观测 660 维** = 42 × 10 帧堆叠（关节角/角速度、机身 rpy、角速度、上一帧动作）
  + 机身高度、四足高度、触觉（足底法向力/滑移/立面擦碰）、前方射线扇（探测下一级）、
  漂移量。**没有接触传感器以外的真值**：楼梯高度是策略自己「看」出来的。
* **课程**：`--stage medium --stage-top-cm 11.0` 把 medium 段设成 8~11 cm，
  `--mixed-heights 8.0,11.0` 让每个环境回合随机取一个高度 —— 这一步（course22）
  比先前的「同一段固定高度」明显更稳。
* **三个环境侧的关键修补**（都是「几十点」级别的改动，不是调参）：
  1. 踏步立面与顶面的共面接缝重叠（`robotdog/terrain.py` 的 `CREST_SEAM_OVERLAP`）
     —— 修之前 8 cm 台阶只有 1/10，修之后 8/10；9.0 cm 从 0% 到 65%；
  2. 躯干接触去抖 `terminate_contact_steps=10`（0.2 s）—— 爬高台阶时胸口
     蹭一下立面就被判死，会把回合从 ~290 步砍到 ~190 步；
  3. 摔倒参照面 `terminate_height_ref=support` —— 老判据用「机身 x 处的地形高度」，
     那是个**阶跃函数**，机身跨过立面瞬间参考面跳一整个台阶高 ⇒ 幽灵摔倒。
* **奖励**：前进速度、朝向（`heading=-30.0`，上限 0.35）、抬足高度、直行、
  台阶进度、躯干/腿碰撞、存活/终止/成功。全部权重和它们的实测理由都写在
  `robotdog/env.py` 的 `RewardConfig` docstring 里（每条都记着「哪次实测得出这个默认值」）。

## 8. 注意事项（原项目踩过的坑）

1. **`--resume` 会静默忽略优化器/网络开关**：`--net-arch`、`--learning-rate`、
   `--target-kl`、`--n-epochs` 只在**从零训练**那个分支里生效，续训分支只加载权重 +
   VecNormalize。所以要改这些，只能在从零那一次写进去。§4.2/§4.3 只加 `--cfg` 正是这个原因。
2. **不要改踏面宽度**：`--tread` 保持 0.50 m。实测把它换成 0.40/0.60 会让成功率从
   34% 掉到 10%（0.65 m 直接 0%）。课程表、奖励尺度、终止判据都是围绕 50 cm 调的。
3. **评估至少 40 回合**：10 回合的确定性评估能给出 +15 pt 的错觉（见 §6）。
4. **`summary.json` 的 `reproduce` 字段分两代**：新版本（course31 之后）原样回显
   真实命令行，可信；旧版本（course22）只有 5 个参数，是有损的。以 §4.1 为准。
5. **`model_best.zip` 不是总会存在**：只有「毕业过至少一个阶段」的 run 才会写它
   （course22 有，course31/49 没有）。想拿「阶段内最优」用 `artifacts\best_<stage>.zip`。
6. **观测归一化必须跟着权重走**：`*_vecnormalize.pkl` 和 `*.zip` 是一对，`--model`
   会自动找同名的 pkl；手工加载时别只加载权重。
7. **Windows 上 `>` 重定向会写成 UTF-16**（`python ... > log.txt`），事后 `read`
   会判成二进制。要留日志就用 PowerShell 的 `Out-File -Encoding utf8` 或不重定向。
8. **随机性**：见 §6 末尾，能力带内波动是正常的。

## 9. 和完整仓的关系

完整项目（`D:\项目\RobortDog\upstair`）里还有：

* `scripts\verify.py --all`：10 套回归（环境/地形/观测/奖励/导出/步态…），全绿；
* `scripts\diagnose.py <reward|trace|terrain|policy|prior|curve>`：6 个诊断工具；
* `COURSES.md`：course1 ~ course49 全部差异、血统、复现命令与历史结论；
* `output\gifs\`：75 份渲染好的 GIF（本仓的 `output\gifs\` 是空的）。

本仓的代码与完整仓**逐字节相同**（就是从那里面复制出来的），所以本仓能跑的东西
完整仓都能跑；反过来不成立（完整仓多了验证/诊断/归档）。

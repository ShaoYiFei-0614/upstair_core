# `robotdog/tools/terrain.py` — 参数化楼梯几何

对应代码：`robotdog/tools/terrain.py`。接缝夹脚与渲染消失两个坑的完整复盘见
`experience.md` §1.1、§1.2；本文只讲抽象结构。

## 职责

楼梯的**单一真值来源**：同一份 `StairSpec` 同时生成 MuJoCo 的 box 几何与解析高度
剖面 `height_at(x)`，保证物理、奖励、出生点、评估四者对「地形有多高」的理解
逐位一致。

## 核心抽象

### StairSpec：一份参数，两种投影

`step_blocks()` 生成每个实心 box 的名字/中心/半尺寸（上行段、可选 crest、
可选下行段或顶层平台）；`height_at(x)` 给出同一几何的解析高度。布局沿 +x：
平地 → n_up 级上升立面 → 平台（或 crest → n_down 级下降 → 地面）。

`__post_init__` 拒绝三类「各自以更晚、更难懂的方式失败」的参数组合
（零高度台阶 → MuJoCo 编译期报错、零踏面 → 除零、下降段无 landing → 重叠）。

### 两个必须知道的常量

- **`TERRAIN_GEOM_GROUP = 1`**：地形（含地面）全部放这一组。MuJoCo 渲染默认隐藏
  group 3–5，选错组会得到「物理在爬楼梯、画面里狗在爬空气」的世界。
- **`CREST_SEAM_OVERLAP = 1.0`**：crest 向最后一级台阶内部**回伸一个踏面**。
  共面相邻 box 的接缝是一条零高度的缝，脚球落上去会被两个 box 水平对推、z 向
  无支撑而夹死 —— 回伸让共面对消失，实心块并集不变。

### 便捷构造器的分工

- `flat_spec()` — 平地。**必须**同时把 `n_up`、`n_down`、`step_height` 归零：
  前者让几何不生成 box，后者让观测里的 `step_height` 读数真的是 0。
- `demo_stairs_spec()` — 台阶段（默认顶层平台 1.2 m）。

`build_terrain_xml` 把 spec 变成注入 `<worldbody>` 的 MJCF 片段（平地返回空串）。

## 踩坑指引

零高度台阶不能编译、平地必须走 `flat_spec` —— 见 `experience.md` §1.3 与
`agent.md` §5。

# `robotdog/tools/paths.py` —— 项目文件系统位置

## 1. 一句话定位

**工具层**（`robotdog/tools/`）里的**纯数据模块**：把整个项目的目录与关键文件位置定义成一组 `pathlib.Path` 常量，并附两个 `mkdir` 辅助函数。它解决的问题是「同一个目录被十个模块用十种写法拼出来」——所有派生路径只有一个来源，改目录结构只改这一个文件。

## 2. 职责边界

**做**：

* 定义唯一的工作区根 `WORKSPACE`，其余全部由它派生（模型、脚本、实验产物、交付物、生成物）。
* 提供两个目录创建辅助：`ensure_dir(path)`（`mkdir -p` 并返回路径）、`ensure_output_dirs()`（只建交付物部分）。
* 用注释把「哪些是交付物、哪些是按需创建的工作目录」写死在常量旁边。

**不做**：

* **不做 glob / 不搜文件、不校验文件是否存在、不读文件内容**。
* **不 import 任何项目模块**（连 `robotdog.tools.*` 都不），只依赖 stdlib `pathlib`。
* **不在 import 时创建任何目录**（模块顶层没有任何 `mkdir` 调用）。
* **不清理 / 不删除 / 不归档**目录。
* **不处理 CLI 覆盖**（例如「用 `--root` 指定另外的输出目录」在调用方实现）。
* **不 import `robotdog.env`**，不知道 PPO / 课程 / 训练循环。

## 3. 关键设计决策与理由

### 3.1 `WORKSPACE` 用 `Path(__file__).resolve().parents[2]`，并配一条显式注释

```python
# .../upstair_core  -- 本文件在 <workspace>/robotdog/tools/ 下，所以往上走三级。
# （它曾经是 robotdog/paths.py 走两级；搬进 tools 子包后所有派生路径都指错，
# 直到冒烟测试才被抓到，因此保留这条注释。）
WORKSPACE: Path = Path(__file__).resolve().parents[2]
```

这是本文件**最重要**的一条决策，理由有三层：

1. **`parents[2]` 是位置相关的**。`__file__` = `<ws>/robotdog/tools/paths.py`，`resolve()` 消掉符号链接与 `..`，于是 `parents[0]=<ws>/robotdog/tools`、`parents[1]=<ws>/robotdog`、`parents[2]=<ws>`。这个文件**再挪一次目录就会全仓路径错位**，所以必须把「为什么是 2」写在紧邻的位置。
2. **失败模式是静默的**。路径拼错不会立刻报错 —— 所有 `Path` 对象照样构造成功，只有在真正打开文件、并且**恰好**有对应文件时才会炸。历史上正是这样：本文件曾经住在 `robotdog/paths.py`（走上溯两级），被移进 `tools/` 子包后，`parents[1]` 变成了 `<ws>/robotdog`，于是 `MODELS_DIR` 变成 `<ws>/robotdog/models`、`GO2_XML` 变成 `<ws>/robotdog/models/unitree_go2/go2.xml`，直到冒烟测试打开模型时抛 `FileNotFoundError: ...\robotdog\models\unitree_go2\go2.xml` 才暴露。**报错串里多出来的那个 `robotdog\` 前缀就是「少上溯一级」的指纹**，可以直接用来定位同类错误。
3. **「显式注释」本身是决策**：注释写明历史事故（用了「直到冒烟测试才被抓到」这种因果句），使得后来者想再搬这个文件时先读到代价。

**被否决的方案**：

* 用 `Path(__file__).resolve().parents[1] / "..."` 逐级拼 —— 否决，没有减少位置相关性，反而让每一条常量都各自上溯。
* 用环境变量 `ROBOTDOG_ROOT` / 命令行 `--root` 作为唯一来源 —— 否决，因为绝大多数调用方（`training.py`、`go2_model.py`）**只想要一个默认值**，强制传参会把「能不能跑」变成一个额外的失败点；本仓的默认值就是仓库自身，不需要可配置。
* 用 `importlib.resources` / 打包元数据定位 —— 否决，本仓是「源代码目录 = 工作目录」的形态（`models/`、`scripts/` 与包同级），资源 API 面向的是安装后的包数据，语义不符。
* 用 `os.getcwd()` —— 否决，工作目录随调用位置变化，训练脚本从别的目录启动就会指向错误位置。

### 3.2 只有 `OUTPUT_GIFS_DIR` 是「交付物」，`output/` 下只有这两个目录

源码把这条写成注释 + 一个专用函数：

```python
#: 人要**看**的东西放这里。只有 ``gifs/`` 是交付件（``output/gifs/<名字>.gif``），
#: 其余是工作目录，谁写谁按需创建。
OUTPUT_DIR: Path = WORKSPACE / "output"
OUTPUT_GIFS_DIR: Path = OUTPUT_DIR / "gifs"
```

理由：`OUTPUT_GIFS_DIR` 存的是**给人看的最终动画**，是仓库的交付物。历史上这里还有 `stills/`（拼图 PNG）、`frames/`（逐帧转储）、`reports/`（渲染 JSON）、`exports/`（预留导出）四个常量，它们随 `scripts/render_checkpoint.py` 一起被删除 —— 那两个脚本与 **4 个 output 子目录常量**是同一批淘汰物（见 `experience.md` 的重构小节）。留下的教训仍然有效：**不要为了「以后可能用得上」预先声明目录**，空目录在归档与对比产物时是噪声，而交付物是否存在本身就是「渲染成功」的证据。

因此现在的形状是：`ensure_output_dirs()` 只建 `(OUTPUT_DIR, OUTPUT_GIFS_DIR)` 两项，其余任何工作目录由**真正要写东西的那个调用方**用 `ensure_dir(path)` 现建（`scripts/play.py:346` 写 GIF 前就走这条约定）。

**被否决的方案**：干脆不提供 `ensure_output_dirs()`，让每个写者都自己 `ensure_dir` —— 否决，因为 GIF 目录是**约定**（多个脚本都往同一个固定位置写），约定需要一个单一入口来保证一致，否则改目录名要改多处。

### 3.3 `ensure_dir` 返回路径，`ensure_output_dirs` 返回 `None`

```python
def ensure_dir(path: Path) -> Path:
    """``mkdir -p``，并把路径还回来（方便一行写完）。"""
```

理由：`ensure_dir` 的典型调用点是**表达式里** —— 把「建目录」和「用目录」写成一行：

```python
write_png(ensure_dir(some_work_dir) / "x.png", img)
```

返回 `None` 就只能先写一行语句。而 `ensure_output_dirs` 是「启动时一次性准备」，不需要返回值（返回一个目录列表只会诱导调用方去用它做别的事）。

**被否决的方案**：两个函数统一返回 `None` 或统一返回 `list[Path]` —— 前者牺牲一行式调用点，后者给批量函数凭空加了一个没人要的返回值。

### 3.4 创建一律用 `path.mkdir(parents=True, exist_ok=True)`

* `parents=True`：允许 `output/gifs` 在 `output/` 不存在时一次建成（对应 `mkdir -p`）。
* `exist_ok=True`：并发/重复调用不抛 `FileExistsError`。这条很重要，因为多个脚本会各自调用 `ensure_output_dirs()`，而本仓明确允许并行 run（上限见 `experience.md` §5.5）。

**被否决的方案**：先 `if not path.exists(): path.mkdir(...)` —— 否决，是 TOCTOU 竞态，且多一次系统调用。

### 3.5 常量全部用**类型注解**写（`WORKSPACE: Path = ...`）

理由：`Path` 与 `str` 的混淆是本项目最容易犯的类型错误（`path / "x"` 对 `str` 会退化成字符串拼接或抛类型错误）。带注解让 IDE 与类型检查器在调用点就能拦住「把常量当字符串用」。注释用 `#:` 前缀（Sphinx 属性文档语法），说明这些注释是**生成文档用**的，而不是普通行内注释。

### 3.6 `GENERATED_DIR` 用点号前缀，且与 `ARTIFACTS_DIR` 分开

```python
#: 动态生成的 MJCF 场景（不污染 vendored 的模型目录）。
GENERATED_DIR: Path = WORKSPACE / ".generated"
```

理由：这里放的是**调试产物**（编译失败的 MJCF 转储 `failed_scene.xml`，见 `robotdog/go2_model.py:423-426`），不是交付物、也不是可复现的实验产物。两条约束：(1) 不能写进 `MODELS_DIR` —— 那是 vendored 的 Unitree 模型目录，往里丢生成文件会污染「原样拷贝的第三方资产」；(2) 点号前缀让它在 `ls` 里默认不可见，与 `artifacts/`（要长期保留的实验产物）在视觉上区分开。

## 4. 公开 API

常量（签名逐字来自源码，`robotdog/tools/paths.py:10-29`）：

```python
WORKSPACE: Path = Path(__file__).resolve().parents[2]

MODELS_DIR: Path = WORKSPACE / "models"
GO2_DIR: Path = MODELS_DIR / "unitree_go2"
GO2_XML: Path = GO2_DIR / "go2.xml"
GO2_SCENE_XML: Path = GO2_DIR / "scene.xml"
GO2_ASSETS_DIR: Path = GO2_DIR / "assets"

SCRIPTS_DIR: Path = WORKSPACE / "scripts"
ARTIFACTS_DIR: Path = WORKSPACE / "artifacts"
RUNS_DIR: Path = WORKSPACE / "runs"
LOGS_DIR: Path = WORKSPACE / "logs"

#: 人要**看**的东西放这里。只有 ``gifs/`` 是交付件（``output/gifs/<名字>.gif``），
#: 其余是工作目录，谁写谁按需创建。
OUTPUT_DIR: Path = WORKSPACE / "output"
OUTPUT_GIFS_DIR: Path = OUTPUT_DIR / "gifs"

#: 动态生成的 MJCF 场景（不污染 vendored 的模型目录）。
GENERATED_DIR: Path = WORKSPACE / ".generated"
```

| 常量 | 值（相对工作区） | 语义 | 谁是使用者 |
| --- | --- | --- | --- |
| `WORKSPACE` | `<ws>` | 仓库根，唯一来源 | 所有派生的基准，很少被直接使用 |
| `MODELS_DIR` | `models/` | vendored 模型根 | 派生用 |
| `GO2_DIR` | `models/unitree_go2/` | Go2 模型目录 | 派生用 |
| `GO2_XML` | `models/unitree_go2/go2.xml` | 机器人的主 MJCF | `robotdog/go2_model.py:41` |
| `GO2_SCENE_XML` | `models/unitree_go2/scene.xml` | 官方场景 MJCF | 参考用 |
| `GO2_ASSETS_DIR` | `models/unitree_go2/assets` | mesh/纹理目录，编译 MJCF 时的 `meshdir` | `robotdog/go2_model.py:41` |
| `SCRIPTS_DIR` | `scripts/` | 脚本目录 | 当前**无使用者**（派生用） |
| `ARTIFACTS_DIR` | `artifacts/` | 长期实验产物（出厂先验 `trot_stairs.json`） | 常量本身**无代码使用者**，但 `config.yaml:39` 的 `env.gait_file: artifacts/trot_stairs.json` 经 `paths.WORKSPACE` 指向它（`robotdog/config.py:181`） |
| `RUNS_DIR` | `runs/` | 每次训练 run 的子目录 | `robotdog/training.py:39, 271` |
| `LOGS_DIR` | `logs/` | 日志目录 | 当前**无使用者**（课程报告已改写入 `runs/<name>/summary.json`） |
| `OUTPUT_DIR` | `output/` | 给人看的产物根 | `ensure_output_dirs()` |
| `OUTPUT_GIFS_DIR` | `output/gifs/` | **交付物**：动画 GIF | `scripts/play.py:346` |
| `GENERATED_DIR` | `.generated/` | 调试用生成的 MJCF | `robotdog/go2_model.py:423` |

函数（`robotdog/tools/paths.py:32-41`）：

```python
def ensure_dir(path: Path) -> Path:
    """``mkdir -p``，并把路径还回来（方便一行写完）。"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_dirs() -> None:
    """创建交付需要的 ``output/`` 与 ``output/gifs/``。"""
    for path in (OUTPUT_DIR, OUTPUT_GIFS_DIR):
        path.mkdir(parents=True, exist_ok=True)
```

| 函数 | 参数 | 返回 | 语义 | 异常 |
| --- | --- | --- | --- | --- |
| `ensure_dir(path: Path)` | `path`：要建的目录 | `Path`（原样返回入参） | 对**任意**目录做 `mkdir -p`，返回值便于一行式调用 | 权限/磁盘问题冒泡 `OSError`（`PermissionError` 是其子类） |
| `ensure_output_dirs()` | 无 | `None` | 只建 `OUTPUT_DIR` 与 `OUTPUT_GIFS_DIR` 两个**交付物**目录 | 同上 |

* 两个函数都不吞异常，都不打印，都不检查路径是否真的是目录（若同名文件已存在，`exist_ok=True` 仍会抛 `FileExistsError`）。
* `ensure_dir` 的入参注解是 `Path` 而非 `str | Path`，传字符串会在 `path.mkdir` 处抛 `AttributeError: 'str' object has no attribute 'mkdir'`。

## 5. 内部结构与数据流

常量是**声明式**的，唯一的数据流是「`__file__` → 根 → 派生」：

```
Path(__file__).resolve()                          # <ws>/robotdog/tools/paths.py
  → .parents[2]                                   # <ws>
      → WORKSPACE
          → WORKSPACE / "models"                  → MODELS_DIR
              → MODELS_DIR / "unitree_go2"        → GO2_DIR
                  → GO2_DIR / "go2.xml"           → GO2_XML
                  → GO2_DIR / "scene.xml"         → GO2_SCENE_XML
                  → GO2_DIR / "assets"            → GO2_ASSETS_DIR
          → WORKSPACE / "scripts"                 → SCRIPTS_DIR
          → WORKSPACE / "artifacts"               → ARTIFACTS_DIR
          → WORKSPACE / "runs"                    → RUNS_DIR
          → WORKSPACE / "logs"                    → LOGS_DIR
          → WORKSPACE / "output"                  → OUTPUT_DIR
              → OUTPUT_DIR / "gifs"               → OUTPUT_GIFS_DIR
          → WORKSPACE / ".generated"              → GENERATED_DIR
```

两个创建辅助各自独立，互不调用（`ensure_output_dirs` 不调用 `ensure_dir`，它直接写 `mkdir`，因为返回值无用）：

```
ensure_dir(path)      → path.mkdir(parents=True, exist_ok=True)   → return path
ensure_output_dirs()  → path.mkdir(parents=True, exist_ok=True)   × 2
```

调用方侧的典型链（给出 GIF）：

```
scripts/play.py:48   from robotdog.tools import console, paths
  → :346  gif = paths.OUTPUT_GIFS_DIR / f"{name}.gif"
```

模型编译侧的链（延迟导入 + 仅在失败时落盘）：

```
robotdog/go2_model.py:41   from robotdog.tools.paths import GO2_ASSETS_DIR, GO2_XML
robotdog/go2_model.py:423  from robotdog.tools.paths import GENERATED_DIR   # 函数内，失败分支
              → :425  GENERATED_DIR.mkdir(parents=True, exist_ok=True)
              → :426  dump = GENERATED_DIR / "failed_scene.xml"
              → :428  raise RuntimeError(f"MJCF compile failed; dumped to {dump}") from exc
```

训练产物侧的链（默认值来自工具层、显式覆盖来自调用方）：

```
robotdog/training.py:39    from robotdog.tools.paths import RUNS_DIR
robotdog/training.py:271   self.root = Path(root) if root else (RUNS_DIR / name)
```

## 6. 关键常量与默认值

| 项 | 真实值 |
| --- | --- |
| 上溯级数 | `3`（`parents[2]`），源码注释：本文件在 `<workspace>/robotdog/tools/` 下，所以往上走三级 |
| 历史级数 | `2`（文件还叫 `robotdog/paths.py` 时的取值） |
| 常量总数 | **13** 个 `Path`（含 `WORKSPACE`） |
| `ensure_output_dirs()` 建立的目录数 | 2（`OUTPUT_DIR`、`OUTPUT_GIFS_DIR`） |
| `output/` 子目录数 | **1**（`gifs`），且它是唯一的交付物 |
| `mkdir` 关键字参数 | `parents=True, exist_ok=True` |
| 实测解析结果（本机） | `WORKSPACE` = `D:\项目\RobortDog\upstair_core` |
| 实测已存在 | 全部 13 个（13/13 `exists() == True`；`output/` 与 `output/gifs/` 由 `play.py --mode gif` 建立） |

度量方法（可复现）：`for n in dir(paths) if n.isupper(): getattr(paths, n).exists()`。

## 7. 与其它模块的关系

**上游**：仅 stdlib —— `from __future__ import annotations`、`from pathlib import Path`。**不 import numpy、不 import mujoco、不 import 任何 robotdog 模块**。

**下游**（全仓扫描，仅三处）：

| 位置 | 导入/使用的符号 | 依赖方向为什么是这样 |
| --- | --- | --- |
| `robotdog/go2_model.py:41` | `GO2_ASSETS_DIR`, `GO2_XML` | 编译 MJCF 需要机器人主文件与 mesh 目录；这是**环境层**对**工具层**的依赖，方向向下 |
| `robotdog/go2_model.py:423` | `GENERATED_DIR` | 编译失败要落盘转储；用函数内延迟导入，避免为一个失败分支把常量绑进模块顶层 |
| `robotdog/training.py:39` | `RUNS_DIR` | 训练产物根目录；`:271` `self.root = Path(root) if root else (RUNS_DIR / name)`，即「默认值来自工具层，显式覆盖来自调用方」 |
| `robotdog/config.py:34` | 整体导入 `paths`（用 `paths.WORKSPACE`） | 定位仓根的 `config.yaml`；配置层需要知道仓库在哪 |
| `scripts/play.py:48` | 整体导入 `paths`（用 `paths.OUTPUT_GIFS_DIR`） | 写 GIF 交付物 |

**为什么方向只能是这个方向**：路径常量是**零知识**信息（只需要知道仓库长什么样），所以它可以在依赖图最底端被任何层读取；反过来它一旦 import 任何项目模块，就从最底端变成环的一部分。这也解释了为什么本文件**不允许**包含「确保文件存在」之类的校验逻辑 —— 那需要知道业务语义（哪个文件该存在），属于环境层/训练层。

**导入路径唯一**：规范路径是 `robotdog.tools.paths`。仓里**没有**顶层 `robotdog/paths.py`，`from robotdog.paths import ...` 与 `from robotdog import paths` 都会 `ModuleNotFoundError`。早期版本用 `sys.modules[__name__] = _mod` 的替身模块保留过这两条旧路径，已删除 —— 替身本身没有逻辑，只会让 traceback 与 IDE 静态分析指错文件。

## 8. 实测与经验教训

* 并行 run 的上限（为什么重复 `mkdir(exist_ok=True)` 是必要的）、以及后台 run 的输出丢失，见 `experience.md` §5.5、§5.6。
* Windows 下把脚本输出重定向到文件时的编码/引号坑（会影响「拿到的日志路径里有什么」），见 `experience.md` §5.8。
* 训练产物的目录结构（`runs/<name>/` 下有什么）与课程如何组织，见 `train.md` §1、§2「产物」（本文档不重复其结论）。
* **本文档是「常量删了、文档没删」的典型受害样本**：`CONFIGS_DIR` 与四个 `output/` 子目录常量、以及 `ensure_dirs()` 这个「启动时一次建六个仓库级目录」的函数都是**早期版本**的遗留描述。而 `.tmp/check_lines.py` 当时只抓到**三条行号问题**（两条越界、一条指向空行）—— 「函数根本不存在」「常量少了五个」这类事实**行号检查永远抓不到**，只能靠对着源码逐条重写。所以下面这段文字不再写具体行号：一旦源文件再变，它们自己就会变成下一批腐坏引用。

## 9. 已知限制与坑

* **位置敏感**：把 `robotdog/tools/paths.py` 再挪一层目录，`parents[2]` 立刻错位。历史症状是
  `FileNotFoundError: ...\robotdog\models\unitree_go2\go2.xml`
  —— 路径里多出一个 `robotdog\` 段，说明 `WORKSPACE` 被解析成了 `<ws>/robotdog`（即 `parents[1]`）。看到这个报错先查 `parents[N]`，不要先怀疑文件真的丢了。
* **没有 `ensure_dirs()` 这个函数**。它曾在早期版本里一次性创建 `models/` / `configs/` / `artifacts/` / `runs/` / `logs/` / `.generated/`，已随 `CONFIGS_DIR` 一起删除。现在**没有任何代码创建仓库级目录** —— 它们是仓库快照的一部分（`models/`、`artifacts/`）或由使用者按需创建（`runs/` 由 `RunDir.ensure()`）。想一次建好只能自己循环调 `ensure_dir`。
* **没有 `CONFIGS_DIR`**。配置不再是「一个 `configs/` 目录」，而是仓根的单文件 `config.yaml`（见 `docs/robotdog/config.md`）。
* **`SCRIPTS_DIR` / `ARTIFACTS_DIR` / `LOGS_DIR` 三个常量当前没有代码使用者**（`grep` 只命中定义处与 `__all__`）。它们是声明式的「位置表」条目，改动它们不会影响任何行为 —— 但也意味着**它们指向的目录不会被自动创建**。三个各有不同的情况：`artifacts/` 是仓库快照的一部分（装着出厂先验 `trot_stairs.json`，由 `config.yaml:39` 的**字符串常量**而非本常量指向）；`scripts/` 是仓库快照本身；`LOGS_DIR` 尤其容易误导 —— 课程报告现在写到 `runs/<name>/summary.json`（`robotdog/training.py:298` 的 `RunDir.summary_path`），`logs/` 只是历史遗留的空目录。
* **`ensure_output_dirs()` 只保证 `output/` 与 `output/gifs/` 存在**，不保证任何其他子目录。若将来重新引入工作目录，必须自己先 `ensure_dir(...)`。
* **传字符串给 `ensure_dir` 会抛 `AttributeError: 'str' object has no attribute 'mkdir'`**，因为注解是 `Path` 且内部直接调 `path.mkdir`。
* **`ensure_dir` 不能用于文件路径**：对 `<...>/x.png` 调 `ensure_dir` 会创建出一个**名为 `x.png` 的目录**，后续 `write_bytes` 报 `PermissionError` / `IsADirectoryError`。要建的是 `path.parent`。
* **`output/` 可能不存在**（干净 clone 里就没有），所以直接读 `paths.OUTPUT_GIFS_DIR / "x.gif"` 之前必须调用 `ensure_output_dirs()`，否则 `save_gif` 走 Pillow 会因父目录缺失失败（`write_png` 自己会 `path.parent.mkdir`，但 Pillow 的保存路径不会）。
* **没有「删除/清理」入口**：想清空产物只能自己在脚本里 `shutil.rmtree`，本模块不提供，避免误删 `runs/`。

## 10. 示例用法

```python
from robotdog.tools.paths import (
    LOGS_DIR,
    OUTPUT_GIFS_DIR,
    RUNS_DIR,
    ensure_dir,
    ensure_output_dirs,
)

# 1) 只读常量：不产生任何副作用
print(RUNS_DIR)                       # D:\项目\RobortDog\upstair_core\runs
print(LOGS_DIR / "run_curriculum.json")

# 2) 交付物目录（output/ 与 output/gifs/）
ensure_output_dirs()

# 3) 按需工作目录：先建父目录，再用返回值拼路径
work = ensure_dir(RUNS_DIR / "_scratch")
report = work / "render_run.json"
print(report)

# 4) 常用的整体导入（写明自己用的是工具层，避免只看到一个 `paths` 不知道来自哪层）
from robotdog.tools import paths
assert paths.OUTPUT_GIFS_DIR == OUTPUT_GIFS_DIR
```

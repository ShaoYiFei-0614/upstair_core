# `robotdog/tools/paths.py` —— 项目文件系统位置

## 1. 一句话定位

**工具层**（`robotdog/tools/`）里的**纯数据模块**：把整个项目的目录与关键文件位置定义成一组 `pathlib.Path` 常量。它解决的问题是「同一个目录被十个模块用十种写法拼出来」——所有派生路径只有一个来源，改目录结构只改这一个文件。

## 2. 职责边界

**做**：

* 定义唯一的工作区根 `WORKSPACE`，其余全部由它派生（模型、实验产物、交付物、生成物）。
* 用注释把「哪些是交付物、哪些是按需创建的工作目录」写死在常量旁边。

**不做**：

* **不做 glob / 不搜文件、不校验文件是否存在、不读文件内容**。
* **不 import 任何项目模块**（连 `robotdog.tools.*` 都不），只依赖 stdlib `pathlib`。
* **不在 import 时创建任何目录**（模块顶层没有任何 `mkdir` 调用）。
* **不提供目录创建辅助**。历史上这里有过 `ensure_dir(path)` / `ensure_output_dirs()` 两个函数，在死代码清理中被删除（见 §8）——现在**本模块只有常量，没有任何函数**。
* **不清理 / 不删除 / 不归档**目录。
* **不处理 CLI 覆盖**（例如「用 `--root` 指定另外的输出目录」在调用方实现）。
* **不 import `robotdog.env`**，不知道 PPO / 课程 / 训练循环。

## 3. 关键设计决策与理由

### 3.1 `WORKSPACE` 用 `Path(__file__).resolve().parents[2]`，并配一条显式注释

```python
# .../upstair_core —— 本文件在 <workspace>/robotdog/tools/ 下，所以往上走三级。
# （它原本是 robotdog/paths.py，只走两级；下沉到 tools 子包后所有派生路径都指向了
# robotdog/ 里面，直到冒烟测试才被抓住，故留这条注释。）
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

### 3.2 只有 `OUTPUT_GIFS_DIR` 是「交付物」

```python
#: 人要**看**的东西放这里。只有 ``gifs/`` 是交付件（``output/gifs/<名字>.gif``），
#: 其余是工作目录，谁写谁按需创建。
OUTPUT_DIR: Path = WORKSPACE / "output"
OUTPUT_GIFS_DIR: Path = OUTPUT_DIR / "gifs"
```

理由：`OUTPUT_GIFS_DIR` 存的是**给人看的最终动画**，是仓库的交付物。历史上这里还有 `stills/`（拼图 PNG）、`frames/`（逐帧转储）、`reports/`（渲染 JSON）、`exports/`（预留导出）四个常量，它们随 `scripts/render_checkpoint.py` 一起被删除 —— 那两个脚本与 **4 个 output 子目录常量**是同一批淘汰物（见 `experience.md` 的重构小节）。留下的教训仍然有效：**不要为了「以后可能用得上」预先声明目录**，空目录在归档与对比产物时是噪声，而交付物是否存在本身就是「渲染成功」的证据。

### 3.3 目录由**真正要写东西的调用方**现建（本次清理的决定）

本模块**不提供**创建辅助。当前仓内的三条创建路径全部在调用方：

| 目录 | 谁建 | 代码位置 |
| --- | --- | --- |
| `output/` + `output/gifs/` | `scripts/play.py` 的 `write_gif`，保存前建父目录 | `scripts/play.py:248` `path.parent.mkdir(parents=True, exist_ok=True)` |
| `runs/<名字>/` 及其子目录 | `robotdog/training.py` 的 `RunDir.ensure()` | `robotdog/training.py:264` |
| `.generated/` | `robotdog/go2_model.py` 的 `compile_scene`（编译失败才建） | `robotdog/go2_model.py:405-408` |

**被否决的方案（历史，已随函数一起删除）**：用 `ensure_output_dirs()` 做「启动时一次性准备」，再用 `ensure_dir(path)` 给任意工作目录做 `mkdir -p` 并返回路径（支持 `write_bytes(ensure_dir(d) / "x.bin", blob)` 这种一行式写法）。否决理由是**实证的**：这两个函数从写出来到删除，**全仓零调用者**——`play.py` 由 `write_gif` 自建父目录，`RUNS_DIR` 由 `RunDir.ensure()` 建，`GENERATED_DIR` 由 `compile_scene` 建。一个「约定单一入口」如果没有任何人走这个入口，它就只是**多一条可以腐坏的 API 面**（`ensure_dir` 还有「传字符串抛 `AttributeError: 'str' object has no attribute 'mkdir'`」「对文件路径调用会建出同名目录」两个坑）。真正的约定现在写在常量注释里，而不是写在一个没人调的函数里。

### 3.4 常量全部用**类型注解**写（`WORKSPACE: Path = ...`）

理由：`Path` 与 `str` 的混淆是本项目最容易犯的类型错误（`path / "x"` 对 `str` 会退化成字符串拼接或抛类型错误）。带注解让 IDE 与类型检查器在调用点就能拦住「把常量当字符串用」。注释用 `#:` 前缀（Sphinx 属性文档语法），说明这些注释是**生成文档用**的，而不是普通行内注释。

### 3.5 `GENERATED_DIR` 用点号前缀

```python
#: 动态生成的 MJCF 场景（不污染 vendored 的模型目录）。
GENERATED_DIR: Path = WORKSPACE / ".generated"
```

理由：这里放的是**调试产物**（编译失败的 MJCF 转储 `failed_scene.xml`，见 `robotdog/go2_model.py:405-408`），不是交付物、也不是可复现的实验产物。两条约束：(1) 不能写进 `MODELS_DIR` —— 那是 vendored 的 Unitree 模型目录，往里丢生成文件会污染「原样拷贝的第三方资产」；(2) 点号前缀让它在 `ls` 里默认不可见，与要长期保留的实验产物 `artifacts/` 在视觉上区分开。

> **注**：`artifacts/` 目录**仍然是有效载荷**（装着出厂步态先验 `artifacts/trot_stairs.json`），但它现在**由 `config.yaml` 的字符串常量 `env.gait_file` 经 `paths.WORKSPACE` 指向**，没有任何 `ARTIFACTS_DIR` 常量 —— 该常量已随死代码清理删除（§8）。

## 4. 公开 API

常量（签名逐字来自源码，`robotdog/tools/paths.py:10-25`）——**9 个，没有函数**：
```python
WORKSPACE: Path = Path(__file__).resolve().parents[2]

MODELS_DIR: Path = WORKSPACE / "models"
GO2_DIR: Path = MODELS_DIR / "unitree_go2"
GO2_XML: Path = GO2_DIR / "go2.xml"
GO2_ASSETS_DIR: Path = GO2_DIR / "assets"

RUNS_DIR: Path = WORKSPACE / "runs"

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
| `GO2_XML` | `models/unitree_go2/go2.xml` | 机器人的主 MJCF | `robotdog/go2_model.py:38` |
| `GO2_ASSETS_DIR` | `models/unitree_go2/assets` | mesh/纹理目录，编译 MJCF 时的 `meshdir` | `robotdog/go2_model.py:38` |
| `RUNS_DIR` | `runs/` | 每次训练 run 的子目录 | `robotdog/training.py:37` |
| `OUTPUT_DIR` | `output/` | 给人看的产物根 | `scripts/play.py`（经 `write_gif` 建父目录） |
| `OUTPUT_GIFS_DIR` | `output/gifs/` | **交付物**：动画 GIF | `scripts/play.py:438`（`mode_gif` 里给 `write_gif` 定落点） |
| `GENERATED_DIR` | `.generated/` | 调试用生成的 MJCF | `robotdog/go2_model.py:405` |

`__all__`（`robotdog/tools/paths.py:28-38`）与上表逐字一致，9 项。

**已删除、不要再引用**：`SCRIPTS_DIR`、`ARTIFACTS_DIR`、`LOGS_DIR`、`GO2_SCENE_XML`（= `models/unitree_go2/scene.xml`）、`ensure_dir()`、`ensure_output_dirs()`。其中 `LOGS_DIR` 最具误导性——课程报告早已改写到 `runs/<名字>/summary.json`（`robotdog/training.py` 的 `RunDir.summary_path`）。

## 5. 内部结构与数据流

常量是**声明式**的，唯一的数据流是「`__file__` → 根 → 派生」：

```text
Path(__file__).resolve()                          # <ws>/robotdog/tools/paths.py
  → .parents[2]                                   # <ws>
      → WORKSPACE
          → WORKSPACE / "models"                  → MODELS_DIR
              → MODELS_DIR / "unitree_go2"        → GO2_DIR
                  → GO2_DIR / "go2.xml"           → GO2_XML
                  → GO2_DIR / "assets"            → GO2_ASSETS_DIR
          → WORKSPACE / "runs"                    → RUNS_DIR
          → WORKSPACE / "output"                  → OUTPUT_DIR
              → OUTPUT_DIR / "gifs"               → OUTPUT_GIFS_DIR
          → WORKSPACE / ".generated"              → GENERATED_DIR
```

**模块体内没有第二条语句**：没有 `mkdir`、没有 I/O、没有函数定义。`import robotdog.tools.paths` 的副作用只有构造 9 个 `Path` 对象。

调用方侧的三条典型链：

```text
[编译场景] robotdog/go2_model.py:38 from robotdog.tools.paths import GO2_ASSETS_DIR, GO2_XML
           robotdog/go2_model.py:412（函数内延迟导入）from robotdog.tools.paths import GENERATED_DIR
           robotdog/go2_model.py:414 GENERATED_DIR.mkdir(parents=True, exist_ok=True)
                                :415 dump = GENERATED_DIR / "failed_scene.xml"

[实验产物] robotdog/training.py:37 from robotdog.tools.paths import RUNS_DIR
           robotdog/training.py:256 self.root = Path(root) if root else (RUNS_DIR / name)
           robotdog/training.py:264 def ensure(self) -> Path: （真正 mkdir 的地方）

[交付 GIF] scripts/play.py:48 from robotdog.tools import console, paths  # noqa: E402
           scripts/play.py:248 path.parent.mkdir(parents=True, exist_ok=True)（在 write_gif 内）
           scripts/play.py:362 gif = paths.OUTPUT_GIFS_DIR / f"{name}.gif"
```

## 6. 关键常量与默认值

| 项 | 值 |
| --- | --- |
| 常量数 | 9 |
| 函数数 | **0** |
| `__all__` 项数 | 9 |
| 模块总行数 | **38** |
| import 依赖 | stdlib `pathlib` 一项 |

## 7. 与其它模块的关系

* **向上**被 `robotdog/go2_model.py`（`GO2_ASSETS_DIR`、`GO2_XML`、`GENERATED_DIR`）、`robotdog/training.py`（`RUNS_DIR`）、`robotdog/config.py`（`WORKSPACE`，用于解析 `config.yaml` 里 `env.gait_file` 这种相对路径）、`scripts/play.py`（`OUTPUT_GIFS_DIR`）使用。
* **向下**不依赖任何模块，也不依赖 `robotdog.tools` 的其它三个模块。
* `robotdog/tools/__init__.py:16` 把它列进包级导入，所以 `from robotdog.tools import paths` 与 `from robotdog.tools.paths import ...` 等价。**`import robotdog.tools.paths` 只加载 numpy 之外的 stdlib**（本模块自身不拉 Mujoco）。

## 8. 实测与经验教训

* **本轮死代码清理删掉的 6 个符号**：

  | 符号 | 曾经的用途 | 为什么删 |
  | --- | --- | --- |
  | `ensure_dir(path: Path) -> Path` | `mkdir -p` 并返回路径，便于一行式调用 | 全仓零调用者；且有两个坑（传 `str` 抛 `AttributeError`、对文件路径建出同名目录） |
  | `ensure_output_dirs() -> None` | 启动时建 `output/` 与 `output/gifs/` | 全仓零调用者；`write_gif` 自己建父目录 |
  | `GO2_SCENE_XML` | `models/unitree_go2/scene.xml` 的路径 | 本仓从不用官方 `scene.xml`（场景由 `robotdog/go2_model.py:compile_scene()` 现场拼 XML 字符串）；`scene.xml` 只在 `scripts/download_go2.py:43` 的 `WANT_FILES` 里以字面量出现 |
  | `SCRIPTS_DIR` / `ARTIFACTS_DIR` / `LOGS_DIR` | 早期版本的目录常量 | 只命中定义与 `__all__`；`LOGS_DIR` 尤其误导（报告早已改写到 `runs/<名字>/summary.json`） |

* **本文档曾是「常量删了、文档没删」的典型受害样本**：`CONFIGS_DIR`、四个 `output/` 子目录常量、`ensure_dirs()`，以及后来死掉的 `SCRIPTS_DIR` / `ARTIFACTS_DIR` / `LOGS_DIR` 都曾是这里的描述对象。而 `.tmp/check_lines.py` 当时只抓到**三条行号问题**（两条越界、一条指向空行）——「函数根本不存在」「常量少了五个」这类事实**行号检查永远抓不到**，只能靠对着源码逐条重写。
* **推论**：本次清理之后，本模块的「越界行号」风险降到最低（38 行、9 个常量），但**符号存废**仍然是人工核对项。

## 9. 已知限制与坑

* **没有 `ensure_dirs()` 这个函数**。它曾在早期版本里一次性创建 `models/` / `configs/` / `artifacts/` / `runs/` / `logs/` / `.generated/`，已随 `CONFIGS_DIR` 一起删除。现在**没有任何代码创建仓库级目录** —— 它们是仓库快照的一部分（`models/`、`artifacts/`）或由使用者按需创建（`runs/` 由 `RunDir.ensure()`）。想一次建好只能自己循环 `path.mkdir(parents=True, exist_ok=True)`。
* **`output/` 可能不存在**（干净 clone 里就没有），所以直接读 `paths.OUTPUT_GIFS_DIR / "x.gif"` 之前必须保证父目录存在 —— 现在由 `write_gif` 自己保证（Pillow 的 `save` 不会建父目录）。
* **没有「删除/清理」入口**：想清空产物只能自己在脚本里 `shutil.rmtree`，本模块不提供，避免误删 `runs/`。
* **`GO2_DIR` 里不含 `scene.xml` 常量**：需要读官方场景请自己拼 `paths.GO2_DIR / "scene.xml"`，但要先确认文件真的在（`scripts/download_go2.py` 会下载它）。
* **本模块不保证任何路径存在**：`Path` 的构造永远成功，`exists()` 才是判据。

## 10. 示例用法

```python
from robotdog.tools import paths

# 1) 只读常量：不产生任何副作用
print(paths.RUNS_DIR)                 # D:\项目\RobortDog\upstair_core\runs
print(paths.OUTPUT_GIFS_DIR)          # D:\项目\RobortDog\upstair_core\output\gifs

# 2) 写文件前自己建父目录（本模块不再提供 ensure_* 辅助）
gif = paths.OUTPUT_GIFS_DIR / "model_final.gif"
gif.parent.mkdir(parents=True, exist_ok=True)
gif.write_bytes(b"...")

# 3) 判定存在性只能靠 exists()，常量本身不代表文件存在
assert paths.GO2_XML.exists(), f"缺少 Go2 主模型：{paths.GO2_XML}"

# 4) 从 config.yaml 的相对路径还原绝对路径
from robotdog.config import load_config
raw = load_config()
gait_file = paths.WORKSPACE / raw["env"]["gait_file"]
print(gait_file)                      # ...\artifacts\trot_stairs.json
```

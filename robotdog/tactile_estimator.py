"""足底触觉的 GRU 状态估计器：从本体感觉观测回归「三维接触力 / 接触 / 面积 / CoP」。

为什么是 GRU（而不是逐帧 MLP）
----------------------------
足底力不是当前状态的纯函数：落地冲击的峰值、支撑相的负载分配都依赖过去若干帧的
运动历史（腿是否在摆动、脚离地多高、机身有没有下沉）。GRU 用固定长度时间窗
（默认 20 步 = 0.4 s @50 Hz）同时给出：

* **三维接触力**（世界系 + 足体系，每足 3 维）—— 主任务；
* **接触概率**（每足 1 维）—— 接触检测；
* **接触面积**（每足 1 维，仅在接触时监督）—— 辅助任务；
* **CoP**（每足 2 维，仅在接触时监督）—— 辅助任务。

消费端对齐（可选，默认全关）
--------------------------
只看三维力的 R² 会高估能力：它大部分来自「摆动相≈0」的接触判别，支撑相的幅度其实
向均值收缩（留出楼梯上预测/真值均值比 0.79–0.81、回归斜率 0.65）。``EstimatorConfig``
里的 ``w_forcenorm`` / ``w_delta`` / ``w_peak`` / ``w_magrel`` /
``physics_weight`` / ``w_lookahead`` 打开时会额外建头并加损失，直接监督**消费端
真正要用的量**：F̄（法向力/体重）、ΔF̄、落地窗峰值、相对幅值，以及「支撑帧
Σ_f F_z ≈ 体重」（时间平均口径）的物理标定。默认全 0 ⇒ 不建新头、不加新损失，历史 checkpoint
逐键可载入、历史训练逐位等价。

标签来自``robotdog/tactile.py``的虚拟足底测量（仿真真值），特征只用**策略可见的
本体感觉观测**（``Go2StairsEnv._observation()``，默认 53 维）。评估时不看训练地形
的数字——高台阶（16.5/18 cm）与 2–5 cm 随机粗糙地面都是**留出集**。

这个模块只做「训练/评估/推理」三件事，不 import 环境、不读 MJCF；数据采集在
``scripts/collect_tactile_data.py``，训练/评估在 ``scripts/train_tactile_estimator.py``
与 ``scripts/eval_tactile_estimator.py``。
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

DEFAULT_FORCE_REF_N = 149.17   # Go2 15.206 kg × 9.81（触觉归一化用的体重）

# --------------------------------------------------------------------------- #
# 关节力矩特征（v3）：12 个位置执行器的力矩 / 力矩上限
# --------------------------------------------------------------------------- #
# 顺序与 go2.xml 的执行器声明一致：FL_hip, FL_thigh, FL_calf, FR_..., RL_..., RR_...。
# 上限镜像自 go2.xml 的 <motor class=... ctrlrange=...>（abduction 23.7、
# thigh 23.7、knee 45.43 N·m）；这里**只做归一化尺度**，不参与任何控制限幅。
TORQUE_REF_NM: tuple[float, ...] = tuple(
    ref for _ in range(4) for ref in (23.7, 23.7, 45.43))
ACTUATOR_NAMES: tuple[str, ...] = tuple(
    f"{leg}_{suffix}" for leg in ("FL", "FR", "RL", "RR")
    for suffix in ("hip", "thigh", "calf"))
TORQUE_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"tau_{name}_norm" for name in ACTUATOR_NAMES)


def torque_features(tau: np.ndarray) -> np.ndarray:
    """12 维执行器力矩 ⇒ 按上限归一化的 12 维特征（加到 53 维观测末尾）。"""
    t = np.asarray(tau, dtype=np.float32).reshape(-1)
    if t.size != len(TORQUE_REF_NM):
        raise ValueError(f"力矩维数应为 {len(TORQUE_REF_NM)}，实际 {t.size}")
    return (t / np.asarray(TORQUE_REF_NM, dtype=np.float32)).astype(np.float32)


# --------------------------------------------------------------------------- #
# v5 架构规格：四任务（力/接触/面积/CoP）可以共享一个 GRU，也可以各自独立
# --------------------------------------------------------------------------- #
TASKS: tuple[str, ...] = ("force", "contact", "area", "cop")
HEAD_DIMS: dict[str, int] = {"force": 12, "contact": 4, "area": 4, "cop": 8}
# 六条预注册臂的每任务 (层数, 隐层) 规格（参数总量见 ARCH_PARAM_TARGETS）
ARCH_SPECS: dict[str, dict[str, tuple[int, int]]] = {
    "S1": {"force": (1, 128), "contact": (1, 128), "area": (1, 128), "cop": (1, 128)},
    "I1": {"force": (1, 64), "contact": (1, 64), "area": (1, 64), "cop": (1, 64)},
    "S3": {"force": (3, 128), "contact": (3, 128), "area": (3, 128), "cop": (3, 128)},
    "I2": {"force": (2, 128), "contact": (1, 128), "area": (1, 64), "cop": (1, 64)},
    "I3": {"force": (3, 128), "contact": (3, 128), "area": (3, 128), "cop": (3, 128)},
    "I4": {"force": (3, 128), "contact": (1, 128), "area": (1, 128), "cop": (1, 128)},
    # v6：四独立两层 64（单任务 50 892）。容量落在 I1(102 428) 与 I2(301 980) 之间，
    # 是「宽度减半 + 深度 2 层」的中间点，用来分离「深度」与「参数量」。
    "I2x64": {"force": (2, 64), "contact": (2, 64), "area": (2, 64), "cop": (2, 64)},
}
# 输入 65 维、四头（force12/contact4/area4/cop8）时的参数总量（测试常数表）
ARCH_PARAM_TARGETS: dict[str, int] = {
    "S1": 78492, "I1": 102428, "S3": 276636,
    "I2": 301980, "I3": 1095708, "I4": 501276,
    "I2x64": 202268,
}


def parse_task_caps(spec: str) -> dict[str, tuple[int, int]]:
    """解析 "force:2x128,contact:1x128,area:1x64,cop:1x64"。

    四个任务必须各出现一次；层数与隐层都必须是正整数。顺序/空白不影响结果。
    """
    caps: dict[str, tuple[int, int]] = {}
    for chunk in str(spec).replace(" ", "").split(","):
        if not chunk:
            continue
        if ":" not in chunk or "x" not in chunk.split(":", 1)[1]:
            raise ValueError(f"容量片段格式应为 task:LxH，实际 {chunk!r}")
        task, rest = chunk.split(":", 1)
        if task not in HEAD_DIMS:
            raise ValueError(f"未知任务 {task!r}（可选 {TASKS}）")
        if task in caps:
            raise ValueError(f"任务 {task} 重复出现")
        layers_s, hidden_s = rest.split("x", 1)
        layers, hidden = int(layers_s), int(hidden_s)
        if layers < 1 or hidden < 1:
            raise ValueError(f"{task} 的层数/隐层必须 >= 1，实际 {layers}x{hidden}")
        caps[task] = (layers, hidden)
    missing = [task for task in TASKS if task not in caps]
    if missing:
        raise ValueError(f"容量串缺少任务 {missing}（四个任务都要给）")
    return caps


def arch_spec(arch: str, caps: str | None = None,
              layers: int | None = None,
              hidden: int | None = None) -> dict[str, tuple[int, int]]:
    """返回 arch 的每任务容量。

    shared：四个任务共用一层 GRU，容量由 layers/hidden 决定
    （缺省 1x128 = S1；S3 必须传 3/128，否则 checkpoint 自证会写成 1x128）。
    indep：用 caps（缺省 1x64 = I1）。
    """
    if arch == "shared":
        return {task: (int(layers if layers else 1),
                       int(hidden if hidden else 128)) for task in TASKS}
    if caps:
        return parse_task_caps(caps)
    return ARCH_SPECS["I1"]


@dataclass
class EstimatorConfig:
    """估计器超参（数据采集/训练/评估共用；存档在 checkpoint 里）。"""

    window: int = 20          # 输入时间窗 [控制步]（50 Hz ⇒ 0.4 s）
    hidden: int = 128
    layers: int = 1
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch: int = 256
    epochs: int = 40
    val_frac: float = 0.1
    seed: int = 0
    w_force: float = 1.0
    w_contact: float = 0.2
    w_area: float = 0.05
    w_cop: float = 0.05
    contact_threshold: float = 0.5
    force_ref_n: float = DEFAULT_FORCE_REF_N
    # ---- 消费端对齐项（默认全关：不建头、不加损失，历史行为逐位不变）----
    w_forcenorm: float = 0.0     # F̄ 头（法向力/体重，4 足）
    w_delta: float = 0.0         # ΔF̄ 头（单步变化，4 足）
    w_peak: float = 0.0          # 落地窗峰值头（4 足）
    w_magrel: float = 0.0        # 相对幅值损失（对抗向均值收缩/峰值低估）
    w_fbar_rel: float = 0.0      # 接触足上的相对 F̄ 损失（与消费端 slope/均值比同量）
    # ---- v4：支撑相加权 / 解耦头容量（默认关：不改变历史行为）----
    w_stance: float = 0.0        # 支撑相(phase>2.5)帧的额外损失权重倍率
    head_hidden: int = 0         # F̄ 头隐层宽度（0=单层线性；>0 时 Linear-GELU-Linear）
    physics_weight: float = 0.0  # 批次平均 Σ_f F_z ≈ 体重（huber，标定幅值）
    lookahead_contact: int = 0   # 接触前瞻帧数（0/1/2；>0 时必须给 w_lookahead）
    w_lookahead: float = 0.0     # 前瞻接触 BCE 权重
    # ---- v5：架构（共享/分任务）与训练调度 ----
    arch: str = "shared"         # shared=四头共享一个 GRU（历史行为）；indep=每任务独立 GRU
    task_caps: str = ""          # indep 时每任务容量，如 "force:2x128,contact:1x128,area:1x64,cop:1x64"
    lr_gamma: float = 1.0        # 每 epoch 学习率指数衰减（1.0=不衰减）
    early_stop: int = 0          # 连续多少 epoch val 无改善就停（0=不早停）
    patience: int = 5            # early_stop 的耐心（early_stop>0 时生效）
    # ---- v7 Wave C：双频输出（接触 50 Hz 瞬时 + 幅值因果均值）与训练端去抖 ----
    bifreq_k: int = 0            # >0：F̄ 头目标改成「以当前帧结尾的因果 k 帧均值」
    debounce_contact: bool = False   # 训练侧接触标签用 3 帧 ≥2 帧的因果多数票
    # 训练侧标签限幅（单位 mg）。仿真偶发单帧接触穿透会给 10–7000 mg 的尖峰，
    # 少量尖峰就能主导均方损失（0=不限幅，仅用于复现历史行为）。
    label_clip_mg: float = 10.0

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError(f"window 必须 >= 2，实际 {self.window}")
        if self.batch <= 0 or self.epochs <= 0:
            raise ValueError("batch / epochs 必须为正")
        if not (0.0 <= self.val_frac < 0.5):
            raise ValueError(f"val_frac 必须在 [0, 0.5)，实际 {self.val_frac}")
        if not (0 <= int(self.lookahead_contact) <= 4):
            raise ValueError(
                f"lookahead_contact 必须在 0..4，实际 {self.lookahead_contact}")
        if int(self.lookahead_contact) > 0 and self.w_lookahead <= 0.0:
            raise ValueError("lookahead_contact > 0 时 w_lookahead 必须 > 0")
        for name in ("w_forcenorm", "w_delta", "w_peak", "w_magrel",
                     "w_fbar_rel", "w_stance", "physics_weight", "w_lookahead"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} 必须 >= 0，实际 {getattr(self, name)}")
        if int(self.head_hidden) < 0:
            raise ValueError(f"head_hidden 必须 >= 0，实际 {self.head_hidden}")
        if self.arch not in ("shared", "indep"):
            raise ValueError(f"arch 只能是 shared/indep，实际 {self.arch}")
        if float(self.lr_gamma) <= 0.0:
            raise ValueError(f"lr_gamma 必须 > 0，实际 {self.lr_gamma}")
        if int(self.early_stop) < 0 or int(self.patience) < 1:
            raise ValueError("early_stop 必须 >= 0 且 patience >= 1")
        if self.task_caps:
            parse_task_caps(self.task_caps)   # 早失败：非法容量串在构造期就报错
        if not (0 <= int(self.bifreq_k) <= 50):
            raise ValueError(f"bifreq_k 必须在 0..50，实际 {self.bifreq_k}")
        if int(self.bifreq_k) > 0 and self.w_forcenorm <= 0.0:
            raise ValueError("bifreq_k > 0 时 w_forcenorm 必须 > 0（双频需要一个幅值头）")
        if float(self.label_clip_mg) < 0.0:
            raise ValueError(f"label_clip_mg 必须 >= 0，实际 {self.label_clip_mg}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class TactileGRU(nn.Module):
    """GRU + 四个基础头（力 / 接触 / 面积 / CoP）+ 可选消费端头。"""

    def __init__(self, n_features: int, cfg: EstimatorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_features = int(n_features)
        self.arch = cfg.arch
        if cfg.arch == "shared" and str(cfg.task_caps or "").strip():
            raise ValueError(
                f"arch=shared 不接受 task_caps={cfg.task_caps!r}（会被静默忽略）；"
                "解耦请用 arch=indep，共享容量请用 --layers/--hidden")
        if cfg.arch == "indep":
            dead = [name for name, w in (("w_forcenorm", cfg.w_forcenorm),
                                         ("w_delta", cfg.w_delta),
                                         ("w_peak", cfg.w_peak),
                                         ("w_lookahead", cfg.w_lookahead),
                                         ("w_fbar_rel", cfg.w_fbar_rel))
                    if float(w) > 0.0]
            if dead:
                raise ValueError(
                    "arch=indep 的 forward 只产出四个基础任务，消费端头不会被训练："
                    + f"{dead} 权重必须为 0（否则「只改架构」不成立、且头是死参数）")
            # 每任务独立 GRU（Wave B 解耦臂）：任务间没有任何共享参数。
            caps = arch_spec("indep", cfg.task_caps)
            self.caps = caps
            self.gru = None
            self.grus = nn.ModuleDict()
            for task in TASKS:
                layers, hidden = caps[task]
                self.grus[task] = nn.GRU(
                    input_size=self.n_features,
                    hidden_size=int(hidden),
                    num_layers=int(layers),
                    batch_first=True,
                    dropout=cfg.dropout if layers > 1 else 0.0,
                )
                setattr(self, f"{task}_head",
                        nn.Linear(int(hidden), HEAD_DIMS[task]))
            h = max(hidden for _layers, hidden in caps.values())
        else:
            caps = None
            self.caps = None
            self.gru = nn.GRU(
                input_size=self.n_features,
                hidden_size=cfg.hidden,
                num_layers=cfg.layers,
                batch_first=True,
                dropout=cfg.dropout if cfg.layers > 1 else 0.0,
            )
            h = cfg.hidden
            self.force_head = nn.Linear(h, 12)     # 4 足 × 3 维
            self.contact_head = nn.Linear(h, 4)    # 4 足 logits
            self.area_head = nn.Linear(h, 4)
            self.cop_head = nn.Linear(h, 8)
        # 可选消费端头：权重为 0 时是 None，不注册任何参数，
        # 因此历史 checkpoint 的 state_dict 逐键不变、可直接 load。
        if cfg.w_forcenorm > 0.0 and int(cfg.head_hidden) > 0:
            # v4 解耦臂：F̄ 头加一层隐层（GELU）提升容量；参数名仍是
            # forcenorm_head.* 前缀 ⇒ freeze_except_forcenorm 与「只允许头部缺键」
            # 的热启动规则不变。
            self.forcenorm_head: nn.Module | None = nn.Sequential(
                nn.Linear(h, int(cfg.head_hidden)), nn.GELU(),
                nn.Linear(int(cfg.head_hidden), 4))
        else:
            self.forcenorm_head = (nn.Linear(h, 4) if cfg.w_forcenorm > 0.0
                                   else None)
        self.delta_head = nn.Linear(h, 4) if cfg.w_delta > 0.0 else None
        self.peak_head = nn.Linear(h, 4) if cfg.w_peak > 0.0 else None
        k = int(cfg.lookahead_contact)
        self.lookahead_head = (nn.Linear(h, 4 * k)
                               if (k > 0 and cfg.w_lookahead > 0.0) else None)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """``x``: (B, T, F) ⇒ 各头输出（B, ·）；四个基础键永远存在。"""
        if self.arch == "indep":
            out = {}
            for task in TASKS:
                _o, h_n = self.grus[task](x)
                out[task] = getattr(self, f"{task}_head")(h_n[-1])
            return out
        _out, h_n = self.gru(x)
        feat = h_n[-1]
        out = {
            "force": self.force_head(feat),
            "contact": self.contact_head(feat),
            "area": self.area_head(feat),
            "cop": self.cop_head(feat),
        }
        if self.forcenorm_head is not None:
            out["forcenorm"] = self.forcenorm_head(feat)
        if self.delta_head is not None:
            out["delta"] = self.delta_head(feat)
        if self.peak_head is not None:
            out["peak"] = self.peak_head(feat)
        if self.lookahead_head is not None:
            out["contact_next"] = self.lookahead_head(feat)
        return out


# --------------------------------------------------------------------------- #
# 数据：窗口化与读写
# --------------------------------------------------------------------------- #


def window_starts(n_steps: int, window: int, stride: int) -> np.ndarray:
    """长度为 ``n_steps`` 的序列里所有窗口**终点**的下标。

    按 ``stride`` 从 ``window-1`` 开始取，并**保证包含最后一帧**（不足一个 stride 的
    尾巴由最后再加一个终点覆盖，避免回合末尾样本被系统性丢弃）。
    """
    if n_steps < window:
        return np.zeros(0, dtype=np.int64)
    last = list(range(int(window) - 1, int(n_steps), max(1, int(stride))))
    if last[-1] != n_steps - 1:
        last.append(n_steps - 1)
    return np.asarray(last, dtype=np.int64)


def gather_windows(features: np.ndarray, ends: np.ndarray,
                   window: int) -> np.ndarray:
    """``features``: (N, F) ⇒ (n_win, window, F)，每个窗口以 ``ends`` 为最后一帧。"""
    idx = ends[:, None] + np.arange(-window + 1, 1)[None, :]
    return features[idx]


def slice_features(windows: dict[str, np.ndarray],
                   n_features: int) -> dict[str, np.ndarray]:
    """只取特征前 ``n_features`` 列（v3 把 12 维力矩列追加在末尾）。

    ``n_features <= 0`` 或等于当前宽度时原样返回；这样旧数据集（53 列）与
    新数据集（65 列）可以用同一份代码、靠 ``--features 53`` 对齐口径。"""
    n = int(n_features)
    if "x" not in windows:
        raise ValueError("windows 缺少 'x' 字段")
    width = int(windows["x"].shape[-1])
    if n <= 0 or n == width:
        return windows
    if n > width:
        raise ValueError(f"n_features={n} 超过特征宽度 {width}")
    out = dict(windows)
    out["x"] = windows["x"][..., :n]    # 视图：训练侧只读，避免复制 ~GB 级窗口
    return out


def save_dataset(path: str | Path, *, features: np.ndarray, force_world: np.ndarray,
                 force_foot: np.ndarray, contact: np.ndarray,
                 area_norm: np.ndarray, cop_norm: np.ndarray,
                 phase: np.ndarray, episode: np.ndarray,
                 meta: dict[str, Any],
                 force_norm: np.ndarray | None = None,
                 delta_force_norm: np.ndarray | None = None,
                 impact_peak: np.ndarray | None = None,
                 torque: np.ndarray | None = None,
                 extra: dict[str, np.ndarray] | None = None) -> Path:
    """写 npz + 同名 json。后三个是可选**消费端标签**（F̄ / ΔF̄ / 落地峰值），
    旧数据集没有这些字段时训练侧用世界系力的等价代理（见 ``consumer_label_arrays``）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "features": np.asarray(features, dtype=np.float32),
        "force_world": np.asarray(force_world, dtype=np.float32),
        "force_foot": np.asarray(force_foot, dtype=np.float32),
        "contact": np.asarray(contact, dtype=np.uint8),
        "area_norm": np.asarray(area_norm, dtype=np.float32),
        "cop_norm": np.asarray(cop_norm, dtype=np.float32),
        "phase": np.asarray(phase, dtype=np.int8),
        "episode": np.asarray(episode, dtype=np.int32),
    }
    for key, value in (("force_norm", force_norm),
                       ("delta_force_norm", delta_force_norm),
                       ("impact_peak", impact_peak),
                       # 原始 12 维执行器力矩（未归一化），仅用于核对特征列/重标定
                       ("torque", torque)):
        if value is not None:
            arrays[key] = np.asarray(value, dtype=np.float32)
    # 可选的外部数组（v5 的 D1 特权输入 priv_*）：不参与窗口化/训练，只给诊断脚本读。
    # 白名单 + 长度契约：拼错名字（priv_clearence）必须当场报错，否则诊断脚本
    # 要到读数据时才 KeyError，报错指向错误位置。
    n_steps = int(np.asarray(features).shape[0])
    for key, value in (extra or {}).items():
        if key in arrays:
            raise ValueError(f"extra 与已有字段重名: {key}")
        if not str(key).startswith("priv_"):
            raise ValueError(
                f"extra 只接受 priv_* 诊断字段（白名单），实际 {key!r}；"
                "新增落盘字段请显式改 save_dataset 签名")
        arr = np.asarray(value)
        if arr.shape[0] != n_steps:
            raise ValueError(
                f"extra 字段 {key} 首维应为 {n_steps}，实际 {arr.shape[0]}")
        arrays[key] = arr
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_dataset(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with np.load(path) as z:
        out = {k: z[k] for k in z.files}
    meta_path = path.with_suffix(".json")
    out["meta"] = (json.loads(meta_path.read_text(encoding="utf-8"))
                   if meta_path.is_file() else {})
    return out


def consumer_label_arrays(data: dict[str, Any], cfg: EstimatorConfig
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """取消费端标签 `(F̄, ΔF̄, 落地峰值, source)`，缺字段时用世界系力代理。

    `TactileSensor` 的真标签（`force_norm` / `delta_force_norm` /
    `impact_peak`）已按体重归一化，由 `scripts/collect_tactile_data.py` 写进
    npz；旧数据集没有这些字段，这里退化成 `|F|/mg`、回合内单步差、5 帧因果
    滚动最大值（与「法向力/合力」「落地窗峰值」语义接近但不相等），并把 `source`
    记成 `proxy`，避免把代理指标当成真标签报告。
    """
    fw = np.asarray(data["force_world"], dtype=np.float32)
    ep = np.asarray(data["episode"], dtype=np.int64)
    ref = max(float(cfg.force_ref_n), 1e-6)
    if "force_norm" in data:
        fn = np.asarray(data["force_norm"], dtype=np.float32)
        source = "sensor"
    else:
        fn = (np.linalg.norm(fw, axis=2) / ref).astype(np.float32)
        source = "proxy"
    if "delta_force_norm" in data:
        dl = np.asarray(data["delta_force_norm"], dtype=np.float32)
    else:
        dl = np.zeros_like(fn)
        for epi in np.unique(ep):
            idx = np.flatnonzero(ep == epi)
            if idx.size == 0:
                continue
            dl[idx[0]] = fn[idx[0]]        # 与 tactile.py 一致：首步上一帧力记 0
            if idx.size > 1:
                dl[idx[1:]] = np.diff(fn[idx], axis=0)
    if "impact_peak" in data:
        pk = np.asarray(data["impact_peak"], dtype=np.float32)
    else:
        pk = np.zeros_like(fn)
        for epi in np.unique(ep):
            idx = np.flatnonzero(ep == epi)
            for j, t in enumerate(idx):
                pk[t] = fn[idx[max(0, j - 4):j + 1]].max(axis=0)
    return fn, dl, pk, source


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #


def causal_mean_labels(values: np.ndarray, k: int,
                       episode: np.ndarray) -> np.ndarray:
    """把逐帧标签换成「以当前帧结尾的因果 k 帧均值」（逐回合，不跨回合）。

    v7 Wave C 的双频输出：幅值通道用低频目标（k 帧因果均值）换时序容差，
    接触通道保持 50 Hz 瞬时。k<=1 时逐位返回原值（历史行为不变）。
    """
    v = np.asarray(values, dtype=np.float32)
    ep = np.asarray(episode)
    kk = int(k)
    if kk <= 1:
        return v.copy()
    out = np.empty_like(v)
    for e in np.unique(ep):
        idx = np.flatnonzero(ep == e)
        seg = v[idx].astype(np.float64)
        cs = np.cumsum(seg, axis=0)
        for t in range(seg.shape[0]):
            lo = max(0, t - kk + 1)
            s = cs[t] - (cs[lo - 1] if lo > 0 else 0.0)
            out[idx[t]] = (s / float(t - lo + 1)).astype(np.float32)
    return out


def debounce_contact_labels(contact: np.ndarray, episode: np.ndarray,
                            window: int = 3, votes: int = 2) -> np.ndarray:
    """训练侧接触标签的因果去抖：最近 window 帧里 >=votes 帧接触才判接触。

    只在训练标签上用（评估仍按原始仿真接触算 F1），回合开头不足 window 帧时
    用已有帧投票；votes<=1 时逐位返回原值。
    """
    c = np.asarray(contact, dtype=np.float32)
    ep = np.asarray(episode)
    if int(votes) <= 1:
        return c.copy()
    out = np.zeros_like(c)
    w, need = int(window), int(votes)
    for e in np.unique(ep):
        idx = np.flatnonzero(ep == e)
        seg = (c[idx] > 0.5).astype(np.float32)
        run = np.zeros(seg.shape[1], dtype=np.float32)
        for t in range(seg.shape[0]):
            run += seg[t]
            if t >= w:
                run -= seg[t - w]
            out[idx[t]] = (run >= need).astype(np.float32)
    return out


def force_metrics(pred: np.ndarray, target: np.ndarray,
                  force_ref_n: float) -> dict[str, float]:
    """三维力 RMSE/MAE（N 与 %体重）。``pred/target``: (N, 4, 3)。"""
    err = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    out: dict[str, float] = {}
    for axis, name in enumerate(("x", "y", "z")):
        e = err[:, :, axis]
        out[f"rmse_{name}_n"] = float(np.sqrt(np.mean(e ** 2)))
        out[f"mae_{name}_n"] = float(np.mean(np.abs(e)))
        out[f"rmse_{name}_bw"] = out[f"rmse_{name}_n"] / force_ref_n
        out[f"mae_{name}_bw"] = out[f"mae_{name}_n"] / force_ref_n
        out[f"p95_{name}_n"] = float(np.percentile(np.abs(e), 95))
        t = np.asarray(target, dtype=np.float64)[:, :, axis]
        out[f"r2_{name}"] = 1.0 - float(np.sum(e ** 2)) / max(
            1e-12, float(np.sum((t - np.mean(t)) ** 2)))
    mag_pred = np.linalg.norm(np.asarray(pred, dtype=np.float64), axis=2)
    mag_true = np.linalg.norm(np.asarray(target, dtype=np.float64), axis=2)
    mag_err = mag_pred - mag_true
    out["rmse_mag_n"] = float(np.sqrt(np.mean(mag_err ** 2)))
    out["mae_mag_n"] = float(np.mean(np.abs(mag_err)))
    out["rmse_mag_bw"] = out["rmse_mag_n"] / force_ref_n
    out["mae_mag_bw"] = out["mae_mag_n"] / force_ref_n
    out["rmse_all_n"] = float(np.sqrt(np.mean(err ** 2)))
    out["mae_all_n"] = float(np.mean(np.abs(err)))
    out["rmse_all_bw"] = out["rmse_all_n"] / force_ref_n
    out["mae_all_bw"] = out["mae_all_n"] / force_ref_n
    out["p95_all_n"] = float(np.percentile(np.abs(err), 95))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((np.asarray(target, dtype=np.float64) - np.mean(target)) ** 2))
    out["r2_all"] = 1.0 - ss_res / max(1e-12, ss_tot)
    return out


def contact_metrics(prob: np.ndarray, target: np.ndarray,
                    threshold: float = 0.5) -> dict[str, float]:
    """接触检测（4 足并成一列的二分类指标）。"""
    p = (np.asarray(prob) >= threshold).astype(np.int64).reshape(-1)
    y = (np.asarray(target) > 0.5).astype(np.int64).reshape(-1)
    tp = int(np.sum((p == 1) & (y == 1)))
    tn = int(np.sum((p == 0) & (y == 0)))
    fp = int(np.sum((p == 1) & (y == 0)))
    fn = int(np.sum((p == 0) & (y == 1)))
    total = max(1, p.size)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {
        "accuracy": (tp + tn) / total,
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / max(1e-9, prec + rec),
        "tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn),
        "pos_rate": float(y.mean()),
    }


def regression_metrics(pred: np.ndarray, target: np.ndarray,
                       mask: np.ndarray | None = None) -> dict[str, float]:
    """面积/CoP 的 RMSE/MAE（``mask`` 只用接触帧）。"""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.any():
            pred = pred[m]
            target = target[m]
    err = pred - target
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": 1.0 - ss_res / max(1e-12, ss_tot),
        "n": float(err.size),
    }

def stance_force_metrics(pred: np.ndarray, target: np.ndarray,
                         contact: np.ndarray, *,
                         force_ref_n: float,
                         threshold: float = 0.5,
                         phase: np.ndarray | None = None,
                         phase_min: float = 2.5) -> dict[str, float]:
    """**消费端口径**：支撑相（phase >= STANCE）接触足上的 F̄ 幅度质量。

    为什么还要这个：`force_metrics` 的 R² 大部分来自「摆动相 f≈0」的接触判别，
    支撑相的幅度会向均值收缩（斜率 < 1、峰值低估）。这里只统计接触足，给出
    R² / 回归斜率 / 均值比 / p95 比，直接暴露「幅度是否被低估」。

    `phase` 给出时再叠加 `phase > phase_min`（默认 2.5 ⇒ 只留 STANCE），
    把落地冲击帧（LANDING）排除；**标定拟合与评估必须用同一套掩码**，
    否则系数会被作用到拟合时没见过的分布上（评审 B3）。
    """
    m = np.asarray(contact) > threshold
    if phase is not None:
        ph = np.asarray(phase)
        if ph.shape != m.shape:
            raise ValueError(f"phase 形状 {ph.shape} 与 contact {m.shape} 不一致")
        m = m & (ph > float(phase_min))
    p = np.asarray(pred, dtype=np.float64)[m]
    t = np.asarray(target, dtype=np.float64)[m]
    ref = max(float(force_ref_n), 1e-6)
    if p.size == 0:
        return {"n": 0.0, "rmse_norm": float("nan"), "mae_norm": float("nan"),
                "rmse_n": float("nan"), "rmse_bw": float("nan"),
                "mae_n": float("nan"), "r2": float("nan"), "slope": float("nan"),
                "mean_pred_n": float("nan"), "mean_target_n": float("nan"),
                "mean_ratio": float("nan"), "p95_ratio": float("nan")}
    err = p - t
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    cov = float(np.sum((t - t.mean()) * (p - p.mean())))
    p95_p = float(np.percentile(p, 95))
    p95_t = float(np.percentile(t, 95))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    return {
        "n": float(p.size),
        # rmse_norm/mae_norm 是归一化（单位 mg）误差；rmse_n/mae_n 已换算成牛顿。
        # 旧版把归一化误差直接命名 rmse_n 并印成 N，读表时会差约 150 倍。
        "rmse_norm": rmse,
        "mae_norm": mae,
        "rmse_n": rmse * ref,
        # target 已按体重归一化 ⇒ rmse 本身就是「体重倍数」，不要再除一次 ref
        "rmse_bw": rmse,
        "mae_n": mae * ref,
        "r2": 1.0 - ss_res / max(1e-12, ss_tot),
        "slope": cov / ss_tot if ss_tot > 1e-12 else float("nan"),
        "mean_pred_n": float(p.mean() * ref),
        "mean_target_n": float(t.mean() * ref),
        "mean_ratio": float(p.mean() / max(t.mean(), 1e-9)),
        "p95_ratio": p95_p / max(p95_t, 1e-9),
    }


def per_foot_constant_baseline(force_world: np.ndarray) -> dict[str, float]:
    """「每足每轴常数」弱基线（只用本数据集均值），用来判断 GRU 到底赢多少。

    弱基线的 RMSE 在留出楼梯上是 36.9–51.8 N，而 GRU 是 27.9–29.8 N ⇒ 报告的
    R² 0.56–0.81 里很大一部分是「摆动相≈0」被学会，不是支撑相幅度。
    """
    t = np.asarray(force_world, dtype=np.float64)
    const = np.broadcast_to(t.mean(axis=0, keepdims=True), t.shape)
    m = force_metrics(const, t, 1.0)
    return {"rmse_all_n": m["rmse_all_n"], "r2_all": m["r2_all"],
            "r2_x": m["r2_x"], "r2_y": m["r2_y"], "r2_z": m["r2_z"]}


def contact_latency(prob: np.ndarray, target: np.ndarray, *,
                    threshold: float = 0.5) -> dict[str, float]:
    """接触**起始帧**的检出时延（输入必须是一段连续序列，按回合切片调用）。

    对每个「上一帧无接触、本帧有接触」的起始点记 `p_onset`，并取前后各一帧的
    概率。真机闭环里最关心的是「触地那一帧能不能报出来」，平均 F1 看不出来。
    """
    p = np.asarray(prob, dtype=np.float64)
    y = np.asarray(target) > 0.5
    if y.ndim != 2:
        raise ValueError("contact_latency 需要 (T, 4) 的连续序列")
    prev = np.vstack([np.zeros((1, y.shape[1]), dtype=bool), y[:-1]])
    onsets = np.argwhere(y & ~prev)
    if onsets.size == 0:
        return {"n_onsets": 0.0, "onset_recall": float("nan"),
                "onset_prob_mean": float("nan"),
                "prev_prob_mean": float("nan"), "prev_recall": float("nan"),
                "next_prob_mean": float("nan"), "next_recall": float("nan")}
    on = np.array([p[i, j] for i, j in onsets])
    pv = np.array([p[i - 1, j] if i > 0 else 0.0 for i, j in onsets])
    nx = np.array([p[i + 1, j] if i + 1 < p.shape[0] else 0.0 for i, j in onsets])
    return {
        "n_onsets": float(onsets.shape[0]),
        "onset_recall": float(np.mean(on >= threshold)),
        "onset_prob_mean": float(on.mean()),
        "prev_prob_mean": float(pv.mean()),
        "prev_recall": float(np.mean(pv >= threshold)),
        "next_prob_mean": float(nx.mean()),
        "next_recall": float(np.mean(nx >= threshold)),
    }


def physics_penalty(sum_z_norm: torch.Tensor, mask: torch.Tensor | None = None,
                    delta: float = 0.25) -> torch.Tensor:
    """体重平衡：**批次平均**的四足竖直力之和 ≈ 体重（`sum_z_norm` 已按体重归一）。

    实测（train_flat 11200 帧）逐帧 Σ_f F_z 的均值是 222.5 N（比 mg 高 49%）、
    p05–p95 = 44–515 N，但**全部帧的时间平均**是 151.0 N ≈ mg = 149.17 N（8 个数据集
    0.966–1.044 BW）。逐帧约束会把触地冲击帧（单帧可达 2 kN）压平、与标签矛盾，
    所以这里只在**时间/批次平均**上施加约束，用来对抗整体幅值收缩
    （v1 预测幅值只有真值的 0.80）。
    """
    if mask is None:
        per = sum_z_norm.mean()
    else:
        m = mask.float()
        per = (sum_z_norm * m).sum() / m.sum().clamp(min=1.0)
    resid = per - 1.0
    return torch.nn.functional.huber_loss(
        resid, torch.zeros_like(resid), delta=float(delta))


def relative_magnitude_penalty(pred_mag: torch.Tensor, target_mag: torch.Tensor,
                               mask: torch.Tensor,
                               eps: float = 0.01) -> torch.Tensor:
    """相对幅值误差 |pred−target|²/(target²+eps)，只在接触足上平均。

    纯 MSE 会奖励「向均值收缩」（把 0.8 BW 的支撑力猜成 0.6 BW），相对误差对
    大力的欠拟合更敏感，是压住「峰值低估 15–25%」的直接手段。
    """
    err = (pred_mag - target_mag) ** 2 / (target_mag ** 2 + float(eps))
    m = mask.float()
    return (err * m).sum() / m.sum().clamp(min=1.0)


# --------------------------------------------------------------------------- #
# 数据集窗口化 / 标准化 / 划分
# --------------------------------------------------------------------------- #


def make_dataset_windows(data: dict[str, Any], cfg: EstimatorConfig,
                        stride: int) -> dict[str, np.ndarray]:
    """把一个地形数据集切成窗口样本（按回合拼出连续序列）。"""
    features = np.asarray(data["features"], dtype=np.float32)
    episode = np.asarray(data["episode"], dtype=np.int64)
    force_world = np.asarray(data["force_world"], dtype=np.float32)
    contact = np.asarray(data["contact"], dtype=np.float32)
    area = np.asarray(data["area_norm"], dtype=np.float32)
    cop = np.asarray(data["cop_norm"], dtype=np.float32)
    phase = (np.asarray(data["phase"], dtype=np.float32)
             if "phase" in data else np.zeros(contact.shape, dtype=np.float32))
    fn, dl, pk, _src = consumer_label_arrays(data, cfg)
    clip = float(getattr(cfg, "label_clip_mg", 0.0) or 0.0)
    if clip > 0.0:
        # 训练侧限幅：只影响损失/头目标，评估仍用原始标签（_src 不变）
        lim = clip * float(cfg.force_ref_n)
        force_world = np.clip(force_world, -lim, lim)
        fn = np.clip(fn, 0.0, clip)
        dl = np.clip(dl, -clip, clip)
        pk = np.clip(pk, 0.0, clip)
    k = int(cfg.lookahead_contact)
    # v7 Wave C：训练侧接触去抖（标签层）与幅值因果均值目标（双频低频通道）
    contact_train = (debounce_contact_labels(contact, episode)
                     if cfg.debounce_contact else contact)
    forcenorm_mean = causal_mean_labels(fn, cfg.bifreq_k, episode)
    xs, yf, yc, ya, yp, eps = [], [], [], [], [], []
    yfn, ydl, ypk, yph, ycn = [], [], [], [], []
    yct, yfm = [], []
    n_total = 0
    for ep in np.unique(episode):
        idx = np.flatnonzero(episode == ep)
        if idx.size < cfg.window:
            continue
        ends = idx[window_starts(idx.size, cfg.window, stride)]
        xs.append(gather_windows(features, ends, cfg.window))
        yf.append(force_world[ends])
        yc.append(contact[ends])
        ya.append(area[ends])
        yp.append(cop[ends])
        yfn.append(fn[ends])
        yfm.append(forcenorm_mean[ends])
        yct.append(contact_train[ends])
        ydl.append(dl[ends])
        ypk.append(pk[ends])
        yph.append(phase[ends])
        eps.append(np.full(ends.shape, int(ep), dtype=np.int64))
        if k > 0:
            # 接触前瞻：标签取「本帧之后第 j 帧」是否接触（回合末尾越界取 0）
            nxt = np.zeros((ends.size, 4 * k), dtype=np.float32)
            for j in range(1, k + 1):
                pos = ends + j
                valid = pos <= idx[-1]
                block = np.zeros((ends.size, 4), dtype=np.float32)
                if valid.any():
                    block[valid] = contact[pos[valid]]
                nxt[:, 4 * (j - 1):4 * j] = block
            ycn.append(nxt)
        n_total += int(ends.size)
    if not xs:
        raise ValueError("数据集里没有任何长度 >= window 的回合")
    return {
        "x": np.concatenate(xs, axis=0),
        "force": np.concatenate(yf, axis=0),
        "contact": np.concatenate(yc, axis=0),
        "area": np.concatenate(ya, axis=0),
        "cop": np.concatenate(yp, axis=0),
        # 消费端标签（F̄ / ΔF̄ / 落地峰值 / 足相位）——训练侧只在对应权重非 0 时用
        "forcenorm": np.concatenate(yfn, axis=0),
        # 训练侧接触标签（默认等于 contact；debounce_contact 时是去抖后的）
        "contact_train": np.concatenate(yct, axis=0),
        # 双频幅值目标（bifreq_k<=1 时逐位等于 forcenorm）
        "forcenorm_mean": np.concatenate(yfm, axis=0),
        "delta": np.concatenate(ydl, axis=0),
        "peak": np.concatenate(ypk, axis=0),
        "phase": np.concatenate(yph, axis=0),
        "contact_next": (np.concatenate(ycn, axis=0) if ycn
                         else np.zeros((n_total, 0), dtype=np.float32)),
        "episode": np.concatenate(eps, axis=0),
    }


def merge_windows(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}


def feature_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐特征均值/标准差（只从训练集算；标准差下限 1e-6）。"""
    flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def split_episodes(episode: np.ndarray, val_frac: float,
                   seed: int) -> tuple[np.ndarray, np.ndarray]:
    """按**回合**划分 train/val，返回两个布尔掩码（同一回合只出现在一侧）。"""
    uniq = np.unique(episode)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(uniq)
    n_val = int(round(len(uniq) * val_frac))
    if val_frac > 0.0:
        n_val = max(1, n_val)
    val_eps = set(int(e) for e in uniq[:n_val])
    val_mask = np.array([int(e) in val_eps for e in episode], dtype=bool)
    return ~val_mask, val_mask


# --------------------------------------------------------------------------- #
# 训练 / 存档 / 推理
# --------------------------------------------------------------------------- #


def _opt_target(d: dict[str, np.ndarray] | None, key: str,
                shape: tuple[int, ...]) -> torch.Tensor:
    """取可选目标张量；缺字段时给全 0（对应权重为 0 时不影响 loss）。"""
    v = None if d is None else d.get(key)
    if v is None:
        return torch.zeros(shape, dtype=torch.float32)
    arr = np.asarray(v, dtype=np.float32)
    if arr.size != int(np.prod(shape)):
        raise ValueError(f"{key} 形状 {arr.shape} 与期望 {shape} 不一致")
    return torch.from_numpy(arr.reshape(shape))


def multi_task_loss(out: dict[str, torch.Tensor],
                    targets: dict[str, torch.Tensor],
                    cfg: EstimatorConfig) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """多任务损失（训练与验证**同口径**）。返回 `(total, terms)`。

    `targets["force"]` 必须已经按 `force_ref_n` 归一；`forcenorm/delta/peak/
    contact_next` 缺失时全 0（且对应权重为 0，不影响结果）。
    """
    bce = nn.BCEWithLogitsLoss()
    yf = targets["force"]
    # v7 Wave C：训练侧接触标签可去抖（评估仍按原始 contact 算 F1）
    yc = targets.get("contact_train", targets["contact"])
    # 双频幅值头：bifreq_k>0 时 F̄ 头目标换成因果 k 帧均值
    fn_target = (targets["forcenorm_mean"]
                 if (cfg.bifreq_k > 0 and "forcenorm_mean" in targets)
                 else targets.get("forcenorm"))
    mask_c = (yc > 0.5).float()
    denom = mask_c.sum().clamp(min=1.0)
    pred_f = out["force"].reshape(-1, 4, 3)
    # v4 支撑相加权：楼梯上 78% 的接触帧是 LANDING/SWING 抖动，稳态支撑帧只占
    # 22%，不加权时损失被冲击帧主导 ⇒ 幅值被往均值拉（斜率收缩的直接来源）。
    # w_stance=0 时 frame_w==mask_c、force 走原分支 ⇒ 与历史逐位等价。
    phase = targets.get("phase")
    if phase is None:
        stance = torch.zeros_like(mask_c)
    else:
        stance = (phase > 2.5).float()
    w_stance = float(getattr(cfg, "w_stance", 0.0))
    frame_w = mask_c * (1.0 + w_stance * stance) if w_stance > 0.0 else mask_c
    terms: dict[str, torch.Tensor] = {}
    if w_stance > 0.0:
        w_force = 1.0 + w_stance * stance     # 全帧按 1 计权，支撑相再加权
        terms["force"] = (((pred_f - yf) ** 2).mean(dim=2) * w_force).sum() / (
            w_force.sum().clamp(min=1.0))
    else:
        terms["force"] = torch.mean((pred_f - yf) ** 2)
    terms["contact"] = bce(out["contact"], yc)
    # 面积头输出 4 维（每足一个），只在接触足上算均方误差
    terms["area"] = (((out["area"] - targets["area"]) ** 2) * mask_c).sum() / denom
    terms["cop"] = (torch.mean((out["cop"].reshape(-1, 4, 2) - targets["cop"]) ** 2,
                               dim=2) * mask_c).sum() / denom
    total = (cfg.w_force * terms["force"] + cfg.w_contact * terms["contact"]
             + cfg.w_area * terms["area"] + cfg.w_cop * terms["cop"])
    for key, weight in (("forcenorm", cfg.w_forcenorm), ("delta", cfg.w_delta),
                        ("peak", cfg.w_peak)):
        if key in out and weight > 0.0:
            # 只在接触足上算：这三项在 71% 的「无接触」帧上恒为 0，全帧 MSE 会把
            # 预测往 0 拉（均值收缩的直接来源），消费端只看接触/支撑帧。
            tgt = targets[key]
            if key == "forcenorm" and fn_target is not None:
                tgt = fn_target
            terms[key] = (((out[key] - tgt) ** 2) * frame_w).sum() / (
                frame_w.sum().clamp(min=1.0))
            total = total + weight * terms[key]
    if cfg.w_fbar_rel > 0.0 and "forcenorm" in out and fn_target is not None:
        # 与消费端 slope/mean_ratio 同量：接触足上 pred/target 的幅值比损失
        terms["fbar_rel"] = relative_magnitude_penalty(
            out["forcenorm"], fn_target, frame_w)
        total = total + cfg.w_fbar_rel * terms["fbar_rel"]
    if cfg.w_magrel > 0.0:
        terms["magrel"] = relative_magnitude_penalty(
            torch.linalg.vector_norm(pred_f, dim=2),
            torch.linalg.vector_norm(yf, dim=2), mask_c)
        total = total + cfg.w_magrel * terms["magrel"]
    if cfg.physics_weight > 0.0:
        # 时间/批次平均口径：逐帧 Σz 与 mg 的偏差是真实的（冲击帧可达 +49%），
        # 只有「平均 = 体重」成立，见 physics_penalty 的 docstring。
        terms["physics"] = physics_penalty(pred_f[:, :, 2].sum(dim=1))
        total = total + cfg.physics_weight * terms["physics"]
    if "contact_next" in out and cfg.w_lookahead > 0.0:
        if targets["contact_next"].numel() > 0:
            terms["lookahead"] = bce(out["contact_next"], targets["contact_next"])
            total = total + cfg.w_lookahead * terms["lookahead"]
    return total, terms


class TactileEstimator:
    """训练好的估计器 + 输入标准化参数（保存/加载的最小单元）。"""

    def __init__(self, model: TactileGRU, cfg: EstimatorConfig,
                 feature_mean: np.ndarray, feature_std: np.ndarray) -> None:
        self.model = model
        self.cfg = cfg
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)

    @property
    def n_features(self) -> int:
        return int(self.feature_mean.size)

    def predict(self, features_window: np.ndarray) -> dict[str, np.ndarray]:
        """``features_window``: (B, T, F) 原始观测 ⇒ 各头 numpy 输出。

        ``force`` 单位 N（世界系，足维 4×3），``contact`` 是概率，
        ``area`` 是 Ā，``cop`` 是归一化 CoP。
        """
        x = (np.asarray(features_window, dtype=np.float32)
             - self.feature_mean) / self.feature_std
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.from_numpy(x))
        res = {
            "force": out["force"].numpy().reshape(-1, 4, 3) * self.cfg.force_ref_n,
            "contact": torch.sigmoid(out["contact"]).numpy(),
            "area": out["area"].numpy(),
            "cop": out["cop"].numpy().reshape(-1, 4, 2),
        }
        if "forcenorm" in out:      # F̄（法向力/体重，已是归一化量，不再乘 ref）
            res["forcenorm"] = out["forcenorm"].numpy()
        if "delta" in out:
            res["delta"] = out["delta"].numpy()
        if "peak" in out:
            res["peak"] = out["peak"].numpy()
        if "contact_next" in out:   # 前瞻接触概率（每足 lookahead_contact 帧）
            res["contact_next"] = torch.sigmoid(out["contact_next"]).numpy()
        return res

    def save(self, path: str | Path, extra_meta: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "config": self.cfg.to_dict(),
            "state_dict": self.model.state_dict(),
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "n_features": self.n_features,
            "meta": dict(extra_meta or {}),
        }, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TactileEstimator":
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = EstimatorConfig(**blob["config"])
        model = TactileGRU(int(blob["n_features"]), cfg)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return cls(model, cfg, blob["feature_mean"], blob["feature_std"])


def init_model_from_checkpoint(model: nn.Module, path: str | Path
                                ) -> dict[str, list[str]]:
    """把旧 checkpoint 的权重灌进新模型（``strict=False``，返回缺失/多余键）。

    典型用法：拿没有 F̄ 头的 c0 权重热启动带 F̄ 头的模型 —— 主干与四个基础头
    全部命中，只有 ``forcenorm_head.*`` 是缺失键（随机初始化）。要求
    ``n_features`` 一致，避免把 53 列权重悄悄灌进 65 列模型。"""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    n_in = int(blob.get("n_features", -1))
    if n_in != int(getattr(model, "n_features", -1)):
        raise ValueError(f"checkpoint n_features={n_in} 与模型 "
                         f"{getattr(model, 'n_features', -1)} 不一致")
    missing, unexpected = model.load_state_dict(blob["state_dict"], strict=False)
    return {"missing": list(missing), "unexpected": list(unexpected),
            "feature_mean": blob.get("feature_mean"),
            "feature_std": blob.get("feature_std")}


def freeze_except_forcenorm(model: nn.Module,
                            keep_prefixes: tuple[str, ...] = ()) -> list[str]:
    """冻结主干与非解码头，只让 forcenorm_head（及 keep_prefixes）参与训练。

    keep_prefixes 用于「冻结主干 + 只训前瞻接触头」这类单头实验。
    返回可训练参数名列表；没有任何可训参数时返回空列表（调用方视为配置错误）。
    """
    prefixes = ("forcenorm_head",) + tuple(p for p in keep_prefixes if p)
    trainable: list[str] = []
    for name, param in model.named_parameters():
        keep = any(name.startswith(p) for p in prefixes)
        param.requires_grad = bool(keep)
        if keep:
            trainable.append(name)
    return trainable


def apply_affine(y: np.ndarray, a: float, b: float) -> np.ndarray:
    """标定：``y' = a*y + b``（后验仿射，只改报告口径，不改权重）。"""
    return (np.asarray(y, dtype=np.float64) * float(a) + float(b)).astype(np.float32)


def _calib_slope(p: np.ndarray, t: np.ndarray) -> float:
    """标定后的回归斜率 cov(p,t)/var(t)（与 stance_force_metrics 同口径）。"""
    if p.size < 2:
        return float("nan")
    ss = float(np.sum((t - t.mean()) ** 2))
    if ss < 1e-12:
        return float("nan")
    return float(np.sum((t - t.mean()) * (p - p.mean())) / ss)


def fit_affine(y: np.ndarray, t: np.ndarray,
               method: str = "ls",
               a_min: float | None = None,
               a_max: float | None = None) -> dict[str, float]:
    """拟合后验仿射 ``y' = a*y + b``，返回 a/a_raw/b/矩统计/rmse/slope_after。

    两种方向，回答两个不同的问题（评审 B2 指出混用会让 P3 的斜率判据恒不可达）：

    - ``method="ls"``：最小二乘 ``t ≈ a*y + b``（RMSE 最优）。它是**收缩型**标定：
      标定后 cov(p,t)/var(t) = r²(y,t) < 1，所以「斜率→1」不可能由它实现。
    - ``method="inverse"``：反向回归 ``y ≈ α*t + β`` 再反演，a = var(t)/cov(y,t)。
      这是**衰减偏差修正**：标定后斜率恒为 1，但 RMSE 可能变差。

    ``a_min``/``a_max``（v4 限幅）：只对 ``a`` 做限幅（斜率限幅），``b`` 重新取
    ``mean(t) - a*mean(y)`` 以保持均值——v3 实测「inverse 把斜率推上去 R² 崩」，
    限幅就是「斜率换 R²」的折中旋钮；原始 a 保留在 ``a_raw`` 供审计与扫描。
    同时把 mean/var/cov 一起返回：评估脚本可以只改 a 重算 b，不必重新拟合
    （b = mean_t - a*mean_y 与这里完全同式）。

    少于 2 个样本或退化（方差/协方差≈0）时返回 ``a=1, b=mean(t)-mean(y)``。"""
    if method not in ("ls", "inverse"):
        raise ValueError(f"method 必须是 ls 或 inverse，收到 {method!r}")
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    tv = np.asarray(t, dtype=np.float64).reshape(-1)
    if yv.size != tv.size:
        raise ValueError(f"y/t 样本数不一致：{yv.size} vs {tv.size}")
    ok = np.isfinite(yv) & np.isfinite(tv)
    yv, tv = yv[ok], tv[ok]
    mean_y = float(np.mean(yv)) if yv.size else float("nan")
    mean_t = float(np.mean(tv)) if tv.size else float("nan")
    a, b = 1.0, (mean_t - mean_y if yv.size else 0.0)
    a_raw = a
    if yv.size >= 2 and float(np.std(yv)) > 1e-9:
        if method == "ls":
            a, b = (float(v) for v in np.polyfit(yv, tv, 1))
        else:
            cov0 = float(np.mean((yv - yv.mean()) * (tv - tv.mean())))
            if abs(cov0) > 1e-12:
                a = float(np.var(tv) / cov0)
                b = float(tv.mean() - a * yv.mean())
        a_raw = a
    clipped = False
    if a_min is not None and a < float(a_min):
        a, clipped = float(a_min), True
    if a_max is not None and a > float(a_max):
        a, clipped = float(a_max), True
    if clipped and yv.size:
        b = mean_t - a * mean_y
    p = a * yv + b if yv.size else yv
    before = float(np.sqrt(np.mean((yv - tv) ** 2))) if yv.size else float("nan")
    after = float(np.sqrt(np.mean((p - tv) ** 2))) if yv.size else float("nan")
    return {"a": a, "b": b, "a_raw": a_raw, "clipped": bool(clipped),
            "n": int(yv.size), "method": method,
            "mean_y": mean_y, "mean_t": mean_t,
            "var_y": float(np.var(yv)) if yv.size else 0.0,
            "var_t": float(np.var(tv)) if yv.size else 0.0,
            "cov": (float(np.mean((yv - mean_y) * (tv - mean_t)))
                    if yv.size else 0.0),
            "rmse_before": before, "rmse_after": after,
            "slope_after": _calib_slope(p, tv) if yv.size else float("nan")}


def window_embedding(x: np.ndarray) -> np.ndarray:
    """把 (N, window, F) 窗口折成聚类用的定长嵌入 [mean(F), std(F)]。

    v6 聚类条件标定用它把窗口分成地形/姿态档位：只用策略可见特征
    （不含标签、不含地形高度），因此推理时可复现，不是用真值分箱。
    """
    xa = np.asarray(x, dtype=np.float64)
    if xa.ndim != 3:
        raise ValueError(f"window_embedding 期望 (N, window, F)，收到 {xa.shape}")
    return np.concatenate([xa.mean(axis=1), xa.std(axis=1)], axis=1)


def fit_standardizer(x: np.ndarray) -> dict[str, np.ndarray]:
    """逐列 z-score 的 mean/std（std 过小则置 1，避免除零）。"""
    xa = np.asarray(x, dtype=np.float64)
    mean = xa.mean(axis=0)
    std = xa.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean, "std": std}


def apply_standardizer(x: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    """按 fit_standardizer 的统计量标准化（宽度不一致直接报错）。"""
    xa = np.asarray(x, dtype=np.float64)
    if xa.shape[-1] != np.asarray(scaler["mean"]).size:
        raise ValueError(
            f"嵌入宽度 {xa.shape[-1]} 与 scaler {np.asarray(scaler['mean']).size} 不一致")
    return (xa - np.asarray(scaler["mean"], dtype=np.float64)) / np.asarray(
        scaler["std"], dtype=np.float64)


def fit_kmeans(x: np.ndarray, k: int, *, seed: int = 0, iters: int = 50
               ) -> dict[str, Any]:
    """确定性 k-means（k-means++ 初始化，空簇用最远点重播）。

    返回 {centers, counts, inertia, n_iter, k, seed}。k<=1 时退化为单中心
    （即全局标定）；k>样本数直接报错（聚类不能静默截断）。
    """
    xa = np.asarray(x, dtype=np.float64)
    n = int(xa.shape[0])
    kk = int(k)
    if kk < 1:
        raise ValueError(f"k 必须 >= 1，收到 {k}")
    if kk > n:
        raise ValueError(f"k={kk} 大于样本数 {n}：聚类不能静默截断")
    rng = np.random.default_rng(int(seed))
    centers = np.empty((kk, xa.shape[1]), dtype=np.float64)
    centers[0] = xa[int(rng.integers(n))]
    closest = np.sum((xa - centers[0]) ** 2, axis=1)
    for c in range(1, kk):
        tot = float(closest.sum())
        if tot <= 0.0:
            centers[c] = xa[int(rng.integers(n))]
        else:
            centers[c] = xa[int(rng.choice(n, p=closest / tot))]
        closest = np.minimum(closest, np.sum((xa - centers[c]) ** 2, axis=1))
    labels = np.zeros(n, dtype=np.int64)
    it = 0
    for it in range(int(iters)):
        d2 = ((xa[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(d2, axis=1)
        changed = int(np.sum(new_labels != labels))
        labels = new_labels
        for c in range(kk):
            m = labels == c
            if not m.any():
                far = int(np.argmax(d2[np.arange(n), labels]))
                centers[c] = xa[far]
                labels[far] = c
                continue
            centers[c] = xa[m].mean(axis=0)
        if changed == 0 and it > 0:
            break
    d2 = ((xa[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(d2, axis=1)
    counts = [int(np.sum(labels == c)) for c in range(kk)]
    return {"centers": centers, "counts": counts,
            "inertia": float(np.sum(d2[np.arange(n), labels])),
            "n_iter": int(it + 1), "k": kk, "seed": int(seed)}


def assign_clusters(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """按最近中心分配簇（欧氏距离；输入必须已用同一 scaler 标准化）。"""
    xa = np.asarray(x, dtype=np.float64)
    cs = np.asarray(centers, dtype=np.float64)
    if cs.ndim != 2 or xa.shape[-1] != cs.shape[1]:
        raise ValueError(f"centers {cs.shape} 与嵌入 {xa.shape} 宽度不一致")
    d2 = ((xa[:, None, :] - cs[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1)


def apply_clustered_affine(y: np.ndarray, labels: np.ndarray,
                           fits: list[dict[str, float]],
                           n_clusters: int) -> np.ndarray:
    """按窗口簇标签施加 per-cluster 仿射（y 形状 (N,4)，labels 形状 (N,)）。"""
    yv = np.asarray(y, dtype=np.float64)
    lbl = np.asarray(labels, dtype=np.int64).reshape(-1)
    if yv.shape[0] != lbl.size:
        raise ValueError(f"y 首维 {yv.shape[0]} 与 labels {lbl.size} 不一致")
    if len(fits) != int(n_clusters):
        raise ValueError(f"fits 数量 {len(fits)} 与 n_clusters {n_clusters} 不一致")
    out = np.empty_like(yv)
    for c, fit in enumerate(fits):
        m = lbl == c
        if not m.any():
            continue
        out[m] = yv[m] * float(fit["a"]) + float(fit["b"])
    return out.astype(np.float32)


def train_estimator(train: dict[str, np.ndarray],
                    val: dict[str, np.ndarray] | None,
                    cfg: EstimatorConfig,
                    *,
                    device: str = "cpu",
                    verbose: bool = True,
                    init_from: str | Path | None = None,
                    freeze_trunk: bool = False
                    ) -> tuple[TactileEstimator, list[dict[str, float]]]:
    """多任务监督训练。返回 ``(estimator, history)``；history 每 epoch 一项。

    基础四项（力 / 接触 / 面积 / CoP）与历史一致；``EstimatorConfig`` 里消费端
    权重非 0 时由 ``multi_task_loss`` 追加 F̄ / ΔF̄ / 峰值 / 相对幅值 / 物理一致性 /
    前瞻接触各项。全 0 时新建的头为 None ⇒ 参数量与历史完全相同。
    """
    val_empty = val is None or int(np.asarray(val["x"]).shape[0]) == 0
    if val_empty and val is not None:
        raise ValueError(
            "val 集为空：无法做 best-val 选择/早停，也不能静默保存最后一轮"
            "（检查 val_frac 与回合数）")
    if val_empty:
        print("  警告：val=None（仅测试路径），将保存最后一轮并写 val_empty=True")
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    mean, std = feature_stats(train["x"])
    n_features = int(train["x"].shape[-1])
    model = TactileGRU(n_features, cfg).to(device)
    init_info: dict | None = None
    if init_from is not None:
        raw = init_model_from_checkpoint(model, init_from)
        # 只允许「新增的消费端头」缺失；主干/基础头/其它头缺失或形状不符说明
        # window/hidden/layers 与 checkpoint 不同，那时 strict=False 会静默随机初始化，
        # 得到的权重不能再说「来自 init_from」（评审 B5）。
        missing = list(raw["missing"])
        unexpected = list(raw["unexpected"])
        bad = [k for k in missing if not k.startswith("forcenorm_head")
               and not k.startswith("delta_head") and not k.startswith("peak_head")
               and not k.startswith("lookahead_head")]
        if bad or unexpected:
            raise ValueError(
                f"init_from 与当前模型不兼容：missing={bad} unexpected={unexpected}"
                f"（只有消费端新头允许缺失；请核对 window/hidden/layers/features）")
        init_info = {"path": str(init_from), "missing": missing,
                     "unexpected": unexpected,
                     "n_features": int(n_features),
                     # 冻结/热启动时的输入标准化必须能核对：把 checkpoint 的统计量
                     # 一起带出来（否则 freeze 只覆盖权重，换数据集会静默偏移）。
                     "feature_mean": (None if raw.get("feature_mean") is None
                                      else np.asarray(raw["feature_mean"]).tolist()),
                     "feature_std": (None if raw.get("feature_std") is None
                                     else np.asarray(raw["feature_std"]).tolist())}
        if verbose:
            print(f"  热启动 {init_from}: missing={missing} unexpected={unexpected}")
    if freeze_trunk:
        if cfg.w_forcenorm <= 0.0 and cfg.w_lookahead <= 0.0:
            raise ValueError("freeze_trunk 需要 cfg.w_forcenorm > 0 或 "
                             "cfg.w_lookahead > 0（否则没有可训的头）")
        keep_extra = ("lookahead_head",) if cfg.w_lookahead > 0.0 else ()
        trainable = freeze_except_forcenorm(model, keep_prefixes=keep_extra)
        if not trainable:
            raise ValueError("freeze_trunk 但模型没有可训的解码头")
        if init_info is None:
            init_info = {"path": None, "missing": [], "unexpected": [],
                         "n_features": int(n_features)}
        init_info["frozen_trainable"] = trainable
        if verbose:
            print(f"  冻结主干：只训练 {trainable}")
    if freeze_trunk and init_info is not None and init_info.get("feature_mean") is not None:  # noqa: E501
        # 冻结主干的输入标准化必须与它训练时一致，否则换数据集会静默偏移（评审 R8）。
        cm = np.asarray(init_info["feature_mean"], dtype=np.float32)
        cs = np.asarray(init_info["feature_std"], dtype=np.float32)
        if cm.shape != np.asarray(mean).shape:
            raise ValueError(f"checkpoint feature_mean 形状 {cm.shape} 与当前特征数不符")
        if not np.allclose(cm, mean, atol=1e-4) or not np.allclose(cs, std, atol=1e-4):
            print("  [warn] 冻结主干：沿用 checkpoint 的 feature_mean/std（与当前数据统计不同）")
        mean, std = cm, cs
        init_info["feature_stats_source"] = "checkpoint"
    x_train = ((train["x"] - mean) / std).astype(np.float32)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=cfg.lr, weight_decay=cfg.weight_decay)
    ref = float(cfg.force_ref_n)
    n = int(x_train.shape[0])
    k = int(cfg.lookahead_contact)

    tensors = {
        "x": torch.from_numpy(x_train),
        "force": torch.from_numpy(train["force"].astype(np.float32) / ref),
        "contact": torch.from_numpy(train["contact"].astype(np.float32)),
        "area": torch.from_numpy(train["area"].astype(np.float32)),
        "cop": torch.from_numpy(train["cop"].astype(np.float32)),
        "forcenorm": _opt_target(train, "forcenorm", (n, 4)),
        "delta": _opt_target(train, "delta", (n, 4)),
        "peak": _opt_target(train, "peak", (n, 4)),
        "phase": _opt_target(train, "phase", (n, 4)),   # v4 支撑相加权用
        "contact_next": _opt_target(train, "contact_next", (n, 4 * k)),
        # v7 Wave C：训练侧接触标签与双频幅值目标
        "contact_train": _opt_target(train, "contact_train", (n, 4)),
        "forcenorm_mean": _opt_target(train, "forcenorm_mean", (n, 4)),
    }
    targets_tr = {key: value for key, value in tensors.items() if key != "x"}
    gen = torch.Generator().manual_seed(int(cfg.seed))
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = 0
    stopped_early = False
    for epoch in range(int(cfg.epochs)):
        epoch_lr = float(opt.param_groups[0]["lr"])
        order = torch.randperm(n, generator=gen)
        model.train()
        tot = 0.0
        for start in range(0, n, cfg.batch):
            sel = order[start:start + cfg.batch]
            batch = {key: value[sel] for key, value in targets_tr.items()}
            loss, _terms = multi_task_loss(model(tensors["x"][sel]), batch, cfg)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.item()) * int(sel.numel())
        row = {"epoch": float(epoch + 1), "train_loss": tot / max(1, n)}
        if val is not None and val["x"].shape[0] > 0:
            xv_np = ((val["x"] - mean) / std).astype(np.float32)
            nv = int(xv_np.shape[0])
            vt = {
                "force": torch.from_numpy(val["force"].astype(np.float32) / ref),
                "contact": torch.from_numpy(val["contact"].astype(np.float32)),
                "area": torch.from_numpy(val["area"].astype(np.float32)),
                "cop": torch.from_numpy(val["cop"].astype(np.float32)),
                "forcenorm": _opt_target(val, "forcenorm", (nv, 4)),
                "delta": _opt_target(val, "delta", (nv, 4)),
                "peak": _opt_target(val, "peak", (nv, 4)),
                "phase": _opt_target(val, "phase", (nv, 4)),
                "contact_next": _opt_target(val, "contact_next", (nv, 4 * k)),
            }
            model.eval()
            with torch.no_grad():
                vloss, _vterms = multi_task_loss(model(torch.from_numpy(xv_np)), vt, cfg)
            row["val_loss"] = float(vloss)
        row["lr"] = epoch_lr
        history.append(row)
        # best-val 追踪：早停/最终保存都以 val 最优轮为准（不再无条件存最后一轮）
        cur = row.get("val_loss")
        if cur is not None and not np.isfinite(cur):
            # NaN/Inf 的 val_loss 会让 best_state=None、best_epoch=0，最后静默保存
            # 最后一轮权重并且看起来「训练成功」（评审 C 的假绿点）⇒ 直接失败。
            raise ValueError(f"第 {epoch + 1} 轮 val_loss={cur} 非有限，拒绝继续训练")
        improved = cur is not None and cur < best_val - 1e-12
        if improved:
            best_val = float(cur)
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (epoch % 5 == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch + 1:>3}/{cfg.epochs}  train {row['train_loss']:.4f}"
                  + (f"  val {row['val_loss']:.4f}" if "val_loss" in row else "")
                  + f"  lr {epoch_lr:.2e}")
        if cfg.lr_gamma != 1.0:
            for group in opt.param_groups:
                group["lr"] = float(group["lr"]) * float(cfg.lr_gamma)
        if (cfg.early_stop > 0 and cur is not None
                and epoch + 1 - best_epoch >= int(cfg.early_stop)):
            stopped_early = True
            if verbose:
                print(f"  早停：第 {epoch + 1} 轮，最优 val 在 {best_epoch}"
                      f"（窗口 --early-stop={cfg.early_stop}）")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    est = TactileEstimator(model, cfg, mean, std)
    est.init_info = init_info      # 供训练脚本写进 meta（不参与 save/load）
    est.best_epoch = int(best_epoch)
    est.best_val_loss = None if best_state is None else float(best_val)
    est.stopped_early = bool(stopped_early)
    est.epochs_run = len(history)
    est.val_empty = bool(val_empty)
    return est, history

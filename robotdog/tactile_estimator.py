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

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError(f"window 必须 >= 2，实际 {self.window}")
        if self.batch <= 0 or self.epochs <= 0:
            raise ValueError("batch / epochs 必须为正")
        if not (0.0 <= self.val_frac < 0.5):
            raise ValueError(f"val_frac 必须在 [0, 0.5)，实际 {self.val_frac}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class TactileGRU(nn.Module):
    """GRU + 四个头（力 / 接触 / 面积 / CoP）。"""

    def __init__(self, n_features: int, cfg: EstimatorConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_features = int(n_features)
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """``x``: (B, T, F) ⇒ 各头输出（B, ·）。"""
        _out, h_n = self.gru(x)
        feat = h_n[-1]
        return {
            "force": self.force_head(feat),
            "contact": self.contact_head(feat),
            "area": self.area_head(feat),
            "cop": self.cop_head(feat),
        }


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


def save_dataset(path: str | Path, *, features: np.ndarray, force_world: np.ndarray,
                 force_foot: np.ndarray, contact: np.ndarray,
                 area_norm: np.ndarray, cop_norm: np.ndarray,
                 phase: np.ndarray, episode: np.ndarray,
                 meta: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.asarray(features, dtype=np.float32),
        force_world=np.asarray(force_world, dtype=np.float32),
        force_foot=np.asarray(force_foot, dtype=np.float32),
        contact=np.asarray(contact, dtype=np.uint8),
        area_norm=np.asarray(area_norm, dtype=np.float32),
        cop_norm=np.asarray(cop_norm, dtype=np.float32),
        phase=np.asarray(phase, dtype=np.int8),
        episode=np.asarray(episode, dtype=np.int32),
    )
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


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #


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
    xs, yf, yc, ya, yp, eps = [], [], [], [], [], []
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
        eps.append(np.full(ends.shape, int(ep), dtype=np.int64))
    if not xs:
        raise ValueError("数据集里没有任何长度 >= window 的回合")
    return {
        "x": np.concatenate(xs, axis=0),
        "force": np.concatenate(yf, axis=0),
        "contact": np.concatenate(yc, axis=0),
        "area": np.concatenate(ya, axis=0),
        "cop": np.concatenate(yp, axis=0),
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


def train_estimator(train: dict[str, np.ndarray],
                    val: dict[str, np.ndarray] | None,
                    cfg: EstimatorConfig,
                    *,
                    device: str = "cpu",
                    verbose: bool = True) -> tuple[TactileEstimator, list[dict[str, float]]]:
    """多任务监督训练。返回 ``(estimator, history)``；history 每 epoch 一项。"""
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    mean, std = feature_stats(train["x"])
    x_train = ((train["x"] - mean) / std).astype(np.float32)
    n_features = int(x_train.shape[-1])
    model = TactileGRU(n_features, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    bce = nn.BCEWithLogitsLoss()
    ref = float(cfg.force_ref_n)

    tensors = {
        "x": torch.from_numpy(x_train),
        "force": torch.from_numpy(train["force"].astype(np.float32) / ref),
        "contact": torch.from_numpy(train["contact"].astype(np.float32)),
        "area": torch.from_numpy(train["area"].astype(np.float32)),
        "cop": torch.from_numpy(train["cop"].astype(np.float32)),
    }
    n = int(tensors["x"].shape[0])
    gen = torch.Generator().manual_seed(int(cfg.seed))
    history: list[dict[str, float]] = []
    for epoch in range(int(cfg.epochs)):
        order = torch.randperm(n, generator=gen)
        model.train()
        tot = 0.0
        for start in range(0, n, cfg.batch):
            sel = order[start:start + cfg.batch]
            xb = tensors["x"][sel]
            yf = tensors["force"][sel]
            yc = tensors["contact"][sel]
            ya = tensors["area"][sel]
            yp = tensors["cop"][sel]
            out = model(xb)
            loss_f = torch.mean((out["force"].reshape(-1, 4, 3) - yf) ** 2)
            loss_c = bce(out["contact"], yc)
            mask_c = (yc > 0.5).float()
            denom = mask_c.sum().clamp(min=1.0)
            # 面积头输出 4 维（每足一个），只在接触足上算均方误差
            loss_a = (((out["area"] - ya) ** 2) * mask_c).sum() / denom
            loss_p = (torch.mean((out["cop"].reshape(-1, 4, 2) - yp) ** 2, dim=2)
                      * mask_c).sum() / denom
            loss = (cfg.w_force * loss_f + cfg.w_contact * loss_c
                    + cfg.w_area * loss_a + cfg.w_cop * loss_p)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss.item()) * int(sel.numel())
        row = {"epoch": float(epoch + 1), "train_loss": tot / max(1, n)}
        if val is not None and val["x"].shape[0] > 0:
            xv_np = ((val["x"] - mean) / std).astype(np.float32)
            model.eval()
            with torch.no_grad():
                xv = torch.from_numpy(xv_np)
                ov = model(xv)
                vf = torch.mean((ov["force"].reshape(-1, 4, 3)
                                 - torch.from_numpy(val["force"].astype(np.float32) / ref)) ** 2)
                vc = bce(ov["contact"], torch.from_numpy(val["contact"].astype(np.float32)))
                vm = (val["contact"] > 0.5).astype(np.float32)
                vm_t = torch.from_numpy(vm)
                vdenom = vm_t.sum().clamp(min=1.0)
                # 与训练 loss 同口径：面积/CoP 只在接触足上算（否则 val_loss 会被
                # 大量「无接触、标签恒 0」的样本稀释，曲线不可比）
                va = float((((ov["area"] - torch.from_numpy(val["area"])) ** 2)
                            * vm_t).sum() / vdenom)
                vp = float((((ov["cop"].reshape(-1, 4, 2)
                              - torch.from_numpy(val["cop"])) ** 2)
                            .mean(dim=2) * vm_t).sum() / vdenom)
            row["val_loss"] = float(cfg.w_force * float(vf)
                                    + cfg.w_contact * float(vc)
                                    + cfg.w_area * float(va)
                                    + cfg.w_cop * float(vp))
        history.append(row)
        if verbose and (epoch % 5 == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch + 1:>3}/{cfg.epochs}  train {row['train_loss']:.4f}"
                  + (f"  val {row['val_loss']:.4f}" if "val_loss" in row else ""))
    return TactileEstimator(model, cfg, mean, std), history

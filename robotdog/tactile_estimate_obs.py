"""隐式触觉估计块（Level2「隐式」臂的 critic 特权观测）。

它是什么：robotdog/tactile_estimator.py 里的 GRU 估计器（离线用足端触觉真值训练）
在**在线**时的前向。输入 = 策略可见的 actor 帧（53 维）+ 12 维执行器力矩特征
（与 scripts/collect_tactile_data.py 的采集口径逐位一致），输出 8 维：

    4 足 |F| / mg    —— force 头的力矢量（世界系 N）取模长，再除以 force_ref_n
    4 足 Ā           —— area 头输出（已归一化接触面积，训练标签就是它）

不含接触状态（0/1）、不含 CoP —— 用户对 Level2 隐式臂的要求只有「合力 + 接触面积」。
与显式臂（ObsConfig.privileged_foot_contact 4 维 + privileged_tactile 16 维）
相比，这里少掉的正是这两类信息，文档里必须写清楚。

为什么不 import torch：训练时 8 个 SubprocVecEnv worker 各持一份估计器，每个
worker 都 import torch 会把数 GB 的常驻内存再乘 8（本机 16.8 GB，见 experience.md
的 OOM 记录）。所以权重先由 scripts/export_tactile_estimator.py 导出成 .npz，
这里用 numpy 手写 3 层 GRU 前向（逐位对齐 TactileEstimator.predict，见
tests/test_tactile_estimate_obs.py 的对拍）。

窗口语义：估计器训练时的输入是连续 window 个控制步（默认 20）。这里每个控制步
都按滑窗**从零重算**（与 predict 的截断窗口完全一致），不做增量隐状态递推 ——
增量递推的「记忆」长度不再等于窗口，对拍会漂、且越走越偏离训练分布。

成本：3 层 × 128 隐层 × 20 步的滑窗重算在 numpy 上是硬性的 ~60 次 sgemv。
实测 float32 + 「layer0 整窗 GEMM + 每层一次 sgemv」约 0.9 ms/控制步
（.tmp/level2/bench_variants.py），相对 env 单步 ~5.6 ms 是 +16% CPU；
L3 冒烟专门核对 SPS 掉幅，超过 15% 就退化到 stride=2（25 Hz 更新）并在
trace 里记一条 key_change。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: 与 robotdog/tactile_estimator.py 的 DEFAULT_FORCE_REF_N 同值（Go2 体重）。
DEFAULT_FORCE_REF_N = 149.17

#: 本模块在每帧末尾追加的维数（4 足合力 + 4 足面积）。
EST_BLOCK_SIZE = 8

#: 力矩归一化尺度，顺序 FL/FR/RL/RR × hip/thigh/calf（镜像 go2.xml ctrlrange）。
TORQUE_REF_NM: tuple[float, ...] = tuple(
    ref for _ in range(4) for ref in (23.7, 23.7, 45.43))

#: force 头的维数（4 足 × 3 轴）与 area 头的维数（4 足）。
FORCE_DIM = 12
AREA_DIM = 4


@dataclass
class EstimateObsConfig:
    """隐式触觉块（Level2「隐式」臂）的开关与超参。

    enabled=False 时 EnvConfig 里连估计器都不构造，训练热路径零额外开销；
    打开后由 ObsConfig.privileged_tactile_est 决定这块进不进入每帧末尾。
    """

    enabled: bool = False
    #: scripts/export_tactile_estimator.py 导出的 .npz（不要用 .pt：8 个 worker
    #: 各 import 一次 torch 会把常驻内存乘 8）。
    path: str = ""
    #: 每 stride 个控制步更新一次（1 = 50 Hz，与训练口径一致；2 = 25 Hz 省一半
    #: 的滑窗重算成本，代价是块最多滞后 1 步）。
    stride: int = 1
    #: force 头的归一化尺度 [N]（Go2 体重）；必须与训练 checkpoint 里的一致。
    force_ref_n: float = DEFAULT_FORCE_REF_N

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.path = str(self.path or "")
        self.stride = int(self.stride)
        if self.enabled and not self.path:
            raise ValueError(
                "env.estimator.enabled=true 时必须给 env.estimator.path（.npz 权重）"
            )
        if self.stride < 1:
            raise ValueError(f"env.estimator.stride 必须 >= 1，实际 {self.stride}")
        if float(self.force_ref_n) <= 0.0:
            raise ValueError(
                f"env.estimator.force_ref_n 必须 > 0，实际 {self.force_ref_n}"
            )


def torque_features(tau: np.ndarray) -> np.ndarray:
    """12 维执行器力矩 ⇒ 归一化特征。

    与 robotdog/tactile_estimator.torque_features **逐位一致**（那里的测试断言了
    两者相等）；复制一份是为了不 import torch（见模块 docstring）。
    """
    t = np.asarray(tau, dtype=np.float32).reshape(-1)
    if t.size != len(TORQUE_REF_NM):
        raise ValueError(f"力矩维数应为 {len(TORQUE_REF_NM)}，实际 {t.size}")
    return (t / np.asarray(TORQUE_REF_NM, dtype=np.float32)).astype(np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _scalar(blob: dict, key: str):
    return np.asarray(blob[key]).item()


class TactileEstimateObserver:
    """滑窗 GRU 前向：actor 帧 + 执行器力矩 ⇒ 8 维隐式触觉块。"""

    def __init__(self, blob: dict, *, source: str = "", dtype=np.float32,
                 stride: int = 1) -> None:
        self.source = source
        self.dtype = np.dtype(dtype)
        self.n_features = int(_scalar(blob, "n_features"))
        self.hidden = int(_scalar(blob, "hidden"))
        self.layers = int(_scalar(blob, "layers"))
        self.window = int(_scalar(blob, "window"))
        self.force_ref_n = float(_scalar(blob, "force_ref_n"))
        self.feature_mean = np.asarray(blob["feature_mean"], dtype=self.dtype)
        std = np.asarray(blob["feature_std"], dtype=self.dtype)
        # 采集数据里某列可能恒为常数（std=0）—— 训练侧同样会把 0 夹成 1，否则
        # 标准化除出 inf，整个块变 NaN（NaN 进 critic 就是静默中毒）。
        self.feature_std = np.where(np.abs(std) < 1e-8, 1.0, std).astype(self.dtype)
        self.n_torque = len(TORQUE_REF_NM)
        self.n_actor = self.n_features - self.n_torque
        if self.n_actor <= 0:
            raise ValueError(
                f"估计器特征数 {self.n_features} 小于力矩维数 {self.n_torque}；"
                "这个 checkpoint 不是「观测 + 力矩」口径，不能用于隐式块。"
            )
        if self.feature_mean.size != self.n_features:
            raise ValueError(
                f"feature_mean 长度 {self.feature_mean.size} != n_features "
                f"{self.n_features}（{source}）"
            )
        # 只取 force / area 两个头：隐式块 = 合力 + 面积，其它头（接触概率/CoP）
        # 是显式臂的信息，这里用不上；同时把「缺头」变成构造期报错。
        for head, dim in (("force_head", FORCE_DIM), ("area_head", AREA_DIM)):
            w_key, b_key = head + ".weight", head + ".bias"
            if w_key not in blob or b_key not in blob:
                raise ValueError(
                    f"估计器缺少 {head} 头（{source}）—— 隐式块必须同时有 force 与 "
                    "area 两个头，否则只能给显式臂用。"
                )
            w = np.asarray(blob[w_key], dtype=self.dtype)
            if w.shape != (dim, self.hidden):
                raise ValueError(
                    f"{head}.weight 形状 {w.shape} != ({dim}, {self.hidden})（{source}）"
                )
        self._w_ih = [np.asarray(blob[f"gru.weight_ih_l{i}"], dtype=self.dtype)
                      for i in range(self.layers)]
        self._w_hh = [np.asarray(blob[f"gru.weight_hh_l{i}"], dtype=self.dtype)
                      for i in range(self.layers)]
        # 两个偏置不能简单相加！PyTorch GRU 的更新式是
        #   n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn))
        # b_in 在 r 外面、b_hn 在 r 里面。融合成 b_ih+b_hh 只在 h=0 且 bias=0 时
        # 才等价 —— 实测会给出 ~0.1 的块级偏差（.tmp/level2/dbg_formula.py 对齐）。
        self._b_ih = [np.asarray(blob[f"gru.bias_ih_l{i}"], dtype=self.dtype)
                      for i in range(self.layers)]
        self._b_hh = [np.asarray(blob[f"gru.bias_hh_l{i}"], dtype=self.dtype)
                      for i in range(self.layers)]
        for i in range(self.layers):
            H = self.hidden
            in_dim = self.n_features if i == 0 else H
            if self._w_ih[i].shape != (3 * H, in_dim):
                raise ValueError(
                    f"gru.weight_ih_l{i} 形状 {self._w_ih[i].shape} 与 "
                    f"(3×{H}, {in_dim}) 不符（{source}）"
                )
            if self._w_hh[i].shape != (3 * H, H):
                raise ValueError(f"gru.weight_hh_l{i} 形状不符（{source}）")
            if self._b_ih[i].shape != (3 * H,) or self._b_hh[i].shape != (3 * H,):
                raise ValueError(f"gru.bias_l{i} 形状不符（{source}）")
        self._force_w = np.asarray(blob["force_head.weight"], dtype=self.dtype)
        self._force_b = np.asarray(blob["force_head.bias"], dtype=self.dtype)
        self._area_w = np.asarray(blob["area_head.weight"], dtype=self.dtype)
        self._area_b = np.asarray(blob["area_head.bias"], dtype=self.dtype)

        self._buf = np.zeros((self.window, self.n_features), dtype=self.dtype)
        self._ord = np.arange(self.window)
        self._pos = 0
        self._block = np.zeros(EST_BLOCK_SIZE, dtype=np.float64)
        self.stride = max(1, int(stride))
        self._tick = 0

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_npz(cls, path: str | Path, *, n_actor: int | None = None,
                 stride: int = 1) -> "TactileEstimateObserver":
        """从 scripts/export_tactile_estimator.py 导出的 .npz 构造。"""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"找不到隐式触觉估计器权重 {path}；先用 "
                "scripts/export_tactile_estimator.py 从 .pt 导出 .npz。"
            )
        with np.load(path, allow_pickle=False) as z:
            blob = {k: z[k] for k in z.files}
        obs = cls(blob, source=str(path), stride=stride)
        if n_actor is not None and obs.n_actor != int(n_actor):
            raise ValueError(
                f"估计器的观测维数 {obs.n_actor} 与当前 actor 帧维数 {int(n_actor)} "
                f"不一致（{path}）—— 估计器是用另一套 obs 布局训练的，隐式块会喂错列。"
            )
        return obs

    # ------------------------------------------------------------------ 前向
    def _feature(self, actor_frame: np.ndarray, tau: np.ndarray) -> np.ndarray:
        a = np.asarray(actor_frame, dtype=self.dtype).reshape(-1)
        if a.size != self.n_actor:
            raise ValueError(
                f"隐式估计器的输入应为 {self.n_actor} 维 actor 帧，实际 {a.size} 维"
            )
        return np.concatenate([a, torque_features(tau).astype(self.dtype)])

    def _decode(self, feat: np.ndarray) -> np.ndarray:
        force = (self._force_w @ feat + self._force_b).reshape(4, 3) * self.force_ref_n
        mag = np.linalg.norm(force.astype(np.float64), axis=1) / self.force_ref_n
        area = self._area_w @ feat + self._area_b
        return np.concatenate([mag, area.astype(np.float64)]).astype(np.float64)

    def _run(self, x: np.ndarray) -> np.ndarray:
        """x: (T, n_features) 已标准化 ⇒ GRU ⇒ 8 维块。

        性能敏感路径（每个控制步一次，8 个 worker 各一份）：第 0 层「来自输入的
        门控项」对整窗一次 GEMM（gate0），之后按时间步递推，每层每步只有一次
        sgemv（W_hh @ h）。门序与 PyTorch 一致：权重行块是 [r, z, n]，
        更新式 n = tanh(W_in x + b_in + r ⊙ (W_hh h + b_hh))、h' = (1-z)⊙n + z⊙h。
        """
        dtype = self.dtype
        H = self.hidden
        layers = self.layers
        w_ih, w_hh = self._w_ih, self._w_hh
        b_ih, b_hh = self._b_ih, self._b_hh
        sig = _sigmoid
        tanh = np.tanh
        h = np.zeros((layers, H), dtype=dtype)
        gate0 = np.dot(x, w_ih[0].T) + b_ih[0]      # (T, 3H)
        for t in range(x.shape[0]):
            gi = gate0[t]
            gh = np.dot(w_hh[0], h[0]) + b_hh[0]
            r = sig(gi[:H] + gh[:H])
            z = sig(gi[H:2 * H] + gh[H:2 * H])
            n = tanh(gi[2 * H:] + r * gh[2 * H:])
            h[0] = (1.0 - z) * n + z * h[0]
            for l in range(1, layers):
                gi = np.dot(w_ih[l], h[l - 1]) + b_ih[l]
                gh = np.dot(w_hh[l], h[l]) + b_hh[l]
                r = sig(gi[:H] + gh[:H])
                z = sig(gi[H:2 * H] + gh[H:2 * H])
                n = tanh(gi[2 * H:] + r * gh[2 * H:])
                h[l] = (1.0 - z) * n + z * h[l]
        return self._decode(h[layers - 1])

    def _standardize(self, w: np.ndarray) -> np.ndarray:
        return (w - self.feature_mean) / self.feature_std

    # ------------------------------------------------------------------ 接口
    def reset(self, actor_frame: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """回合开始：用首帧**重复填满**窗口，再算第一块。

        填零是不行的：标准化后全 0 输入对应「远低于训练均值」的恒定输入，GRU 会
        给出系统性偏置的初始估计；用首帧重复更像「站着不动的稳态」。

        ⚠ 已知偏差（登记，不改代码）：离线训练窗口永远是 20 帧**真实**历史
        （``tactile_estimator.py`` 的滑窗切法），所以 episode 前 ``window-1`` 帧
        （默认 19 步 ≈ 0.38 s）的输入是分布外的常数窗。实测在 l2_i0 + tactile 开启、
        平地零动作 200 步下，pad 期（1–19）F̄ corr 0.784 / MAE 0.180 并不差于
        纯真实窗（20–200）的 0.692 / 0.1725 ⇒ 影响可忽略，只在报告里注明。
        """
        self._buf[:] = self._feature(actor_frame, tau)
        self._pos = 0
        self._tick = 0
        self._block = self._run(self._standardize(self._buf[self._ord]))
        return self._block.copy()

    def update(self, actor_frame: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """推进一步，返回本步的 8 维块（float64；帧本身是 float64）。

        ``stride > 1`` 时只在每 stride 步真正重算，其余步复用上一块 —— 缓冲照旧
        每步写入，所以恢复计算那一步的窗口仍然连续（不是跳采，只是「读数」稀疏）。
        """
        self._buf[self._pos] = self._feature(actor_frame, tau)
        self._pos = (self._pos + 1) % self.window
        self._tick += 1
        if self.stride > 1 and (self._tick % self.stride) != 0:
            return self._block.copy()
        idx = (self._ord + self._pos) % self.window      # 最旧 → 最新
        self._block = self._run(self._standardize(self._buf[idx]))
        return self._block.copy()

    def predict_window(self, window: np.ndarray) -> np.ndarray:
        """对完整窗口 (T, n_features) 直接前向（对拍用；不改内部状态）。

        与 update 的区别：T 任意（可小于 window，不会复用历史帧），返回 (8,)。
        测试拿它与 TactileEstimator.predict 对比。
        """
        w = np.asarray(window, dtype=self.dtype)
        if w.ndim != 2 or w.shape[1] != self.n_features:
            raise ValueError(f"窗口形状应为 (T, {self.n_features})，实际 {w.shape}")
        if w.shape[0] == 0:
            raise ValueError("窗口不能为空")
        return self._run(self._standardize(w))
r"""scripts/frames.py 的切片契约测试（unittest，不依赖 pytest）。

跑法::

    .venv\Scripts\python.exe -m unittest tests.test_frames -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import frames  # noqa: E402


def write_video(path: Path, *, frame_count: int = 100, fps: float = 25.0) -> None:
    """写一段小尺寸、颜色随时间变化的测试视频（够读出帧号即可）。"""
    import imageio

    width, height = 64, 36
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                            macro_block_size=1) as writer:
        for i in range(frame_count):
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:, :, i % 3] = int(20 + 220 * i / max(1, frame_count - 1))
            writer.append_data(img)


class CoarseIndexTest(unittest.TestCase):
    def test_keeps_head_and_tail(self) -> None:
        idx = frames.coarse_indices(100, 12)
        self.assertEqual(len(idx), 12)
        self.assertEqual(len(set(idx)), 12)
        self.assertEqual(idx, sorted(idx))
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], 99)
        for want in (0, 1, 96, 97, 98, 99):
            self.assertIn(want, idx)

    def test_short_video_returns_all(self) -> None:
        self.assertEqual(frames.coarse_indices(3, 12), [0, 1, 2])

    def test_single_frame(self) -> None:
        self.assertEqual(frames.coarse_indices(1, 12), [0])


class WindowIndexTest(unittest.TestCase):
    def test_25fps_window_is_40ms_per_frame(self) -> None:
        idx = frames.window_indices(100, 25.0, 1.0, 1.4)
        self.assertEqual(idx, list(range(25, 36)))
        gaps = {round((b - a) * 1000.0 / 25.0) for a, b in zip(idx, idx[1:])}
        self.assertEqual(gaps, {40})

    def test_low_fps_is_rejected_not_interpolated(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            frames.window_indices(100, 12.5, 1.0, 1.4)
        self.assertIn("不插帧", str(caught.exception))

    def test_too_many_frames_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            frames.window_indices(1000, 25.0, 0.0, 5.0)

    def test_50fps_is_subsampled_to_25(self) -> None:
        idx = frames.window_indices(200, 50.0, 1.0, 1.4)
        self.assertEqual(len(idx), 11)
        self.assertEqual(idx[0], 50)
        self.assertEqual(idx[-1], 70)


class CliTest(unittest.TestCase):
    def test_coarse_writes_frames_sheet_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video = tmp / "clip.mp4"
            write_video(video, frame_count=100, fps=25.0)
            out = tmp / "coarse"
            self.assertEqual(frames.main([str(video), str(out), "--coarse", "12"]), 0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "coarse")
            self.assertEqual(manifest["sampled"], 12)
            self.assertEqual(manifest["frames"][0]["frame_index"], 0)
            self.assertEqual(manifest["frames"][-1]["frame_index"], 99)
            self.assertEqual(len(list(out.glob("f*.png"))), 12)
            self.assertEqual(len(list(out.glob("sheet_*.png"))), 1)
            self.assertIn("sheet_000.png", manifest["images"])

    def test_window_writes_native_frames(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video = tmp / "clip.mp4"
            write_video(video, frame_count=100, fps=25.0)
            out = tmp / "window"
            self.assertEqual(frames.main(
                [str(video), str(out), "--window", "1.0:1.4"]), 0)
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["mode"], "window")
            self.assertEqual(manifest["sampled"], 11)
            self.assertEqual(manifest["window_ms"], [1000, 1400])
            t_ms = [f["t_ms"] for f in manifest["frames"]]
            self.assertEqual({b - a for a, b in zip(t_ms, t_ms[1:])}, {40})

    def test_low_fps_video_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video = tmp / "slow.mp4"
            write_video(video, frame_count=40, fps=12.5)
            with self.assertRaises(SystemExit):
                frames.main([str(video), str(tmp / "w"), "--window", "0.8:1.6"])

    def test_budget_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            video = tmp / "clip.mp4"
            write_video(video, frame_count=20, fps=25.0)
            with self.assertRaises(SystemExit):
                frames.main([str(video), str(tmp / "b"), "--coarse", "12",
                             "--budget", "500"])

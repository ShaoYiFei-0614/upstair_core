# 把训练视频(MP4)抽成带时间戳的帧序列,供监视器子代理评审。
#
# 用法:
#   .venv\Scripts\python.exe .claude\skills\visual-monitor\read_video.py ^
#       <mp4路径> <输出目录> [--max-frames 16] [--max-width 768] [--start 秒] [--end 秒]
#
# 产物:
#   <输出目录>/f000.png ...   均匀采样帧(必含首尾帧),已按 max-width 缩放
#   <输出目录>/manifest.json  帧清单(file / frame_index / t_ms) + fps/时长,
#                             评审 prompt 直接引用其中的 t_ms 还原时间轴
#
# 为什么必须抽帧:Read 工具无法直接读视频;直接把视频「给」模型是看不到内容的。
# 先抽帧,再让子代理逐帧 Read PNG,是唯一可靠的看片方式。

import argparse
import json
from pathlib import Path

import imageio
import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="输入 MP4 路径")
    ap.add_argument("outdir", help="帧序列输出目录(建议放 .tmp/ 下)")
    ap.add_argument("--max-frames", type=int, default=16,
                    help="最多采多少帧(默认 16,控制 token 预算)")
    ap.add_argument("--max-width", type=int, default=768,
                    help="帧缩放到的最大宽度(默认 768)")
    ap.add_argument("--start", type=float, default=None, help="采样起点 [秒]")
    ap.add_argument("--end", type=float, default=None, help="采样终点 [秒]")
    args = ap.parse_args()

    src = Path(args.video)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    reader = imageio.get_reader(src)
    meta = reader.get_meta_data()
    fps = meta.get("fps")
    n = reader.count_frames()

    i0 = int((args.start or 0.0) * fps) if (args.start is not None and fps) else 0
    i1 = int(args.end * fps) if (args.end is not None and fps) else n - 1
    i0, i1 = max(0, i0), min(i1, n - 1)

    m = min(args.max_frames, i1 - i0 + 1)
    idxs = sorted({round(i0 + k * (i1 - i0) / (m - 1)) for k in range(m)}) if i1 > i0 else [i0]

    frames = []
    for k, i in enumerate(idxs):
        frame = np.asarray(reader.get_data(i))
        img = Image.fromarray(frame)
        if img.width > args.max_width:
            img = img.resize((args.max_width,
                              round(img.height * args.max_width / img.width)),
                             Image.LANCZOS)
        name = f"f{k:03d}.png"
        img.save(out / name)
        frames.append({"file": name, "frame_index": i,
                       "t_ms": round(i * 1000.0 / fps) if fps else None})

    manifest = {
        "video": str(src),
        "total_frames": n,
        "fps": fps,
        "duration_ms": round(n * 1000.0 / fps) if fps else None,
        "sampled": len(frames),
        "frames": frames,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()

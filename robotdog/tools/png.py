"""Minimal dependency-free PNG writer.

The sandbox has no Pillow and installing one just to dump a frame is wasteful,
so this module hand-rolls the PNG container (zlib + struct only).  It is shared
by the verification scripts, the evaluation CLI and any future video export.

``write_png`` handles truecolour 8-bit RGB only, which is exactly what
``mujoco.Renderer`` returns.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def write_png(path: str | Path, rgb: np.ndarray, *, compress_level: int = 6) -> Path:
    """Write an ``(H, W, 3)`` uint8 array as a PNG file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"需要 (H, W, 3) 的 RGB 数组, 收到 {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"需要 uint8, 收到 {arr.dtype}")
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, compress_level))
        + chunk(b"IEND", b"")
    )
    return path


def read_png(path: str | Path) -> np.ndarray:
    """Read back a PNG written by :func:`write_png` (8-bit truecolour RGB)."""
    data = Path(path).read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("不是 PNG 文件")
    pos = 8
    w = h = None
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            w, h, depth, colour = struct.unpack(">IIBB", payload[:10])
            if depth != 8 or colour != 2:
                raise ValueError(f"仅支持 8-bit RGB, 收到 depth={depth} colour={colour}")
        elif tag == b"IDAT":
            idat += payload
        elif tag == b"IEND":
            break
    if w is None or h is None:
        raise ValueError("缺少 IHDR")
    raw = zlib.decompress(bytes(idat))
    stride = w * 3
    out = np.empty((h, w, 3), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.uint8)
    for y in range(h):
        base = y * (stride + 1)
        filt = raw[base]
        line = np.frombuffer(raw[base + 1:base + 1 + stride], dtype=np.uint8).copy()
        if filt == 0:
            pass
        elif filt == 1:  # Sub
            for i in range(3, stride):
                line[i] = (int(line[i]) + int(line[i - 3])) & 0xFF
        elif filt == 2:  # Up
            line = (line.astype(np.int32) + prev.astype(np.int32)).astype(np.uint8)
        elif filt == 3:  # Average
            for i in range(stride):
                left = int(line[i - 3]) if i >= 3 else 0
                line[i] = (int(line[i]) + ((left + int(prev[i])) >> 1)) & 0xFF
        elif filt == 4:  # Paeth
            for i in range(stride):
                a = int(line[i - 3]) if i >= 3 else 0
                b = int(prev[i])
                c = int(prev[i - 3]) if i >= 3 else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (int(line[i]) + pr) & 0xFF
        else:
            raise ValueError(f"未知的 PNG 行滤波类型 {filt}")
        out[y] = line.reshape(w, 3)
        prev = line
    return out


def downscale(img: np.ndarray, factor: int) -> np.ndarray:
    """Box-filter downscale by an integer ``factor``."""
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    h2, w2 = h // factor, w // factor
    if h2 == 0 or w2 == 0:
        return img
    crop = img[: h2 * factor, : w2 * factor]
    return crop.reshape(h2, factor, w2, factor, 3).mean(axis=(1, 3)).astype(np.uint8)


def hstack(images: list[np.ndarray], *, pad: int = 0,
           background: int = 255) -> np.ndarray:
    """Lay images out left-to-right (all must share a height after cropping)."""
    if not images:
        raise ValueError("没有图像")
    h = max(im.shape[0] for im in images)
    w = sum(im.shape[1] for im in images) + pad * (len(images) - 1)
    out = np.full((h, w, 3), background, dtype=np.uint8)
    x = 0
    for im in images:
        out[: im.shape[0], x:x + im.shape[1]] = im
        x += im.shape[1] + pad
    return out

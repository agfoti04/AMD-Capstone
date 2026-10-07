"""Image and calibration writers that do not require OpenCV."""

import json
import struct
from pathlib import Path


def write_image(path, width, height, rgb_bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        import numpy as np
        array = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape(height, width, 3)
        if cv2.imwrite(str(path.with_suffix(".png")), array[:, :, ::-1]):
            return path.with_suffix(".png")
    except Exception:
        pass
    try:
        from PIL import Image
        image = Image.frombytes("RGB", (width, height), rgb_bytes)
        dest = path.with_suffix(".png")
        image.save(dest)
        return dest
    except Exception:
        pass
    dest = path.with_suffix(".bmp")
    dest.write_bytes(encode_bmp(width, height, rgb_bytes))
    return dest


def encode_bmp(width, height, rgb_bytes):
    expected = width * height * 3
    if len(rgb_bytes) != expected:
        raise ValueError(f"RGB buffer is {len(rgb_bytes)} bytes, expected {expected}")
    row_stride = (width * 3 + 3) & ~3
    pixels = bytearray()
    for y in range(height - 1, -1, -1):
        row = bytearray()
        start = y * width * 3
        for x in range(width):
            r, g, b = rgb_bytes[start + x * 3:start + x * 3 + 3]
            row.extend((b, g, r))
        row.extend(b"\x00" * (row_stride - width * 3))
        pixels.extend(row)
    header = struct.pack("<2sIHHI", b"BM", 54 + len(pixels), 0, 0, 54)
    info = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, len(pixels), 2835, 2835, 0, 0)
    return header + info + bytes(pixels)


def read_bmp(path):
    data = Path(path).read_bytes()
    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    width, height = struct.unpack_from("<ii", data, 18)
    bits = struct.unpack_from("<H", data, 28)[0]
    if bits != 24:
        raise ValueError("only 24-bit BMP is supported")
    row_stride = (width * 3 + 3) & ~3
    rgb = bytearray(width * height * 3)
    for row in range(height):
        src = pixel_offset + (height - 1 - row) * row_stride
        for col in range(width):
            b, g, r = data[src + col * 3:src + col * 3 + 3]
            dest = (row * width + col) * 3
            rgb[dest:dest + 3] = bytes((r, g, b))
    return width, height, bytes(rgb)


def write_pfm(path, width, height, depth):
    """Portable float map, bottom row first, optical-axis depth in meters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(depth) != width * height:
        raise ValueError("depth length does not match the image size")
    header = f"Pf\n{width} {height}\n-1.0\n".encode("ascii")
    payload = bytearray()
    for row in range(height - 1, -1, -1):
        start = row * width
        payload.extend(struct.pack("<" + "f" * width, *depth[start:start + width]))
    path.write_bytes(header + payload)
    return path


def write_calibration(path, intrinsics, frame_id, stamp=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "width": intrinsics["width"],
        "height": intrinsics["height"],
        "fx": intrinsics["fx"],
        "fy": intrinsics["fy"],
        "cx": intrinsics["cx"],
        "cy": intrinsics["cy"],
        "frame_id": frame_id,
        "stamp": stamp,
        "distortion": "plumb_bob",
        "d": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def decode_ros_image(msg):
    encoding = msg.encoding.lower()
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported image encoding {msg.encoding}")
    rgb = bytearray(msg.width * msg.height * 3)
    for row in range(msg.height):
        src = row * msg.step
        for col in range(msg.width):
            pixel = msg.data[src + col * channels:src + col * channels + channels]
            if encoding == "rgb8":
                color = pixel[:3]
            elif encoding == "bgr8":
                color = bytes((pixel[2], pixel[1], pixel[0]))
            elif encoding == "rgba8":
                color = pixel[:3]
            elif encoding == "bgra8":
                color = bytes((pixel[2], pixel[1], pixel[0]))
            else:
                color = bytes((pixel[0], pixel[0], pixel[0]))
            dest = (row * msg.width + col) * 3
            rgb[dest:dest + 3] = color
    return msg.width, msg.height, bytes(rgb)


def decode_ros_depth(msg):
    if msg.encoding != "32FC1":
        raise ValueError(f"unsupported depth encoding {msg.encoding}")
    values = []
    for row in range(msg.height):
        src = row * msg.step
        count = msg.width
        values.extend(struct.unpack_from("<" + "f" * count, msg.data, src))
    return values

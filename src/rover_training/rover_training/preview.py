"""Draw saved YOLO boxes on captured images."""

import argparse
from pathlib import Path

from rover_training.geometry import CLASS_NAMES
from rover_training.image_io import read_bmp, write_image
from rover_training.yolo_export import read_label_file


def preview_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    classes_path = dataset_dir / 'classes.txt'
    class_names = classes_path.read_text(encoding='utf-8').splitlines() if classes_path.exists() else CLASS_NAMES
    written = []
    for split in ("train", "val", "test"):
        image_dir = dataset_dir / "images" / split
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in {".bmp", ".png", ".jpg", ".jpeg"}:
                continue
            label_path = dataset_dir / "labels" / split / f"{image_path.stem}.txt"
            width, height, rgb = load_rgb(image_path)
            labels = read_label_file(label_path, width, height, class_names) if label_path.exists() else []
            painted = draw_boxes(rgb, width, height, labels)
            dest = dataset_dir / "previews" / split / image_path.stem
            written.append(write_image(dest, width, height, painted))
    return written


# Preview overlays only. Clean images stay unlabeled, and these colors are not material colors.
_BOX_COLORS = {
    0: (40, 220, 60),
    1: (80, 210, 230),
    2: (240, 200, 40),
    3: (40, 200, 230),
    4: (210, 210, 220),
    5: (70, 140, 230),
    6: (190, 150, 90),
}


def draw_boxes(rgb, width, height, labels, color=(40, 220, 60)):
    pixels = bytearray(rgb)
    for label in labels:
        color = _BOX_COLORS.get(int(label.get("class_id", 0)), color)
        x0 = int(max(0, min(width - 1, round(label["xmin"]))))
        x1 = int(max(0, min(width - 1, round(label["xmax"]))))
        y0 = int(max(0, min(height - 1, round(label["ymin"]))))
        y1 = int(max(0, min(height - 1, round(label["ymax"]))))
        _hline(pixels, width, height, x0, x1, y0, color)
        _hline(pixels, width, height, x0, x1, y1, color)
        _vline(pixels, width, height, y0, y1, x0, color)
        _vline(pixels, width, height, y0, y1, x1, color)
    return bytes(pixels)


def load_rgb(path):
    path = Path(path)
    if path.suffix.lower() == ".bmp":
        return read_bmp(path)
    try:
        from PIL import Image
        image = Image.open(path).convert("RGB")
        return image.size[0], image.size[1], image.tobytes()
    except Exception:
        pass
    try:
        import cv2
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not read {path}")
        rgb = bgr[:, :, ::-1]
        height, width = rgb.shape[:2]
        return width, height, rgb.tobytes()
    except Exception as exc:
        raise ValueError(f"could not read {path}") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description="Draw YOLO rock boxes on a saved dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args(argv)
    written = preview_dataset(args.dataset)
    if not written:
        raise SystemExit(f"no images found under {args.dataset / 'images'}")
    for path in written:
        print(path)


def _hline(pixels, width, height, x0, x1, y, color):
    if y < 0 or y >= height:
        return
    if x1 < x0:
        x0, x1 = x1, x0
    for x in range(x0, x1 + 1):
        _put(pixels, width, x, y, color)


def _vline(pixels, width, height, y0, y1, x, color):
    if x < 0 or x >= width:
        return
    if y1 < y0:
        y0, y1 = y1, y0
    for y in range(y0, y1 + 1):
        _put(pixels, width, x, y, color)


def _put(pixels, width, x, y, color):
    index = (y * width + x) * 3
    pixels[index:index + 3] = bytes(color)


if __name__ == "__main__":
    main()

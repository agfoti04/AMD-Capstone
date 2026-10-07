"""YOLO detection labels. Simulator class names map to zero-based ids."""

from pathlib import Path

from rover_training.geometry import CLASS_NAMES


def yolo_line(label, width, height):
    class_id = int(label["class_id"])
    if class_id < 0 or class_id >= len(CLASS_NAMES):
        raise ValueError(f"class id {class_id} is outside {CLASS_NAMES}")
    box_w = (label["xmax"] - label["xmin"]) / float(width)
    box_h = (label["ymax"] - label["ymin"]) / float(height)
    xc = (label["xmin"] + label["xmax"]) * 0.5 / float(width)
    yc = (label["ymin"] + label["ymax"]) * 0.5 / float(height)
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    box_w = min(max(box_w, 0.0), 1.0)
    box_h = min(max(box_h, 0.0), 1.0)
    return f"{class_id} {xc:.6f} {yc:.6f} {box_w:.6f} {box_h:.6f}"


def write_label_file(path, labels, width, height):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [yolo_line(label, width, height) for label in labels]
    text = ("\n".join(lines) + "\n") if lines else ""
    path.write_text(text, encoding="utf-8")
    return path


def read_label_file(path, width, height):
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    labels = []
    for line in text.splitlines():
        class_id, xc, yc, box_w, box_h = line.split()
        class_id = int(class_id)
        xc, yc, box_w, box_h = (float(value) for value in (xc, yc, box_w, box_h))
        labels.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "xmin": (xc - box_w * 0.5) * width,
            "ymin": (yc - box_h * 0.5) * height,
            "xmax": (xc + box_w * 0.5) * width,
            "ymax": (yc + box_h * 0.5) * height,
        })
    return labels


def write_dataset_index(dataset_dir):
    dataset_dir = Path(dataset_dir)
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    data_yaml = (
        f"path: {dataset_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"{names}\n"
    )
    (dataset_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    (dataset_dir / "classes.txt").write_text(
        "\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

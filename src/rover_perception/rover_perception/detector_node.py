"""Configurable rock detector.

The only inference input is the camera image. Change `image_topic` to point
at the simulator or the physical OAK-D. A missing model publishes no boxes.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


class RockDetector(Node):
    def __init__(self):
        super().__init__("rock_detector")
        self.declare_parameter("image_topic", "/oak/image_raw")
        self.declare_parameter("detection_topic", "/detections")
        self.declare_parameter("backend", "none")
        self.declare_parameter("model_path", "")
        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        detection_topic = self.get_parameter("detection_topic").get_parameter_value().string_value
        self.backend = self.get_parameter("backend").get_parameter_value().string_value
        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.model = None
        if self.backend == "ultralytics" and self.model_path:
            self.model = _load_ultralytics(self.model_path, self.get_logger())
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.pub = self.create_publisher(Detection2DArray, detection_topic, qos)
        self.status_pub = self.create_publisher(String, "~/status", 10)
        self.create_subscription(Image, image_topic, self._on_image, qos)
        ready = self.model is not None
        self.get_logger().info(
            f"listening on {image_topic}; backend={self.backend}; model_loaded={ready}"
        )

    def _on_image(self, msg):
        detections = Detection2DArray()
        detections.header = msg.header
        if self.model is not None:
            detections.detections = _infer(self.model, msg)
        self.pub.publish(detections)
        status = String()
        status.data = f"backend={self.backend}; boxes={len(detections.detections)}"
        self.status_pub.publish(status)


def _set_box_center(bbox, x, y):
    """Humble vision_msgs uses Pose2D.x. Newer messages use center.position."""
    center = bbox.center
    if hasattr(center, "position"):
        center.position.x = float(x)
        center.position.y = float(y)
    else:
        center.x = float(x)
        center.y = float(y)
    if hasattr(center, "theta"):
        center.theta = 0.0


def _load_ultralytics(model_path, logger):
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics is not installed; publishing no boxes")
        return None
    logger.info(f"loading weights from {model_path}")
    return YOLO(model_path)


def _infer(model, msg):
    if msg.encoding.lower() not in ("rgb8", "bgr8"):
        return []
    width = msg.width
    channels = 3
    rows = []
    for row in range(msg.height):
        start = row * msg.step
        row_bytes = bytes(msg.data[start:start + width * channels])
        if msg.encoding.lower() == "bgr8":
            pixels = bytearray(row_bytes)
            for col in range(width):
                i = col * 3
                pixels[i], pixels[i + 2] = pixels[i + 2], pixels[i]
            row_bytes = bytes(pixels)
        rows.append(row_bytes)
    try:
        import numpy as np
    except ImportError:
        return []
    image = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(msg.height, width, 3)
    results = model.predict(image, verbose=False)
    boxes = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            detection = Detection2D()
            detection.header = msg.header
            xyxy = box.xyxy[0].tolist()
            _set_box_center(detection.bbox, (xyxy[0] + xyxy[2]) * 0.5, (xyxy[1] + xyxy[3]) * 0.5)
            detection.bbox.size_x = xyxy[2] - xyxy[0]
            detection.bbox.size_y = xyxy[3] - xyxy[1]
            hypothesis = ObjectHypothesisWithPose()
            class_index = int(box.cls[0])
            hypothesis.hypothesis.class_id = "rock" if class_index == 0 else str(class_index)
            hypothesis.hypothesis.score = float(box.conf[0])
            detection.results.append(hypothesis)
            boxes.append(detection)
    return boxes


def main():
    rclpy.init()
    node = RockDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

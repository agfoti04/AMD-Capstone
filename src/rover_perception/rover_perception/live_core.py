"""Image conversion and expiring drive commands, independent of ROS."""
import math
import threading
import time

import numpy as np


DEFAULT_CAMERA_TIMEOUT = 3.0
MAX_LINEAR_SPEED = 0.6
MAX_ANGULAR_SPEED = 1.5


def camera_timeout(value):
    """Bound the simulation camera grace period without changing the drive lease."""
    if isinstance(value, bool):
        raise ValueError('Camera timeout must be between 0.1 and 10 seconds')
    value = float(value)
    if not math.isfinite(value) or not 0.1 <= value <= 10:
        raise ValueError('Camera timeout must be between 0.1 and 10 seconds')
    return value


def image_to_bgr(msg):
    """Decode ROS row padding and preserve Ultralytics' NumPy BGR convention."""
    if msg.encoding.lower() not in ('rgb8', 'bgr8'):
        raise ValueError(f'Expected rgb8/bgr8, received {msg.encoding}')
    if msg.width <= 0 or msg.height <= 0 or msg.step < msg.width * 3:
        raise ValueError('Invalid image dimensions/stride')
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if raw.size != msg.height * msg.step:
        raise ValueError('Image buffer does not match dimensions/stride')
    image = raw.reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    if msg.encoding.lower() == 'rgb8':
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


class DriveLease:
    """A command expires on wall time, including when simulation time is paused."""
    def __init__(self, timeout=0.35, max_linear=MAX_LINEAR_SPEED, max_angular=MAX_ANGULAR_SPEED, clock=time.monotonic):
        self.timeout, self.max_linear, self.max_angular = timeout, max_linear, max_angular
        self.clock = clock
        self.lock = threading.Lock()
        self.command = (0.0, 0.0)
        self.deadline = 0.0

    def set(self, linear, angular):
        if isinstance(linear, bool) or isinstance(angular, bool):
            raise ValueError('Drive speeds must be finite numbers')
        linear, angular = float(linear), float(angular)
        if not math.isfinite(linear) or not math.isfinite(angular):
            raise ValueError('Drive speeds must be finite numbers')
        with self.lock:
            self.command = (max(-self.max_linear, min(self.max_linear, linear)),
                            max(-self.max_angular, min(self.max_angular, angular)))
            self.deadline = self.clock() + self.timeout

    def get(self):
        with self.lock:
            return self.command if self.clock() < self.deadline else (0.0, 0.0)


def detection_rows(result):
    if result.boxes is None:
        return []
    rows = []
    for box in result.boxes:
        index = int(box.cls[0])
        rows.append({'class_id': index, 'class_name': result.names[index],
                     'confidence': float(box.conf[0]), 'xyxy': box.xyxy[0].tolist()})
    return rows

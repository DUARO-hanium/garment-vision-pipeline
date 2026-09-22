"""Run FashionAI clothing-keypoint inference from a live ROS 2 image topic."""

from __future__ import annotations

import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from fashionai_keypoint_capture import (
    KEYPOINT_NAMES,
    load_net,
    predict_keypoints,
)


class FashionAiKeypointNode(Node):
    def __init__(self) -> None:
        super().__init__("fashionai_keypoint")

        self.declare_parameter("image_topic", "/camera1/image_raw")
        self.declare_parameter("output_topic", "/fashionai/keypoints/image")
        self.declare_parameter("clothing_type", "blouse")
        self.declare_parameter("roi_x", 0)
        self.declare_parameter("roi_y", 0)
        self.declare_parameter("roi_width", 0)
        self.declare_parameter("roi_height", 0)
        self.declare_parameter("draw_labels", True)

        image_topic = str(self.get_parameter("image_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        self.clothing_type = str(self.get_parameter("clothing_type").value)
        if self.clothing_type not in KEYPOINT_NAMES:
            raise ValueError(f"Unsupported clothing_type: {self.clothing_type}")
        names = KEYPOINT_NAMES[self.clothing_type]
        try:
            self.shoulder_indices = [
                names.index("shoulder_left"),
                names.index("shoulder_right"),
            ]
        except ValueError as error:
            raise ValueError(
                f"The '{self.clothing_type}' model does not define shoulder keypoints"
            ) from error
        self.keypoint_names = names

        self.roi_x = int(self.get_parameter("roi_x").value)
        self.roi_y = int(self.get_parameter("roi_y").value)
        self.roi_width = int(self.get_parameter("roi_width").value)
        self.roi_height = int(self.get_parameter("roi_height").value)
        self.draw_labels = bool(self.get_parameter("draw_labels").value)

        self.bridge = CvBridge()
        self.net = load_net(self.clothing_type, env_id=None)
        self.last_log_time = 0.0

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(Image, output_topic, sensor_qos)
        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.on_image,
            sensor_qos,
        )
        self.get_logger().info(f"input: {image_topic}")
        self.get_logger().info(f"annotated output: {output_topic}")
        self.get_logger().info("displaying only keypoints 4 and 5 (left/right shoulder)")

    def crop_roi(self, frame):
        height, width = frame.shape[:2]
        x0 = max(0, min(self.roi_x, width - 1))
        y0 = max(0, min(self.roi_y, height - 1))

        if self.roi_width <= 0 or self.roi_height <= 0:
            return frame, 0, 0, False

        x1 = max(x0 + 1, min(x0 + self.roi_width, width))
        y1 = max(y0 + 1, min(y0 + self.roi_height, height))
        return frame[y0:y1, x0:x1], x0, y0, True

    def draw_shoulders(self, frame, keypoints):
        annotated = frame.copy()
        colors = [(255, 80, 80), (80, 80, 255)]
        for color, index in zip(colors, self.shoulder_indices):
            u, v, visible = keypoints[index].tolist()
            if visible <= 0:
                continue
            point = (int(u), int(v))
            cv2.drawMarker(
                annotated,
                point,
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=3,
            )
            cv2.circle(annotated, point, 6, color, 2)
            if self.draw_labels:
                label = f"{index + 1}: {self.keypoint_names[index]}"
                cv2.putText(
                    annotated,
                    label,
                    (point[0] + 8, max(point[1] - 8, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        return annotated

    def on_image(self, message: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            model_frame, offset_x, offset_y, has_roi = self.crop_roi(frame)

            started = time.perf_counter()
            keypoints = predict_keypoints(self.clothing_type, model_frame, self.net)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            keypoints[:, 0] += offset_x
            keypoints[:, 1] += offset_y
            annotated = self.draw_shoulders(frame, keypoints)
            if has_roi:
                cv2.rectangle(
                    annotated,
                    (offset_x, offset_y),
                    (offset_x + model_frame.shape[1] - 1, offset_y + model_frame.shape[0] - 1),
                    (0, 255, 0),
                    2,
                )
            cv2.putText(
                annotated,
                f"inference: {elapsed_ms:.1f} ms",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            output = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            output.header = message.header
            self.publisher.publish(output)

            now = time.monotonic()
            if now - self.last_log_time >= 2.0:
                self.get_logger().info(f"inference time: {elapsed_ms:.1f} ms")
                self.last_log_time = now
        except Exception as error:  # Keep the live node running after a bad frame.
            self.get_logger().error(f"frame processing failed: {error}")


def main() -> None:
    rclpy.init()
    node = FashionAiKeypointNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

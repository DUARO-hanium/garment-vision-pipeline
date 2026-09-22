"""Save one ROS 2 image frame losslessly and record its source metadata."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/camera1/image_raw")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--expected-width", type=int)
    parser.add_argument("--expected-height", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    rclpy.init()
    node = rclpy.create_node("resolution_test_frame_capture")
    bridge = CvBridge()
    holder = {}

    def callback(message: Image) -> None:
        if "frame" not in holder:
            holder["frame"] = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            holder["header"] = message.header
            holder["encoding"] = message.encoding

    subscription = node.create_subscription(Image, args.topic, callback, qos_profile_sensor_data)
    deadline = time.monotonic() + args.timeout
    try:
        while "frame" not in holder and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()

    if "frame" not in holder:
        raise RuntimeError(f"No image received from {args.topic} within {args.timeout:.1f}s")

    frame = holder["frame"]
    height, width = frame.shape[:2]
    if args.expected_width is not None and width != args.expected_width:
        raise RuntimeError(f"Expected width {args.expected_width}, received {width}")
    if args.expected_height is not None and height != args.expected_height:
        raise RuntimeError(f"Expected height {args.expected_height}, received {height}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() != ".png":
        raise ValueError("Use a .png output so the comparison capture is lossless")
    if not cv2.imwrite(str(output), frame):
        raise RuntimeError(f"Failed to save {output}")

    header = holder["header"]
    metadata = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "topic": args.topic,
        "width": width,
        "height": height,
        "encoding": holder["encoding"],
        "frame_id": header.frame_id,
        "stamp": {"sec": int(header.stamp.sec), "nanosec": int(header.stamp.nanosec)},
        "image": str(output),
    }
    with open(output.with_suffix(".json"), "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
    print(f"[DONE] saved {width}x{height} frame: {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import torch

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge


SHOW_YOLO_WINDOW = False
TARGET_CLASS = "sports ball"

bridge = CvBridge()

model = torch.hub.load(
    "yolov5",
    "yolov5s",
    source="local"
)

state_pub = None


def image_callback(msg):
    try:
        img = bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8"
        )

        frame = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        results = model(frame)

        detections = results.pandas().xyxy[0]

        # Keep only the target class we care about.
        target_detections = detections[
            detections["name"] == TARGET_CLASS
        ]

        state_msg = Float32MultiArray()

        if len(target_detections) > 0:

            # If more than one sports ball is detected,
            # use the highest-confidence detection.
            target = target_detections.sort_values(
                by="confidence",
                ascending=False
            ).iloc[0]

            x1 = float(target["xmin"])
            y1 = float(target["ymin"])
            x2 = float(target["xmax"])
            y2 = float(target["ymax"])

            img_h, img_w = frame.shape[:2]

            center_x = (
                ((x1 + x2) / 2.0)
                / img_w
            )

            center_y = (
                ((y1 + y2) / 2.0)
                / img_h
            )

            width = (
                (x2 - x1)
                / img_w
            )

            height = (
                (y2 - y1)
                / img_h
            )

            state_msg.data = [
                1.0,
                center_x,
                center_y,
                width,
                height
            ]

        else:

            # Explicitly tell the RL system
            # that the target is not visible.
            state_msg.data = [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            ]

        state_pub.publish(
            state_msg
        )

        if SHOW_YOLO_WINDOW:

            rendered = np.squeeze(
                results.render()
            )

            cv2.imshow(
                "YOLO Sports Ball Detection",
                rendered
            )

            cv2.waitKey(1)

    except Exception as e:

        rospy.logerr(
            f"YOLO callback error: {e}"
        )


def main():
    global state_pub

    rospy.init_node(
        "object_detection"
    )

    state_pub = rospy.Publisher(
        "/yolo_state",
        Float32MultiArray,
        queue_size=10
    )

    rospy.Subscriber(
        "/camera/rgb/image_raw",
        Image,
        image_callback,
        queue_size=1
    )

    rospy.loginfo(
        "YOLO perception node started"
    )

    rospy.loginfo(
        f"Target class: {TARGET_CLASS}"
    )

    if not SHOW_YOLO_WINDOW:
        rospy.loginfo(
            "YOLO visualization disabled for faster training"
        )

    rospy.spin()


if __name__ == "__main__":
    main()
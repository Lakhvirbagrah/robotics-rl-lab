#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import torch

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge


SHOW_YOLO_WINDOW = False


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

        detections = results.xyxy[0]

        if len(detections) > 0:
            det = detections[0]

            x1 = float(det[0])
            y1 = float(det[1])
            x2 = float(det[2])
            y2 = float(det[3])

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

            state_msg = Float32MultiArray()

            state_msg.data = [
                center_x,
                center_y,
                width,
                height
            ]

            state_pub.publish(state_msg)

        if SHOW_YOLO_WINDOW:
            rendered = np.squeeze(
                results.render()
            )

            cv2.imshow(
                "YOLO",
                rendered
            )

            cv2.waitKey(1)

    except Exception as e:
        rospy.logerr(str(e))


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

    if not SHOW_YOLO_WINDOW:
        rospy.loginfo(
            "YOLO visualization disabled for training"
        )

    rospy.spin()


if __name__ == "__main__":
    main()
#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
import torch
from std_msgs.msg import String
frame=cv2.imread("white1.jpg")
mydata=''  
model = torch.hub.load('yolov5', 'yolov5s', source='local')
# Callback function to handle received images
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray
import cv2
import numpy as np

bridge = CvBridge()

state_pub = rospy.Publisher(
    "/yolo_state",
    Float32MultiArray,
    queue_size=10
)

def image_callback(data2):
    try:

        img = bridge.imgmsg_to_cv2(data2, desired_encoding='bgr8')

        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = model(frame)

        detections = results.xyxy[0]

        if len(detections) > 0:

            det = detections[0]     # first detected object

            x1 = float(det[0])
            y1 = float(det[1])
            x2 = float(det[2])
            y2 = float(det[3])

            img_h, img_w = frame.shape[:2]

            center_x = ((x1 + x2) / 2.0) / img_w
            center_y = ((y1 + y2) / 2.0) / img_h

            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h

            state_msg = Float32MultiArray()

            state_msg.data = [
                center_x,
                center_y,
                width,
                height
            ]

            state_pub.publish(state_msg)

        cv2.imshow("YOLO", np.squeeze(results.render()))
        cv2.waitKey(1)

    except Exception as e:
        rospy.logerr(str(e))
def main():
    rospy.init_node('Object Detection')
    
    # Create a subscriber to the camera image topic
    rospy.Subscriber('/camera/rgb/image_raw', Image, image_callback)
    rate=rospy.Rate(60)
    #file_name=act+'.txt'
    #image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #results = model(frame)
    #cv2.imshow('raw',frame)
    #cv2.waitKey(1)
    #cv2.imshow('YOLO', np.squeeze(results.render()))
    #rate.sleep()
    rospy.spin()
                # Cv2.waitkey

if __name__ == '__main__':
    main()

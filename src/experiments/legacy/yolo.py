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
def image_callback(data2):
    try:
        # Convert the ROS Image message to OpenCV format
        bridge=CvBridge()
        global mydata2
        global frame
        mydata2=bridge.imgmsg_to_cv2(data2)
        frame=cv2.cvtColor(mydata2, cv2.COLOR_BGR2RGB)
        results = model(frame)
        cv2.imshow('YOLO', np.squeeze(results.render()))
        # Display the image
        #cv2.imshow("Camera", mydata2)
        cv2.waitKey(1)
    except Exception as e:
        print(e)

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

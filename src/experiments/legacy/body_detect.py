#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import cv2

import rospy 

from std_msgs.msg import String
frame=cv2.imread("white1.jpg")
mydata=''  
# Callback function to handle received images
def image_callback(data2):
    try:
        # Convert the ROS Image message to OpenCV format
        bridge=CvBridge()
        global mydata2
        global frame
        mydata2=bridge.imgmsg_to_cv2(data2)
        frame=mydata2
        # Display the image
        ##cv2.imshow("Camera", mydata2)
        #cv2.waitKey(1)
    except Exception as e:
        print(e)

def main():
    rospy.init_node('image_subscriber')
    
    # Create a subscriber to the camera image topic
    rospy.Subscriber('/camera/rgb/image_raw', Image, image_callback)
    rate=rospy.Rate(60)
    #file_name=act+'.txt'
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
    #f=open(file_name,'wb')
    save_list=[]
    body_list=[]
    right_hand=[]
    left_hand=[]
    face=[]
    
    
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while True:
            
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Make Detections
            results = holistic.process(image)
            # print(results.face_landmarks)
            
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
            
            # Recolor image back to BGR for rendering
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # 1. Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                     mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                     )
            face.append(results.face_landmarks)
            # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                     )
            right_hand.append(results.right_hand_landmarks)
                
    
            # 3. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                     )
            left_hand.append(results.left_hand_landmarks)
            # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                     )
                            
    
            body_list.append(results.pose_landmarks) 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            cv2.imshow('Body Detection', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            rate.sleep()
    #save_list.append(body_list)
    #save_list.append(right_hand)
    #save_list.append(left_hand)
    #save_list.append(face)
    #pkl.dump(save_list, f)
    
    # Keep the program running until interrupted
 #           rospy.spin()

if __name__ == '__main__':
    main()

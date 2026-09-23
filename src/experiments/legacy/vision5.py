import mediapipe as mp
import cv2
import pickle as pkl 
from sensor_msgs.msg import Image
import rospy 
from cv_bridge import CvBridge
from std_msgs.msg import String
mydata=''  
    
def callback(data):
    
    global mydata
    mydata=str(data.data)
    #print(mydata) 
    #print(type(mydata))

def callback2(data2):
    bridge=CvBridge()
    global mydata2
    global frame
    mydata2=bridge.imgmsg_to_cv2(data2)
    frame=mydata2
    #print(mydata) 
    #print(type(mydata))
frame=cv2.imread("white1.jpg")
def vision(act,stop):
    rospy.init_node('recorder')
    rospy.Subscriber('sys_pub',String,callback)
    rospy.Subscriber('web_cam',Image,callback2)
    
    rate=rospy.Rate(1)
    file_name=act+'.txt'
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
    f=open(file_name,'wb')
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
            
     
                
            cv2.imshow('Recorder', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            stop=mydata
            print(mydata)
            if stop=='1':
                break
            rate.sleep()
    save_list.append(body_list)
    save_list.append(right_hand)
    save_list.append(left_hand)
    save_list.append(face)
    pkl.dump(save_list, f)
    
    cv2.destroyAllWindows()
    



vision('test','0')
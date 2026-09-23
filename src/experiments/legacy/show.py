import pickle
import mediapipe as mp
import cv2
import numpy as np
import time

def show(file1):
    fname=str(file1)+'.txt'
    f=open(fname, 'rb')
    data=pickle.load(f)
    body=data[0]
    right_hand=data[1]
    left_hand=data[2]
    face=data[3]
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
    mp_pose = mp.solutions.pose
    l=len(body) 
    i=0
    while i<l:
        image = cv2.imread("white1.jpg")
        height = image.shape[0]
        width = image.shape[1]
        channels = image.shape[2]
        mp_drawing.draw_landmarks(image, body[i], mp_holistic.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                  )
        mp_drawing.draw_landmarks(image, face[i], mp_holistic.FACEMESH_TESSELATION, 
                                     mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                     mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                     )
        mp_drawing.draw_landmarks(image, right_hand[i], mp_holistic.HAND_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                     )
        mp_drawing.draw_landmarks(image, left_hand[i], mp_holistic.HAND_CONNECTIONS, 
                                     mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                     mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                     )
        width = 1000
        height = 600
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation =cv2.INTER_AREA)

        cv2.namedWindow("image", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow("image", 1000, 600)
        cv2.moveWindow("image", 50, 50)

        cv2.imshow("image", resized)
        
        cv2.waitKey(5)

        i=i+1
    
    cv2.destroyAllWindows()
 


def desc(file1):
    fname=str(file1)+'.txt'
    f=open(fname, 'rb')
    data=pickle.load(f)
    print('length of data is= ',len(data))
 
show('test')
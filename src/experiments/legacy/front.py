import rospy
from std_msgs.msg import String
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

def publisher():
    # Load your custom gesture model and class names (unchanged)
    model = load_model('mp_hand_gesture')
    f = open('gesture.names', 'r')
    classNames = f.read().split('\n')
    f.close()

    act = 2

    # Initialize ROS and camera
    cap = cv2.VideoCapture(0)
    pub = rospy.Publisher('front_input', String, queue_size=1)
    rospy.init_node('front', anonymous=False)
    rate = rospy.Rate(10)

    # ====================== NEW MEDIAPIPE TASKS API (Correct imports) ======================
    # Official recommended way (2026)
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create the hand landmarker (place hand_landmarker.task in the same folder)
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7
    )
    hand_landmarker = HandLandmarker.create_from_options(options)

    mpDraw = mp.solutions.drawing_utils
    # =====================================================================================

    while True:
        _, frame = cap.read()

        # Mirror the frame
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=framergb)

        # Detect hand landmarks
        result = hand_landmarker.detect(mp_image)

        className = ''

        if result.hand_landmarks:
            landmarks = []
            for hand_landmarks_list in result.hand_landmarks:
                h, w, _ = frame.shape
                for lm in hand_landmarks_list:
                    lmx = int(lm.x * w)
                    lmy = int(lm.y * h)
                    landmarks.append([lmx, lmy])

                # Draw landmarks
                    mpDraw.draw_landmarks(
                    frame,
                    hand_landmarks_list,
                    mp.solutions.hands.HAND_CONNECTIONS
                )

            # Predict gesture with your Keras model
            prediction = model.predict([landmarks])
            classID = np.argmax(prediction)
            className = classNames[classID]

        # Your control logic (unchanged)
        if className == 'stop':
            act = 0
        elif className == 'fist':
            act = 1

        act_str = str(act)
        pub.publish(act_str)

        cv2.imshow("Output", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hand_landmarker.close()   # clean up


if __name__ == '__main__':
    publisher()
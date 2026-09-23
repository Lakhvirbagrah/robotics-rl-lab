import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
model = torch.hub.load('yolov5', 'yolov5s', source='local')
cap = cv2.VideoCapture(0)
while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor Feed
            #image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            


            
    
    # Make detections 
            results = model(frame)
    
            cv2.imshow('YOLO', np.squeeze(results.render()))

                # Cv2.waitkey
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
# Close down the frame
cv2.destroyAllWindows()
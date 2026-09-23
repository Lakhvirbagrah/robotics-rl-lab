import numpy as np
import tensorflow as tf
import cv2

# Load the pre-trained model
model_path = 'path/to/your/model'
model = tf.saved_model.load(model_path)
model = model.signatures['serving_default']

# Load the label map
label_map_path = 'path/to/your/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

# Load the image
image_path = 'path/to/your/image.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert the image to a tensor
image_tensor = tf.convert_to_tensor(image_rgb)
image_tensor = image_tensor[tf.newaxis, ...]

# Run object detection
detections = model(image_tensor)

# Extract the necessary information from the detections
num_detections = int(detections['num_detections'][0])
classes = detections['detection_classes'][0].numpy().astype(np.int32)
scores = detections['detection_scores'][0].numpy()
boxes = detections['detection_boxes'][0].numpy()

# Draw bounding boxes on the image
for i in range(num_detections):
    if scores[i] > 0.5:  # Consider detections with a confidence score above 0.5
        ymin, xmin, ymax, xmax = boxes[i]
        class_id = classes[i]
        class_name = category_index[class_id]['name']
        
        # Scale the bounding box coordinates to the image size
        height, width, _ = image.shape
        ymin = int(ymin * height)
        xmin = int(xmin * width)
        ymax = int(ymax * height)
        xmax = int(xmax * width)
        
        # Draw the bounding box and label on the image
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, class_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the resulting image
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

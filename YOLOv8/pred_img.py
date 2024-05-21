import os

from ultralytics import YOLO
import cv2


image_path = "/home/ksp/Desktop/yolov8/local_env/test_images/image3.png"
# image_path = "/home/ksp/Desktop/yolov8/local_env/test_images/gazebo_image.png"

model_path = "/home/ksp/Desktop/yolov8/local_env/all_128gb/runs/detect/train/weights/best.pt" 
# os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.65

class_colors = {
    0: (255, 0, 0),   # battery_blue
    1: (0, 255, 0),   # battery_green
    2: (0, 165, 255),  # battery_orange
    3: (128, 0, 128),  # battery_purple
    4: (0, 0, 255),    # battery_red
    5: (255, 0, 0),    # pump_blue
    6: (0, 255, 0),    # pump_green
    7: (0, 165, 255),   # pump_orange
    8: (128, 0, 128),   # pump_purple
    9: (0, 0, 255),    # pump_red
    10: (255, 0, 0),   # regulator_blue
    11: (0, 255, 0),   # regulator_green
    12: (0, 165, 255),  # regulator_orange
    13: (128, 0, 128),  # regulator_purple
    14: (0, 0, 255),   # regulator_red
    15: (255, 0, 0),   # sensor_blue
    16: (0, 255, 0),   # sensor_green
    17: (0, 165, 255),  # sensor_orange
    18: (128, 0, 128),  # sensor_purple
    19: (0, 0, 255)    # sensor_red
}

frame = cv2.imread(image_path)

results = model(frame)[0]

for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result

    if score > threshold:
            color = class_colors.get(int(class_id), (0, 0, 0))  # Default to black if class_id is not in class_colors
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

cv2.imshow("Image with Detections", frame)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()



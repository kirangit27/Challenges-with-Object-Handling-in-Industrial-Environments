import os
from ultralytics import YOLO
import cv2

VIDEOS_DIR = "/home/ksp/Desktop/yolov8/local_env/occ_val_50E/"
video_filename = 'occ_input.mp4'
video_path = os.path.join(VIDEOS_DIR, video_filename)
video_path_out = 'occ_output_6.mp4'.format(video_path)

if not os.path.isfile(video_path):
    print(f"Error: Video file '{video_filename}' not found at {VIDEOS_DIR}")
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Unable to open video file at {video_path}")
    exit()

ret, frame = cap.read()

if frame is None:
    print("Error: Unable to read the first frame from the video")
    exit()

H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

model_path = "/home/ksp/Desktop/yolov8/local_env/occ_val_50E/runs/detect/train/weights/best.pt"
model = YOLO(model_path)

threshold = 0.6

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

font_size = 0.8  # Adjust the font size here

while ret:
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            color = class_colors.get(int(class_id), (0, 0, 0))  # Default to black if class_id is not in class_colors
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 2, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()

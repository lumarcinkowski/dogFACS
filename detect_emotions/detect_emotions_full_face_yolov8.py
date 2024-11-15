import os
from ultralytics import YOLO
import cv2

model = YOLO('models\\yolov8_full_face_classifier.pt')


folder_path = 'datasets\\dataset_test2'
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

i = 0
for filename in os.listdir(folder_path):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        image_path = os.path.join(folder_path, filename)

        i += 1
        image = cv2.imread(image_path)

        results = model(image)
        print(f"{i}: {filename}")


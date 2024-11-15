import os
import cv2
from ultralytics import YOLO

input_folder = 'datasets\\dataset_test'
output_folder = 'datasets\\dataset_test2'

os.makedirs(output_folder, exist_ok=True)

model = YOLO('models\yolov8_dogface_dataset1_80epochs_best.pt')


def process_image(image_path, output_path):

    img = cv2.imread(image_path)

    results = model(img)

    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # [x_min, y_min, x_max, y_max]

        face_crop = img[y1:y2, x1:x2]

        cv2.imwrite(output_path, face_crop)
        print(f"Zapisano {output_path}")


def process_folder(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):

        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

        for filename in files:
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                image_path = os.path.join(root, filename)
                output_path = os.path.join(output_subfolder, filename)
                process_image(image_path, output_path)


process_folder(input_folder, output_folder)

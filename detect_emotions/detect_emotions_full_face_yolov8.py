import os
from ultralytics import YOLO
import cv2

folder_path = 'datasets\\dataset_test2'
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
model = YOLO('models\\yolov8_full_face_classifier.pt')

class_names = {
    0: "Zdenerwowany",
    1: "Wesoly",
    2: "Neutralny",
    3: "Smutny"
}

emotion_colors = {
    "Wesoly": (0, 255, 0),
    "Zdenerwowany": (0, 0, 255),
    "Smutny": (255, 0, 0),
    "Neutralny": (128, 128, 128)
}


def process_image(image_path):
    img = cv2.imread(image_path)

    results = model(image_path)
    emotion = class_names.get(results[0].probs.top1)

    img_layer = img.copy()
    alpha = 0.25
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1
    thickness = 1
    text_color = (169, 169, 169)

    # colored
    layer_color = emotion_colors.get(emotion)

    cv2.rectangle(img_layer, (0, 0), (img.shape[1], img.shape[0]), layer_color, -1)
    cv2.addWeighted(img_layer, alpha, img, 1 - alpha, 0, img)

    # emotion text
    (text_width, text_height), _ = cv2.getTextSize(f"{emotion}", font, font_scale, thickness)
    text_x = 15
    text_y = text_height + 15

    # emotion text background
    cv2.rectangle(img, (0, 0), (text_x + text_width + 15, text_y + 15), (0, 0, 0), thickness=cv2.FILLED)
    cv2.putText(img, f"{emotion}", (text_x, text_y), font, font_scale, text_color, thickness)

    cv2.imshow('Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


for filename in os.listdir(folder_path):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        image_path = os.path.join(folder_path, filename)

        process_image(image_path)


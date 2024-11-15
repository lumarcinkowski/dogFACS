import os
import cv2
from ultralytics import YOLO

input_folder = 'datasets\\dataset_test_FACS'

model = YOLO('models\\yolov8_FACS.pt')

class_colors = {
    0: (0, 255, 0),        # Green
    1: (0, 100, 0),        # Dark Green
    2: (255, 165, 0),      # Orange
    3: (128, 128, 128),    # Gray
    4: (173, 216, 230),    # Light Blue
    5: (255, 0, 0),        # Red
    6: (0, 0, 0),          # Black
    7: (0, 255, 255),      # Yellow
    8: (255, 102, 102),    # Light Red
    9: (139, 69, 19),      # Brown
}

class_names = {
    0: 'jezyk',
    1: 'szczeka',
    2: 'kly',
    3: 'smutne_oczy',
    4: 'wesole_oczy',
    5: 'grozne_oczy',
    6: 'usta',
    7: 'uszy_tyl',
    8: 'nos',
    9: 'uszy_przod'
}

dag_emotions = {
    1: {'jezyk': 7, 'else': 2},
    2: {'szczeka': 4, 'else': 3},
    3: {'kly': 6, 'else': 5},
    4: {
        'condition': lambda x: not(x['szczeka'] and all(not x[key] for key in x if key != 'szczeka')),
        'true': 10,
        'else': "Raczej wesoly"
    },
    5: {'uszy_tyl': 9, 'else': 8},
    6: {'nos': "Zdenerwowany", 'else': 9},
    7: {'kly': "Neutralny", 'else': "Wesoly"},
    8: {'smutne_oczy': 13, 'else': 11},
    9: {'grozne_oczy': "Zdenerwowany", 'else': 8},
    10: {'wesole_oczy': "Wesoly", 'else': 3},
    11: {'uszy_przod': 14, 'else': "Neutralny"},
    12: {'uszy_przod': "Smutny", 'else': 15},
    13: {
        'condition': lambda x: not(x['smutne_oczy'] and all(not x[key] for key in x if key != 'smutne_oczy')),
        'true': 12,
        'else': "Raczej smutny"
    },
    14: {'usta': "Smutny", 'else': "Neutralny"},
    15: {'usta': "Smutny", 'else': "Neutralny"}
}


def get_AUs(image_path):
    img = cv2.imread(image_path)
    results = model(img)

    action_detections = []

    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        label = int(result.cls[0].item())
        confidence = float(result.conf[0])

        action_detections.append({
            "label": label,
            "bbox": (x1, y1, x2, y2),
            "confidence": confidence
        })

    return action_detections


def predict_emotion(features):
    current_node = 1

    while current_node is not None:
        print(current_node)
        if isinstance(current_node, str):
            return current_node

        node_info = dag_emotions.get(current_node)

        if not node_info:
            return "Błąd: Nieprawidłowy węzeł grafu"

        if 'condition' in node_info:
            if node_info['condition'](features):
                current_node = node_info['true']
            else:
                current_node = node_info['else']
        else:
            for feature, next_node in node_info.items():
                if feature in features and features[feature]:
                    current_node = next_node
                    break
            else:
                current_node = node_info['else']

    return "Neeutralny"


def process_image(image_path, emotion):
    img = cv2.imread(image_path)

    results = model(img)

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.5
    thickness = 3
    text_color = (169, 169, 169)

    (text_width, text_height), _ = cv2.getTextSize(f"{emotion}", font, font_scale, thickness)
    text_x = (img.shape[1] - text_width) // 2
    text_y = text_height + 20

    cv2.rectangle(img, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), (0, 0, 0), thickness=cv2.FILLED)
    cv2.putText(img, f"{emotion}", (text_x, text_y), font, font_scale, text_color, thickness)

    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        label = int(result.cls[0].item())
        confidence = result.conf[0]
        label_text = f"{model.names[label]}: {confidence:.2f}"

        color = class_colors.get(label, (255, 255, 255))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        font_scale_box = 0.4
        thickness_box = 1

        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale_box, thickness_box)
        cv2.rectangle(img, (x1, y1 - text_height - 2), (x1 + text_width, y1 + 2), (0, 0, 0), thickness=cv2.FILLED)
        cv2.putText(img, label_text, (x1, y1 - 2), font, font_scale_box, color, thickness_box)

    cv2.imshow('Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


for root, dirs, files in os.walk(input_folder):
    for filename in files:

        action_features = {name: False for name in class_names.values()}

        if filename.endswith(('.jpg', '.png', '.jpeg', '.jfif')):
            image_path = os.path.join(root, filename)
            action_detections = get_AUs(image_path)

            for detection in action_detections:
                action_features[class_names[detection['label']]] = True

            print(action_features)
            emotion = predict_emotion(action_features)
            process_image(image_path, emotion)











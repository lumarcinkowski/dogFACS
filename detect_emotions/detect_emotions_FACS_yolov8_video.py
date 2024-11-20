import cv2
from ultralytics import YOLO
import os

input_folder = 'datasets\\dataset_videos\\angry'
output_folder = 'output_videos'
model = YOLO('models\\yolov8_FACS.pt')

class_colors = {
    0: (0, 255, 0),
    1: (0, 100, 0),
    2: (0, 128, 255),
    3: (255, 0, 0),
    4: (255, 50, 255),
    5: (0, 0, 255),
    6: (255, 50, 150),
    7: (0, 255, 255),
    8: (0, 0, 100),
    9: (255, 255, 50),
}

emotion_colors = {
    "Wesoly": (0, 255, 0),
    "Raczej wesoly": (144, 238, 144),
    "Zdenerwowany": (0, 0, 255),
    "Raczej zdenerwowany": (102, 102, 255),
    "Smutny": (255, 0, 0),
    "Raczej smutny": (255, 255, 0),
    "Neutralny": (128, 128, 128)
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
    9: {'grozne_oczy': "Zdenerwowany", 'else': 16},
    10: {'wesole_oczy': "Wesoly", 'else': 3},
    11: {'uszy_przod': 14, 'else': "Neutralny"},
    12: {'uszy_przod': "Smutny", 'else': 15},
    13: {
        'condition': lambda x: not(x['smutne_oczy'] and all(not x[key] for key in x if key != 'smutne_oczy')),
        'true': 12,
        'else': "Raczej smutny"
    },
    14: {'usta': "Smutny", 'else': "Neutralny"},
    15: {'usta': "Smutny", 'else': "Neutralny"},
    16: {
        'condition': lambda x: (x['kly'] and all(not x[key] for key in x if key != 'kly')),
        'true': "Raczej zdenerwowany",
        'else': 8
    }
}


def get_AUs(frame):
    results = model(frame)

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

    return "Neutralny"


def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    # video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    emotion_counts = {
        "Wesoly": 0,
        "Neutralny": 0,
        "Smutny": 0,
        "Zdenerwowany": 0,
        "Raczej wesoly": 0,
        "Raczej smutny": 0,
        "Raczej zdenerwowany": 0
    }

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        action_features = {name: False for name in class_names.values()}
        action_detections = get_AUs(frame)

        for detection in action_detections:
            action_features[class_names[detection['label']]] = True

        emotion = predict_emotion(action_features)
        emotion_counts[emotion] += 1

        frame_layer = frame.copy()
        alpha = 0.4
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1
        thickness = 1
        text_color = (169, 169, 169)

        # colored
        layer_color = emotion_colors.get(emotion)

        cv2.rectangle(frame_layer, (0, 0), (frame.shape[1], frame.shape[0]), layer_color, -1)
        cv2.addWeighted(frame_layer, alpha, frame, 1 - alpha, 0, frame)

        # emotion text
        (text_width, text_height), _ = cv2.getTextSize(f"{emotion}", font, font_scale, thickness)
        text_x = 15
        text_y = text_height + 15

        # emotion text background
        cv2.rectangle(frame, (0, 0), (text_x + text_width + 15, text_y + 15), (0, 0, 0), thickness=cv2.FILLED)
        cv2.putText(frame, f"{emotion}", (text_x, text_y), font, font_scale, text_color, thickness)

        happpy_labels = {0, 1, 4}
        angry_labels = {2, 5, 7, 8}
        sad_labels = {3, 6, 9}

        for detection in action_detections:

            label = detection['label']
            if (emotion == "Wesoly" or emotion == "Raczej wesoly") and label not in happpy_labels:
                break
            if (emotion == "Zdenerwowany" or emotion == "Raczej zdenerwowany") and label not in angry_labels:
                break
            if (emotion == "Smutny" or emotion == "Raczej smutny") and label not in sad_labels:
                break
            if emotion == "Neutralny":
                break
            else:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']

                # bbox text
                label_text = f"{model.names[label]}: {confidence:.2f}"
                color = class_colors.get(label, (255, 255, 255))
                font_scale_box = 0.4
                thickness_box = 1

                # bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale_box, thickness_box)
                cv2.rectangle(frame, (x1, y1 - text_height - 2), (x1 + text_width, y1 + 2), (0, 0, 0), thickness=cv2.FILLED)
                cv2.putText(frame, label_text, (x1, y1 - 2), font, font_scale_box, color, thickness_box)

                cv2.imshow("Podgląd wideo", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Podsumowanie emocji dla {os.path.basename(video_path)}:")
    for emotion, count in emotion_counts.items():
        print(f"{emotion}: {count} klatek")

    dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    print(f"\nDOMINUJĄCA EMOCJA: {dominant_emotion}")


for video_file in os.listdir(input_folder):
    if video_file.endswith(('.mp4', '.avi', '.mkv')):
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, f"processed_{video_file}")
        process_video(input_path, output_path)


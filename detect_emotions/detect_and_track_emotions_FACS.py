import cv2
from ultralytics import YOLO
import os

input_folder = 'datasets\\dataset_videos\\neutral'
output_folder = 'output_videos'
face_model = YOLO('models\\yolov8_dogface_dataset1_80epochs_best.pt')
au_model = YOLO('models\\yolov8_FACS.pt')

detection_interval = 10

output_text_file = os.path.join(input_folder, "wyniki_FACS_track.txt")

if os.path.exists(output_text_file):
    os.remove(output_text_file)

class_colors = {
    0: (0, 255, 0),
    1: (0, 100, 0),
    2: (0, 128, 255),
    3: (255, 0, 0),
    4: (150, 0, 150),
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
    results = au_model(frame)

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


def emotion_sum_up(emotion_number, video_path):
    total_frames = sum(emotion_number.values())

    print(f"Podsumowanie emocji dla {os.path.basename(video_path)}:")

    for emotion, count in emotion_number.items():
        percentage = (count / total_frames) * 100 if total_frames > 0 else 0
        print(f"{emotion}: {count} klatek ({percentage:.2f}%)")

    sorted_emotions = [
        (emotion, (count / total_frames) * 100)
        for emotion, count in sorted(emotion_number.items(), key=lambda x: x[1], reverse=True)
        if count > 0
    ]

    for emotion, percentage in sorted_emotions:
        print(f"{emotion}: {percentage:.2f}%")

    with open(output_text_file, "a", encoding="utf-8") as f:
        f.write(f"{os.path.basename(video_path)}")
        for emotion, percentage in sorted_emotions:
            f.write(f"\t{emotion}\t{percentage:.2f}%")
        f.write("\n")


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

    trackers = []
    frame_count = 0
    x1_face, x2_face, y1_face, y2_face = 0, 0, 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # detekcja co detection_interval klatek
        if frame_count % detection_interval == 0 or not trackers:
            trackers = []
            tracked_faces = []
            face_results = face_model(frame)

            for result in face_results[0].boxes:
                x1_face, y1_face, x2_face, y2_face = map(int, result.xyxy[0])
                confidence = float(result.conf[0])

                if confidence > 0.5:  # prog pewnosci
                    # śledzenie
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, (x1_face, y1_face, x2_face - x1_face, y2_face - y1_face))
                    trackers.append((tracker, confidence))
                    tracked_faces.append((x1_face, y1_face, x2_face, y2_face, confidence))

        is_detection = False

        # klasyfikacja emocji
        for x1_face, y1_face, x2_face, y2_face, confidence in tracked_faces:
            if confidence > 0.5:
                is_detection = True
                face_frame = frame[y1_face:y2_face, x1_face:x2_face]

                action_features = {name: False for name in class_names.values()}
                action_detections = get_AUs(frame)

                for detection in action_detections:
                    action_features[class_names[detection['label']]] = True

                emotion = predict_emotion(action_features)
                emotion_counts[emotion] += 1

                frame_layer = frame.copy()

                alpha = 0.25
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

        # aktualizowanie
        updated_faces = []
        for tracker, conf in trackers:
            success, bbox = tracker.update(frame)
            if success:
                x1_face, y1_face, w, h = map(int, bbox)
                x2_face, y2_face = x1_face + w, y1_face + h
                updated_faces.append((x1_face, y1_face, x2_face, y2_face, conf))

                cv2.rectangle(frame, (x1_face, y1_face), (x2_face, y2_face), (200, 0, 255), 2)

                # napis bbox pies
                label = "twarz psa"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                text_color = (200, 0, 255)
                bg_color = (0, 0, 0)

                (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

                text_bg_x1 = x1_face
                text_bg_y1 = y1_face - text_height - 5 if y1_face - text_height - 5 > 0 else 0
                text_bg_x2 = x1_face + text_width
                text_bg_y2 = y1_face

                cv2.rectangle(frame, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), bg_color, cv2.FILLED)

                text_x = x1_face
                text_y = y1_face - 5 if y1_face - 5 > 0 else text_height
                cv2.putText(frame, label, (text_x, text_y), font, font_scale, text_color, font_thickness)

        tracked_faces = updated_faces

        # ograniczenia AU
        best_detections = {}
        ears_detections = {7: [], 9: []}

        if is_detection:
            for detection in action_detections:
                label = detection['label']
                confidence = detection['confidence']

                if label in ears_detections:
                    ears_detections[label].append(detection)
                else:
                    if label not in best_detections or confidence > best_detections[label]['confidence']:
                        best_detections[label] = detection

            # 2 obiekty dla uszu
            for label in ears_detections:
                sorted_detections = sorted(ears_detections[label], key=lambda x: x['confidence'], reverse=True)
                for detection in sorted_detections[:2]:
                    if label not in best_detections:
                        best_detections[label] = []
                    best_detections[label].append(detection)

            # bboxy AU
            for label, detections in best_detections.items():
                if isinstance(detections, list):  # przypadki uszu
                    for detection in detections:
                        if (emotion == "Wesoly" or emotion == "Raczej wesoly") and label not in happpy_labels:
                            continue
                        if (emotion == "Zdenerwowany" or emotion == "Raczej zdenerwowany") and label not in angry_labels:
                            continue
                        if (emotion == "Smutny" or emotion == "Raczej smutny") and label not in sad_labels:
                            continue
                        if emotion == "Neutralny":
                            continue

                        x1, y1, x2, y2 = detection['bbox']
                        confidence = detection['confidence']

                        label_text = f"{au_model.names[label]}"
                        color = class_colors.get(label, (255, 255, 255))
                        font_scale_box = 0.4
                        thickness_box = 1

                        # Rysowanie bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale_box, thickness_box)
                        cv2.rectangle(frame, (x1, y1 - text_height - 2), (x1 + text_width, y1 + 2), (0, 0, 0), thickness=cv2.FILLED)
                        cv2.putText(frame, label_text, (x1, y1 - 2), font, font_scale_box, color, thickness_box)
                else:  # inne przypadki
                    detection = detections
                    if (emotion == "Wesoly" or emotion == "Raczej wesoly") and label not in happpy_labels:
                        continue
                    if (emotion == "Zdenerwowany" or emotion == "Raczej zdenerwowany") and label not in angry_labels:
                        continue
                    if (emotion == "Smutny" or emotion == "Raczej smutny") and label not in sad_labels:
                        continue
                    if emotion == "Neutralny":
                        continue

                    x1, y1, x2, y2 = detection['bbox']
                    confidence = detection['confidence']

                    label_text = f"{au_model.names[label]}"
                    color = class_colors.get(label, (255, 255, 255))
                    font_scale_box = 0.4
                    thickness_box = 1

                    # Rysowanie bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale_box, thickness_box)
                    cv2.rectangle(frame, (x1, y1 - text_height - 2), (x1 + text_width, y1 + 2), (0, 0, 0), thickness=cv2.FILLED)
                    cv2.putText(frame, label_text, (x1, y1 - 2), font, font_scale_box, color, thickness_box)

        # wyświetlanie i zapis klatki
        cv2.imshow('Frame', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    emotion_sum_up(emotion_counts, video_path)


for video_file in os.listdir(input_folder):
    if video_file.endswith(('.mp4', '.avi', '.mkv')):
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, f"processed_FACS_track_{video_file}")
        process_video(input_path, output_path)

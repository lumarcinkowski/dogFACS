import cv2
import numpy as np
from ultralytics import YOLO
import os
import csv

# wczytanie folderów i modeli
input_folder = 'datasets\\dataset_videos\\neutral'
output_file = 'datasets\\dataset_videos\\neutral\\results_FACS.csv'
output_folder = 'output_videos'
face_model = YOLO('models\\yolov8_dogface_dataset2_80epochs_best.pt')
au_model = YOLO('models\\yolov8_FACS.pt')

with open(output_file, mode="w", encoding="utf-8", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)

detection_interval = 10

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
    "Radosc": (0, 255, 0),
    "Subtelna radosc": (144, 238, 144),
    "Zlosc": (0, 0, 255),
    "Subtelna zlosc": (102, 102, 255),
    "Smutek": (255, 0, 0),
    "Subtelny smutek": (255, 255, 0),
    "Neutralnosc": (128, 128, 128)
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
    1: {
        'condition': lambda x: not(x['szczeka'] and all(not x[key] for key in x if key != 'szczeka')),
        'true': 2,
        'else': "Subtelna radosc"
    },
    2: {'wesole_oczy': "Radosc", 'else': 6},
    3: {'szczeka': 1, 'else': 6},
    4: {'jezyk': 5, 'else': 3},
    5: {'kly': "Neutralnosc", 'else': "Radosc"},
    6: {'kly': 7, 'else': 8},
    7: {
        'condition': lambda x: not(x['kly'] and all(not x[key] for key in x if key != 'kly')),
        'true': 9,
        'else': "Subtelna zlosc"
    },
    8: {'nos': 12, 'else': 11},
    9: {'grozne_oczy': "Zlosc", 'else': 10},
    10: {'nos': "Zlosc", 'else': "Neutralnosc"},
    11: {'smutne_oczy': "Smutek", 'else': 13},
    12: {'grozne_oczy': "Zlosc", 'else': "Neutralnosc"},
    13: {'uszy_przod': 14, 'else': 15},
    14: {
        'condition': lambda x: not(x['uszy_przod'] and all(not x[key] for key in x if key != 'uszy_przod')),
        'true': 15,
        'else': "Subtelny smutek"
    },
    15: {'usta': "Subtelny smutek", 'else': "Neutralnosc"}
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
    current_node = 4

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

    return "Neutralnosc"


def emotion_sum_up(emotion_number, video_path, output_csv_file):
    total_frames = sum(emotion_number.values())

    sorted_emotions = [
        (emotion, (count / total_frames) * 100)
        for emotion, count in sorted(emotion_number.items(), key=lambda x: x[1], reverse=True)
        if count > 0
    ]

    headers = ["Film"]
    for i in range(len(sorted_emotions)):
        headers.extend([f"Emocja{i+1}", f"% klatek{i+1}"])

    file_exists = os.path.exists(output_csv_file)

    with open(output_csv_file, mode="a", encoding="utf-8", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)

        if not file_exists:
            csv_writer.writerow(headers)

        row = [os.path.basename(video_path)]
        for emotion, percentage in sorted_emotions:
            row.extend([emotion, f"{percentage:.2f}"])

        while len(row) < len(headers):
            row.extend(["", ""])

        csv_writer.writerow(row)


def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    # parametry wideo
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    emotion_counts = {
        "Radosc": 0,
        "Neutralnosc": 0,
        "Smutek": 0,
        "Zlosc": 0,
        "Subtelna radosc": 0,
        "Subtelny smutek": 0,
        "Subtelna zlosc": 0
    }

    frame_count = 0
    trackers = []
    tracked_faces = []

    is_face_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if not is_face_detected:
            trackers = []
            tracked_faces = []
            face_results = face_model(frame)

            for result in face_results[0].boxes:
                x1_face, y1_face, x2_face, y2_face = map(int, result.xyxy[0])
                confidence = float(result.conf[0])

                if confidence > 0.4:  # próg pewności wykrycia pyska
                    is_face_detected = True
                    # śledzenie
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, (x1_face, y1_face, x2_face - x1_face, y2_face - y1_face))
                    trackers.append((tracker, confidence))
                    tracked_faces.append((x1_face, y1_face, x2_face, y2_face, confidence))
        else:
            if frame_count % detection_interval == 1 or not trackers:
                trackers = []
                tracked_faces = []
                face_results = face_model(frame)

                for result in face_results[0].boxes:
                    x1_face, y1_face, x2_face, y2_face = map(int, result.xyxy[0])
                    confidence = float(result.conf[0])

                    if confidence > 0.4:  # próg pewności wykrycia pyska
                        # śledzenie
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, (x1_face, y1_face, x2_face - x1_face, y2_face - y1_face))
                        trackers.append((tracker, confidence))
                        tracked_faces.append((x1_face, y1_face, x2_face, y2_face, confidence))
                    else:
                        is_face_detected = False

        updated_faces = []

        if is_face_detected:

            # rozpoznawanie emocji
            for x1_face, y1_face, x2_face, y2_face, confidence in tracked_faces:
                is_AU_detection = False
                if confidence > 0.4:

                    height, width, _ = frame.shape
                    dy = int((y2_face - y1_face) * 0.1)
                    dx = int((x2_face - x1_face) * 0.1)

                    y1_expanded = max(0, y1_face - dy)
                    y2_expanded = min(height, y2_face + dy)
                    x1_expanded = max(0, x1_face - dx)
                    x2_expanded = min(width, x2_face + dx)

                    face_frame = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]

                    if face_frame.size == 0:
                        continue

                    action_features = {name: False for name in class_names.values()}
                    action_detections = get_AUs(face_frame)
                    is_AU_detection = True

                    for detection in action_detections:
                        action_features[class_names[detection['label']]] = True

                    emotion = predict_emotion(action_features)

                    if emotion:
                        emotion_counts[emotion] += 1

                    frame_layer = frame.copy()

                    alpha = 0.25
                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 1
                    thickness = 1
                    text_color = (169, 169, 169)

                    # kolorowanie klatki
                    layer_color = emotion_colors.get(emotion)

                    cv2.rectangle(frame_layer, (0, 0), (frame.shape[1], frame.shape[0]), layer_color, -1)
                    cv2.addWeighted(frame_layer, alpha, frame, 1 - alpha, 0, frame)

                    # emocja tekst
                    (text_width, text_height), _ = cv2.getTextSize(f"{emotion}", font, font_scale, thickness)
                    text_x = 15
                    text_y = text_height + 15

                    # emocja tekst tło
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
                    label = f"pysk psa"
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

        if is_face_detected and is_AU_detection:
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
                        if (emotion == "Radosc" or emotion == "Subtelna radosc") and label not in happpy_labels:
                            continue
                        if (emotion == "Zlosc" or emotion == "Subtelna zlosc") and label not in angry_labels:
                            continue
                        if (emotion == "Smutek" or emotion == "Subtelny smutek") and label not in sad_labels:
                            continue
                        if emotion == "Neutralnosc":
                            continue

                        x1_0, y1_0, x2_0, y2_0 = detection['bbox']
                        dx = np.absolute(x1_0 - x2_0)
                        dy = np.absolute(y1_0 - y2_0)
                        x1 = x1_0 + x1_expanded
                        x2 = x1 + dx
                        y1 = y1_0 + y1_expanded
                        y2 = y1 + dy

                        label_text = f"{au_model.names[label]}"
                        color = class_colors.get(label, (255, 255, 255))
                        font_scale_box = 0.4
                        thickness_box = 1

                        # rysowanie bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale_box, thickness_box)
                        cv2.rectangle(frame, (x1, y1 - text_height - 2), (x1 + text_width, y1 + 2), (0, 0, 0), thickness=cv2.FILLED)
                        cv2.putText(frame, label_text, (x1, y1 - 2), font, font_scale_box, color, thickness_box)
                else:  # inne przypadki
                    detection = detections
                    if (emotion == "Radosc" or emotion == "Subtelna radosc") and label not in happpy_labels:
                        continue
                    if (emotion == "Zlosc" or emotion == "Subtelna zlosc") and label not in angry_labels:
                        continue
                    if (emotion == "Smutek" or emotion == "Subtelny smutek") and label not in sad_labels:
                        continue
                    if emotion == "Neutralnosc":
                        continue

                    x1_0, y1_0, x2_0, y2_0 = detection['bbox']
                    dx = np.absolute(x1_0 - x2_0)
                    dy = np.absolute(y1_0 - y2_0)
                    x1 = x1_0 + x1_expanded
                    x2 = x1 + dx
                    y1 = y1_0 + y1_expanded
                    y2 = y1 + dy

                    label_text = f"{au_model.names[label]}"
                    color = class_colors.get(label, (255, 255, 255))
                    font_scale_box = 0.4
                    thickness_box = 1

                    # rysowanie bbox
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

    emotion_sum_up(emotion_counts, video_path, output_file)


# przetwarzanie filmów z folderu
for video_file in os.listdir(input_folder):
    if video_file.endswith(('.mp4', '.avi', '.mkv')):
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, f"processed_FACS_track_cropeed+_{video_file}")
        process_video(input_path, output_path)

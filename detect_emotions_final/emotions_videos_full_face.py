import cv2
from ultralytics import YOLO
import os
import csv

# wczytanie folderów i modeli
input_folder = 'datasets\\dataset_videos\\sad'
output_file = 'datasets\\dataset_videos\\sad\\results.csv'
output_folder = 'output_videos'
face_model = YOLO('models\\yolov8_dogface_dataset2_80epochs_best.pt')
full_face_model = YOLO('models\\yolov8_full_face_cropped_emotions2.pt')

with open(output_file, mode="w", encoding="utf-8", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)

detection_interval = 10

class_names = {
    0: "Zlosc",
    1: "RFadosc",
    2: "Smutek"
}

emotion_colors = {
    "Radosc": (0, 255, 0),
    "Zlosc": (0, 0, 255),
    "Smutek": (255, 0, 0),
    "Neutralnosc": (128, 128, 128)
}


def emotion_sum_up(emotion_number, emotion_confidence, video_path, output_csv_file):
    total_frames = sum(emotion_number.values())

    sorted_emotions = [
        (emotion, (count / total_frames) * 100, emotion_confidence[emotion] / count if count > 0 else 0)
        for emotion, count in sorted(emotion_number.items(), key=lambda x: x[1], reverse=True)
        if count > 0
    ]

    headers = ["Film"]
    for i in range(len(sorted_emotions)):
        headers.extend([f"Emocja{i+1}", f"% klatek{i+1}", f"Średnia pewność{i+1}"])

    file_exists = os.path.exists(output_csv_file)

    with open(output_csv_file, mode="a", encoding="utf-8", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)

        if not file_exists:
            csv_writer.writerow(headers)

        row = [os.path.basename(video_path)]
        for emotion, percentage, avg_conf in sorted_emotions:
            row.extend([emotion, f"{percentage:.2f}", f"{avg_conf:.2f}"])

        while len(row) < len(headers):
            row.extend(["", "", ""])

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
        "Neutralność": 0,
        "Smutek": 0,
        "Zlosc": 0
    }

    emotion_confidence = {
        "Radosc": 0.0,
        "Neutralnosc": 0.0,
        "Smutek": 0.0,
        "Zlosc": 0.0,
    }

    frame_count = 0
    trackers = []

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

                if confidence > 0.5:  # próg pewności wykrycia pyska
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

                    if confidence > 0.5:  # próg pewności wykrycia pyska
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
                if confidence > 0.5:
                    face_frame = frame[y1_face:y2_face, x1_face:x2_face]

                    if face_frame.size == 0:
                        continue

                    results = full_face_model(face_frame)
                    emotion = class_names.get(results[0].probs.top1)
                    conf_top1_emotion = results[0].probs.top1conf
                    print(f"EMOTION: {emotion} conf: {conf_top1_emotion}\n")

                    if emotion:
                        emotion_counts[emotion] += 1
                        emotion_confidence[emotion] += conf_top1_emotion

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
                    (text_width, text_height), _ = cv2.getTextSize(f"{emotion} {conf_top1_emotion:.2f}", font, font_scale, thickness)
                    text_x = 15
                    text_y = text_height + 15

                    # emocja tekst tło
                    cv2.rectangle(frame, (0, 0), (text_x + text_width + 15, text_y + 15), (0, 0, 0), thickness=cv2.FILLED)
                    cv2.putText(frame, f"{emotion}: {conf_top1_emotion:.2f}", (text_x, text_y), font, font_scale, text_color, thickness)

            # aktualizowanie bboxa
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

        # wyświetlanie i zapis klatki
        cv2.imshow('Frame', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    emotion_sum_up(emotion_counts, emotion_confidence, video_path, output_file)


# przetwarzanie filmów z folderu
for video_file in os.listdir(input_folder):
    if video_file.endswith(('.mp4', '.avi', '.mkv')):
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, f"processed_full_face_track_cropped_{video_file}")
        process_video(input_path, output_path)

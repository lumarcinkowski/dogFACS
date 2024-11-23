import cv2
from ultralytics import YOLO
import os

input_folder = 'datasets\\dataset_videos\\sad'
output_folder = 'output_videos'
model = YOLO('models\\yolov8_full_face_classifier.pt')

output_text_file = os.path.join(input_folder, "wyniki_full_face.txt")

if os.path.exists(output_text_file):
    os.remove(output_text_file)

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


def emotion_sum_up(emotion_number, video_path):
    total_frames = sum(emotion_number.values())

    print(f"Podsumowanie emocji dla {os.path.basename(video_path)}:")

    for emotion, count in emotion_number.items():
        percentage = (count / total_frames) * 100 if total_frames > 0 else 0
        print(f"{emotion}: {count} klatek ({percentage:.2f}%)")

    sorted_emotions = sorted(emotion_number.items(), key=lambda x: x[1], reverse=True)

    dominant_emotion, dominant_count = sorted_emotions[0]
    dominant_percentage = (dominant_count / total_frames) * 100 if total_frames > 0 else 0

    if len(sorted_emotions) > 1:
        second_emotion, second_count = sorted_emotions[1]
        second_percentage = (second_count / total_frames) * 100 if total_frames > 0 else 0
    else:
        second_emotion, second_percentage = None, 0

    print(f"\nDOMINUJĄCA EMOCJA: {dominant_emotion} ({dominant_percentage:.2f}%)")

    if second_emotion:
        print(f"DRUGA EMOCJA: {second_emotion} ({second_percentage:.2f}%)")

    with open(output_text_file, "a", encoding="utf-8") as f:
        f.write(f"{os.path.basename(video_path)} {dominant_emotion} {dominant_percentage:.2f}% ")
        if second_emotion:
            f.write(f"{second_emotion} {second_percentage:.2f}%\n")


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

        results = model(frame)
        emotion = class_names.get(results[0].probs.top1)

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

        cv2.imshow("Podgląd wideo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    emotion_sum_up(emotion_counts, video_path)


for video_file in os.listdir(input_folder):
    if video_file.endswith(('.mp4', '.avi', '.mkv')):
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, f"processed_full_face_{video_file}")
        process_video(input_path, output_path)

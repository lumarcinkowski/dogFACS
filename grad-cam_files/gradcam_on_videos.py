import cv2
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import os
import numpy as np
import tensorflow as tf

# wczytanie folderów i modeli
input_folder = 'datasets\\dataset_videos\\angry'
output_folder = 'output_videos'
face_model = YOLO('models\\yolov8_dogface_dataset2_80epochs_best.pt')
full_face_model_vgg16 = load_model('models\\best_model_60ep.h5')

detection_interval = 10
vgg_img_width, vgg_img_height = 224, 224

base_vgg_model = full_face_model_vgg16.get_layer("vgg16")


def find_last_conv_layer(base_model):
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            print(layer.name)
            return layer.name
    raise ValueError("Model nie zawiera warstw konwolucyjnych.")


last_conv_layer_name = find_last_conv_layer(base_vgg_model)
grad_model = tf.keras.models.Model(
    inputs=[base_vgg_model.input],
    outputs=[base_vgg_model.get_layer(last_conv_layer_name).output, base_vgg_model.output]
    )

class_names = {
    0: "Zlosc",
    1: "Radosc",
    2: "Smutek"
}


def process_image_vgg(img_frame):

    resized_frame = cv2.resize(img_frame, (vgg_img_width, vgg_img_height))
    normalized_frame = resized_frame / 255.0
    processed_frame = np.expand_dims(normalized_frame, axis=0)

    predictions = full_face_model_vgg16.predict(processed_frame)
    predicted_class = np.argmax(predictions, axis=1)
    class_idx = predicted_class.item()

    emotion = class_names.get(class_idx)
    confidence = np.max(predictions)

    #gradCAM
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(processed_frame)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return emotion, confidence, heatmap


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

                    emotion, conf_top1_emotion, heatmap = process_image_vgg(face_frame)
                    print(f"EMOTION: {emotion} conf: {conf_top1_emotion}\n")

                    if emotion:
                        emotion_counts[emotion] += 1
                        emotion_confidence[emotion] += conf_top1_emotion

                    font = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = 1
                    thickness = 1
                    text_color = (169, 169, 169)

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
                    face_frame = frame[y1_face:y2_face, x1_face:x2_face]

                    cv2.rectangle(frame, (x1_face, y1_face), (x2_face, y2_face), (200, 0, 255), 2)

                    print(face_frame.shape[1], face_frame.shape[0])

                    if face_frame.shape[1] != 0 and face_frame.shape[0] != 0:

                        # heatmapa na zdjecie
                        heatmap_resized = cv2.resize(heatmap.numpy(), (face_frame.shape[1], face_frame.shape[0]))
                        heatmap_resized = np.uint8(255 * heatmap_resized)  # Skala od 0 do 255
                        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

                        if face_frame.shape[:2] != heatmap_colored.shape[:2]:
                            heatmap_colored = cv2.resize(heatmap_colored, (face_frame.shape[1], face_frame.shape[0]))

                        if len(heatmap_colored.shape) == 2:
                            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_GRAY2BGR)

                        if len(face_frame.shape) == 2:
                            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_GRAY2BGR)

                        superimposed_face = cv2.addWeighted(face_frame, 0.6, heatmap_colored, 0.4, 0)

                        frame[y1_face:y2_face, x1_face:x2_face] = superimposed_face

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


# przetwarzanie filmów z folderu
for video_file in os.listdir(input_folder):
    if video_file.endswith(('.mp4', '.avi', '.mkv')):
        input_path = os.path.join(input_folder, video_file)
        output_path = os.path.join(output_folder, f"processed_full_face_track_vgg16_testGC_{video_file}")
        process_video(input_path, output_path)

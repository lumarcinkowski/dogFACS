import os
from transformers import ASTForAudioClassification, ASTFeatureExtractor
import torch
import librosa
import numpy as np
import json
from sklearn.metrics import accuracy_score

# Wczytaj plik JSON z mapowaniem id2label
with open('config.json', 'r') as f:
    ontology = json.load(f)

# Inicjalizacja modelu i ekstraktora cech
model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# Funkcja do analizy agresywności szczekania
def is_aggressive_bark(audio, sr, threshold=3, min_energy=0.05, min_rms=0.05, frame_length=1024, hop_length=512):
    # Obliczenie RMS
    stft = np.abs(librosa.stft(audio, n_fft=frame_length, hop_length=hop_length))
    rms = librosa.feature.rms(S=stft, frame_length=frame_length, hop_length=hop_length)[0]

    # Wykrywanie szczytów RMS, gdzie energia jest wyższa od min_rms
    peaks = np.where(rms > min_rms)[0]

    # Jeśli liczba szczytów jest mniejsza niż próg, to nie jest agresywne szczekanie
    if len(peaks) < threshold:
        return False

    # Obliczanie interwałów pomiędzy szczytami w sekundach
    peak_intervals = np.diff(peaks) / sr

    # Sprawdzamy, czy średni interwał pomiędzy szczytami jest mniejszy niż 1/3 sekundy (3 szczeki na sekundę)
    avg_interval = np.mean(peak_intervals) if len(peak_intervals) > 0 else float("inf")
    if avg_interval < (1 / threshold):
        return True

    # Analiza tempa - jeśli tempo jest wysokie (ponad 3 szczeki na sekundę)
    tempogram = librosa.feature.tempogram(y=audio, sr=sr, hop_length=hop_length)
    tempo = np.mean(tempogram)

    # Jeśli tempo przekracza próg, to uznajemy szczekanie za agresywne
    if tempo > threshold:
        return True

    # Jeśli żaden z powyższych warunków nie jest spełniony, uznajemy szczekanie za normalne
    return False

# Funkcja do przewidywania emocji w dźwięku
def predict_emotions(audio_path):
    y, sr = librosa.load(audio_path, sr=16000, duration=10)

    # Przygotowanie spektrogramu dla modelu
    inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt", padding=True)

    # Przewidywanie klas przez model
    logits = model(**inputs).logits

    # Konwersja wyników na prawdopodobieństwa
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Sortowanie wyników według prawdopodobieństw
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Mapowanie emocji na dźwięki
    emotion_graph = {
        "Zlosc": ["Growling", "Fast_bark", "Fast_bow-wow"],
        "Smutek": ["Whimper (dog)", "Yip"],
        "Raczej smutek": ["Howl"]
    }

    # Mapowanie id2label
    id2label = ontology.get('id2label', {})
    if not id2label:
        print("Error: 'id2label' mapping not found in config.json!")
        exit()

    # Analiza wyników modelu
    emotion_results = {}
    for idx, prob in zip(sorted_indices[0][:10], sorted_probs[0][:10]):  # Pobierz top 6 wyników
        class_id = idx.item()
        class_name = id2label.get(str(class_id), "Unknown")

        # Dopasowanie klasy do emocji
        matched = False
        for emotion, sounds in emotion_graph.items():
            if class_name in sounds:
                emotion_results[emotion] = emotion_results.get(emotion, 0) + prob.item()
                matched = True
                break

        # Jeśli dźwięk nie pasuje do żadnej emocji, nic nie rób
        if not matched:
            continue

    return emotion_results

# Funkcja do testowania na zbiorze testowym
def test_model(test_data):
    predicted_labels = { "Raczej smutek": 0, "Smutek": 0, "Zlosc": 0 }
    total_counts = { "Raczej smutek": 0, "Smutek": 0, "Zlosc": 0 }

    for data in test_data:
        audio_path = data['audio_path']
        true_emotion = data['emotion']

        # Przewidywanie emocji
        predicted_emotions = predict_emotions(audio_path)
        if predicted_emotions:
            predicted_emotion = max(predicted_emotions, key=predicted_emotions.get)
        else:
            predicted_emotion = "Unknown"

        if predicted_emotion == true_emotion:
            predicted_labels[true_emotion] += 1
        total_counts[true_emotion] += 1

    # Oblicz dokładność dla każdej emocji
    accuracy_per_emotion = {}
    for emotion in predicted_labels:
        accuracy_per_emotion[emotion] = predicted_labels[emotion] / total_counts[emotion] if total_counts[emotion] > 0 else 0

    return accuracy_per_emotion

# Ścieżka do folderu Eval
eval_folder = 'Eval'

# Mapowanie emocji na foldery
emotion_folders = {
    "Raczej smutek": "raczej_smutny",
    "Smutek": "smutny",
    "Zlosc": "zlosc"
}

# Wczytanie plików audio z folderów
test_data = []
for emotion, folder in emotion_folders.items():
    folder_path = os.path.join(eval_folder, folder)
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            audio_path = os.path.join(folder_path, filename)
            test_data.append({"audio_path": audio_path, "emotion": emotion})

# Testowanie modelu
accuracy_per_emotion = test_model(test_data)

# Wyświetlanie dokładności dla każdej emocji
print("Accuracy per Emotion:")
for emotion, accuracy in accuracy_per_emotion.items():
    print(f"{emotion}: {accuracy * 100:.2f}%")

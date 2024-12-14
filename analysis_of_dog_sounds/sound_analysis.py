import os
from transformers import ASTForAudioClassification, ASTFeatureExtractor
import torch
import librosa
import numpy as np
import json

# Ustawienie zmiennej środowiskowej, aby uniknąć problemów z OpenMP jeśli jest to koniczne
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Wczytaj plik JSON z mapowaniem id2label
with open('config.json', 'r') as f:
    ontology = json.load(f)

# Inicjalizacja modelu i ekstraktora cech
model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

# Funkcja do przewidywania emocji w dźwięku
def predict_emotions(audio, sr):
    # Przygotowanie spektrogramu dla modelu
    inputs = feature_extractor(audio, sampling_rate=sr, return_tensors="pt", padding=True)

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
        "Raczej smutek": ["Howl"],
    }

    # Mapowanie id2label
    id2label = ontology.get('id2label', {})
    if not id2label:
        print("Error: 'id2label' mapping not found in config.json!")
        exit()

    # Analiza wyników modelu
    emotion_results = {}
    total_prob = 0
    detected_sounds = []  # Lista dźwięków wykrytych przez model
    for idx, prob in zip(sorted_indices[0][:10], sorted_probs[0][:10]):  # Pobierz top 10 wyników
        class_id = idx.item()
        class_name = id2label.get(str(class_id), "Unknown")

        # Dopasowanie klasy do emocji
        matched = False
        for emotion, sounds in emotion_graph.items():
            if class_name in sounds:
                emotion_results[emotion] = emotion_results.get(emotion, 0) + prob.item()
                matched = True
                break

        # Dodaj wykryty dźwięk do listy
        if class_name in ["Barking", "Bow-wow"]:
            detected_sounds.append(class_name)

        # Jeśli dźwięk nie pasuje do żadnej emocji, nic nie rób
        if not matched:
            continue

        total_prob += prob.item()

    return emotion_results, total_prob, detected_sounds

# Funkcja do wykrywania agresywnego szczekania
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

# Funkcja do przetwarzania pliku audio i wypisywania emocji dla 10-sekundowych fragmentów
def process_audio_in_chunks(audio_path, chunk_duration=10, min_chunk_duration=2):
    # Wczytanie audio
    y, sr = librosa.load(audio_path, sr=16000)

    # Obliczenie liczby próbek na jeden fragment
    chunk_samples = chunk_duration * sr

    # Podziel audio na fragmenty i analizuj każdy z nich
    total_samples = len(y)
    for start_sample in range(0, total_samples, chunk_samples):
        end_sample = min(start_sample + chunk_samples, total_samples)  # Zapewnia, że nie wyjdziemy poza zakres

        # Wydzielenie fragmentu audio
        audio_chunk = y[start_sample:end_sample]

        # Jeśli długość fragmentu jest mniejsza niż minimalny próg (np. 2 sekundy), pomiń go
        if (end_sample - start_sample) / sr < min_chunk_duration:
            continue

        # Przewidywanie emocji i dźwięków wykrytych przez model
        predicted_emotions, total_prob, detected_sounds = predict_emotions(audio_chunk, sr)

        # Sprawdzanie, czy wykryto dźwięki "Barking" lub "Bow-wow"
        if any(sound in detected_sounds for sound in ["Barking", "Bow-wow"]):
            # Wykrywanie agresywnego szczekania tylko w przypadku wykrycia "Barking" lub "Bow-wow"
            if is_aggressive_bark(audio_chunk, sr):
                print(f"Fragment {start_sample / sr:.2f} - {end_sample / sr:.2f} s: Agresywne szczekanie wykryte!")

        # Jeśli brak wykrytych emocji, pomiń ten fragment
        if not predicted_emotions:
            continue

        # Oblicz procentowy udział każdej emocji
        emotion_percentages = {}
        if total_prob > 0:
            for emotion, prob in predicted_emotions.items():
                emotion_percentages[emotion] = (prob / total_prob) * 100
        else:
            emotion_percentages = {emotion: 0 for emotion in predicted_emotions}

        # Wypisanie wyników w konsoli
        chunk_index = start_sample // chunk_samples + 1
        print(f"Fragment {chunk_index} ({start_sample / sr:.2f} - {end_sample / sr:.2f} s):")
        for emotion, percentage in emotion_percentages.items():
            print(f"  {emotion}: {percentage:.2f}%")

# Testowanie na danym pliku audio
audio_path = 'Eval\\smutny\\Cute Siberian Husky Puppies Playing_9_20.wav'  # Zmienna na ścieżkę do pliku audio
process_audio_in_chunks(audio_path)

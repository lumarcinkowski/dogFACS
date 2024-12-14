
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Wczytanie modelu
model_path ='best_model_60ep.h5'
model = load_model(model_path)

# Dostęp do modelu bazowego (VGG16)
base_model = model.get_layer("vgg16")


# Funkcja do wyszukiwania ostatniej warstwy konwolucyjnej
def find_last_conv_layer(base_model):
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Model nie zawiera warstw konwolucyjnych.")


# Znalezienie ostatniej warstwy konwolucyjnej w bazowym modelu VGG16
last_conv_layer_name = find_last_conv_layer(base_model)
print("Ostatnia warstwa konwolucyjna:", last_conv_layer_name)

# Ścieżka do folderu z obrazami "happy"
folder_path = 'sad'  # Zmień na swoją ścieżkę
output_folder = 'gradcam_happy_4'  # Folder do zapisywania wyników
os.makedirs(output_folder, exist_ok=True)


# Funkcja do przetwarzania obrazu i generowania Grad-CAM
def generate_gradcam(img_path, model, grad_model, class_idx, last_conv_layer_name):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizacja

    # Predykcja
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])

    # Obliczenie Grad-CAM
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    # Nakładanie mapy cieplnej na obraz
    original_img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap.numpy(), (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    return original_img, heatmap, superimposed_img


# Tworzenie modelu Grad-CAM
grad_model = tf.keras.models.Model(
    inputs=[base_model.input],
    outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
)

# Iteracja przez obrazy w wybranym przez użytkownika folderze 
for img_name in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_name)
    if not img_name.lower().endswith(('png', 'jpg', 'jpeg')):
        continue  # Pomijanie nieobrazowych plików

    original_img, heatmap, superimposed_img = generate_gradcam(
        img_path, model, grad_model, class_idx=0, last_conv_layer_name=last_conv_layer_name
    )

    # Zapisywanie wyników oryginalny obraz, mapa cieplna, obraz z naniesiona mapa cieplna
    output_path_original = os.path.join(output_folder, f"original_{img_name}")
    output_path_heatmap = os.path.join(output_folder, f"heatmap_{img_name}")
    output_path_superimposed = os.path.join(output_folder, f"superimposed_{img_name}")

    cv2.imwrite(output_path_original, original_img)
    cv2.imwrite(output_path_heatmap, heatmap)
    cv2.imwrite(output_path_superimposed, superimposed_img)

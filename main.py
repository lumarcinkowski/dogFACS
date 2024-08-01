# Importowanie bibliotek
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Ścieżka do rozpakowanych danych
data_dir = 'Dog Emotion'

# Przygotowanie generatora danych z bardziej zaawansowaną augmentacją
datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Ładowanie pretrenowanego modelu EfficientNetB0
base_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Dodanie własnych warstw
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)  # Dodanie warstwy Dropout
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Zablokowanie większości warstw bazowego modelu, ale odblokowanie kilku do trenowania
for layer in base_model.layers[:100]:
    layer.trainable = False
for layer in base_model.layers[100:]:
    layer.trainable = True

# Kompilacja modelu z mniejszym współczynnikiem uczenia się
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=100 # Zwiększona liczba epok
)

# Function to convert tensors to numpy arrays before saving the model
def convert_tensors_to_numpy(model):
    for layer in model.layers:
        for attr, value in layer.__dict__.items():
            if isinstance(value, tf.Tensor):
                layer.__dict__[attr] = value.numpy()

# Convert tensors in the model to numpy arrays
convert_tensors_to_numpy(model)

# Zapisanie wytrenowanego modelu
model.save('dog_emotion_model.h5')

# Ładowanie obrazu i jego przetwarzanie
def get_img_array(img_path, size):
    img = image.load_img(img_path, target_size=size)
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(array)

# Tworzenie mapy cieplnej Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Wyświetlanie obrazu z nałożoną mapą cieplną
def display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = image.load_img(img_path)
    img = image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = image.array_to_img(superimposed_img)

    superimposed_img.save(cam_path)

    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()

# Testowanie modelu i generowanie wizualizacji
img_path = '94752989_02f166d27d_b.jpg'  # Zastąp odpowiednią ścieżką
img_array = get_img_array(img_path, size=(224, 224))
preds = model.predict(img_array)
print("Predicted class:", np.argmax(preds))

heatmap = make_gradcam_heatmap(img_array, model, 'top_conv')  # Sprawdź poprawną nazwę warstwy
display_gradcam(img_path, heatmap)

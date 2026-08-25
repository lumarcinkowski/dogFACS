# Dog Emotion Detection

A project focused on automatic dog emotion detection and analysis using **video, image, and audio data**.

The project includes several approaches to emotion recognition, including facial analysis, **DogFACS**, **YOLO**, **VGG16**, **Grad-CAM**, and audio classification using a pretrained **Audio Spectrogram Transformer (AST)** model.

## Project Overview

The project contains programs for:

- detecting dog emotions in videos based on a cropped region of the dog's face,
- detecting dog emotions using **DogFACS Action Units and descriptors**,
- visualizing **Grad-CAM** explanations on videos,
- visualizing **Grad-CAM** explanations on images,
- analyzing dog sounds using a pretrained audio classification model.

Due to the size of some datasets and models, larger files are stored on Google Drive. A link to the additional materials is provided in `materialy.txt`.

---

## Requirements

The project was developed and tested using:

- **Python 3.10**

### Required libraries

Install the required dependencies using:

```bash
pip install ultralytics
pip install opencv-contrib-python
pip install tensorflow
pip install transformers
pip install librosa
```

The project also uses libraries such as:

- NumPy
- OpenCV
- Matplotlib
- Ultralytics
- TensorFlow / Keras
- Transformers
- Librosa
- scikit-learn

Some of these packages are installed automatically as dependencies of the packages listed above.

---

# 1. Dog Emotion Detection in Videos

The project provides two different approaches for detecting dog emotions in videos.

### DogFACS-based emotion detection

The following script is used for emotion detection based on **DogFACS**:

```text
emotions_videos_DogFACS.py
```

### Facial-region-based emotion detection

The following script is used for emotion detection based on a cropped region of the dog's face:

```text
emotions_videos_full_face.py
```

Both scripts are located in:

```text
detect_emotions_final/
```

## Configuration

Before running either program, configure the variables defined at the beginning of the script.

### Common parameters

```python
input_folder = "path/to/input/videos"
csv_file = "path/to/results.csv"
output_folder = "path/to/output"
face_model = "path/to/yolov8_dogface_final.pt"
```

### `input_folder`

Path to the folder containing the videos to be analyzed.

### `csv_file`

Path to the CSV file where the analysis results will be saved.

If the file does not exist, it will be created automatically.

### `output_folder`

Path to the folder where processed videos with visualizations will be saved.

### `face_model`

Path to the YOLO model used for detecting the dog's face.

---

## DogFACS configuration

For:

```text
emotions_videos_DogFACS.py
```

an additional parameter must be configured:

```python
au_model = "path/to/yolov8_FACS.pt"
```

This model is responsible for detecting **Action Units and descriptors** used in the DogFACS-based analysis.

---

## Facial emotion classification configuration

For:

```text
emotions_videos_full_face.py
```

the following parameter must be configured:

```python
full_face_model = "path/to/yolov8_emotion_classification.pt"
```

This model is used to classify emotions based on the cropped region of the dog's face.

---

## Available YOLO Models

The YOLO models are located in:

```text
models/
```

| Model | Purpose |
|---|---|
| `yolov8_dogface_final.pt` | Dog face detection |
| `yolov8_emotion_classification.pt` | Emotion classification based on the dog's face |
| `yolov8_FACS.pt` | DogFACS Action Unit and descriptor detection |

### Running the programs

For DogFACS-based detection:

```bash
python emotions_videos_DogFACS.py
```

For facial emotion classification:

```bash
python emotions_videos_full_face.py
```

During processing, the currently analyzed video is displayed together with the generated visualizations.

After processing all videos:

- the analysis results are saved to the specified CSV file,
- processed videos are saved to the specified output directory.

---

# 2. Grad-CAM Visualization on Videos

The project includes a script for visualizing the regions of video frames that are considered important by the emotion classification model using **Grad-CAM**.

The script is:

```text
gradcam_on_videos.py
```

and is located in:

```text
grad-cam_files/
```

## Configuration

The input and output paths should be configured in the same way as for the emotion detection scripts.

An additional parameter is required for the VGG16 model:

```python
full_face_model_vgg16 = "path/to/vgg16_emotion_classification.h5"
```

The model:

```text
vgg16_emotion_classification.h5
```

is used for emotion classification and Grad-CAM visualization.

The model is available on Google Drive. The link to the additional materials is provided in:

```text
materialy.txt
```

### Running the script

```bash
python gradcam_on_videos.py
```

During execution, the currently processed video is displayed together with the Grad-CAM visualization.

After processing, the resulting videos are saved to the specified output directory.

Unlike the emotion detection scripts, this program does not generate a CSV file.

---

# 3. Grad-CAM Visualization on Images

The project also includes a program for generating **Grad-CAM visualizations for individual images**.

The program requires the following VGG16 model:

```text
vgg16_emotion_classification.h5
```

The input folder should contain images in one of the following formats:

```text
.jpg
.jpeg
.png
```

## Configuration

Set the following parameters before running the script:

```python
model_path = "path/to/vgg16_emotion_classification.h5"
folder_path = "path/to/images"
output_folder = "path/to/output_folder"
```

### `model_path`

Path to the VGG16 emotion classification model.

### `folder_path`

Path to the folder containing the input images.

### `output_folder`

Path to the folder where the generated results will be saved.

After configuring the paths, run the script.

The program generates three outputs for each input image:

1. **Original image** – the original image without modifications.
2. **Heatmap** – a visualization of the image regions considered important by the model.
3. **Superimposed image** – the original image with the semi-transparent Grad-CAM heatmap overlaid.

Example output filenames:

```text
original_image1.jpg
heatmap_image1.jpg
superimposed_image1.jpg
```

---

# 4. Audio Analysis

The project also includes a program for analyzing dog sounds.

Audio classification is performed using a pretrained **Audio Spectrogram Transformer (AST)** model available through Hugging Face Transformers:

```text
MIT/ast-finetuned-audioset-10-10-0.4593
```

The model and feature extractor are loaded using:

```python
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)

feature_extractor = ASTFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593"
)
```

## `config.json`

The audio analysis script also requires:

```text
config.json
```

This file contains the `id2label` mapping used to convert the numerical audio class IDs into human-readable labels.

The file must be located in the same directory as the audio analysis script.

Without this file, the audio classes cannot be mapped correctly to their corresponding labels.

## Selecting an audio file

Before running the program, specify the path to the audio file:

```python
audio_path = "Eval/smutny/Cute Siberian Husky Puppies Playing_9_20.wav"
```

Then run the script.

The audio is analyzed in **10-second segments**. The classification results for each segment are displayed in the console.

Potentially aggressive barking is additionally detected and reported by the program.

---

# Project Structure

The main project structure is organized as follows:

```text
project/
│
├── detect_emotions_final/
│   ├── emotions_videos_DogFACS.py
│   └── emotions_videos_full_face.py
│
├── grad-cam_files/
│   ├── gradcam_on_videos.py
│   └── ...
│
├── models/
│   ├── yolov8_dogface_final.pt
│   ├── yolov8_emotion_classification.pt
│   └── yolov8_FACS.pt
│
├── config.json
├── materialy.txt
└── README.md
```

The following model is stored externally due to its size:

```text
vgg16_emotion_classification.h5
```

Additional test materials, including categorized dog videos and the audio dataset, are also available on Google Drive. The link can be found in:

```text
materialy.txt
```

---

# Quick Start

## 1. Install Python

The recommended Python version is:

```text
Python 3.10
```

Check your installed version:

```bash
python --version
```

## 2. Install dependencies

```bash
pip install ultralytics
pip install opencv-contrib-python
pip install tensorflow
pip install transformers
pip install librosa
```

## 3. Download additional materials

Download the additional models and test data from Google Drive using the link provided in:

```text
materialy.txt
```

## 4. Configure the paths

Before running a program, configure the paths to:

- input videos or images,
- YOLO models,
- VGG16 model,
- output directories,
- CSV files,
- audio files.

## 5. Run the desired program

For DogFACS-based emotion detection:

```bash
python emotions_videos_DogFACS.py
```

For facial emotion classification:

```bash
python emotions_videos_full_face.py
```

For Grad-CAM visualization on videos:

```bash
python gradcam_on_videos.py
```

---

## Notes

- The project was developed using **Python 3.10**.
- Larger datasets and models are stored externally due to their size.
- Additional materials can be accessed through the Google Drive link provided in `materialy.txt`.
- The VGG16 model `vgg16_emotion_classification.h5` must be downloaded separately when using the Grad-CAM functionality.
- Processing time for videos depends on the available hardware and video resolution.

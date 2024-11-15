from ultralytics import YOLO
import torch


def train(data, num_epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = YOLO('yolov8n-cls.pt')

    model.train(
        data=data,
        epochs=num_epochs,
        batch=32,
        lr0=0.015,  # Initial learning rate
        lrf=0.15,   # Final OneCycleLR learning rate (lr0 * lrf)
        momentum=0.9,  # SGD momentum
        weight_decay=0.0005  # Optimizer weight decay
    )


if __name__ == '__main__':
    data1 = 'datasets\\dataset_full_face'
    train(data1, 200)


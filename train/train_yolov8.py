import torch
from ultralytics import YOLO


def train(data, num_epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = YOLO('yolov8n.pt').to(device)

    model.train(
        data=data,
        epochs=num_epochs,
        lr0=0.015,
        lrf=0.15,  # ostatecznie lr to (lr0 * lrf)
        momentum=0.9,
        weight_decay=0.0005
    )


if __name__ == '__main__':
    data1 = 'train\\data1.yaml'
    train(data1, 80)

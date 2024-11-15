import torch
import torchvision
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
import os
import xml.etree.ElementTree as ET
from PIL import Image
from functools import partial
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from src.collate_fn import collate_fn
import csv


class DogFaceDataset(Dataset):
    def __init__(self, root, dataset_type='train', transforms=None):
        self.root = root
        self.dataset_type = dataset_type
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, f"{dataset_type}\\images"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, f"{dataset_type}\\annotations"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, f"{self.dataset_type}\\images", self.imgs[idx])
        ann_path = os.path.join(self.root, f"{self.dataset_type}\\annotations", self.annotations[idx])

        img = Image.open(img_path).convert("RGB")

        tree = ET.parse(ann_path)
        root = tree.getroot()

        boxes = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms:
            img, target = self.transforms(img, target)

        img = F.to_tensor(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model(num_classes):

    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights="DEFAULT")
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )

    return model


def validate(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    tp, fp, tn, fn = 0, 0, 0, 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for i, output in enumerate(outputs):
                pred_boxes = output['boxes']
                true_boxes = targets[i]['boxes'].to(device)

                if len(pred_boxes) == 0 and len(true_boxes) > 0:
                    fn += len(true_boxes)
                elif len(pred_boxes) > 0 and len(true_boxes) == 0:
                    fp += len(pred_boxes)
                else:
                    matched_pred = set()
                    matched_true = set()

                    for true_idx, true in enumerate(true_boxes):
                        ious = torchvision.ops.box_iou(true.unsqueeze(0), pred_boxes)
                        max_iou, pred_idx = ious.max(dim=1)

                        if max_iou > iou_threshold:
                            tp += 1
                            matched_pred.add(pred_idx.item())
                            matched_true.add(true_idx)
                        else:
                            fn += 1

                    for pred_idx in range(len(pred_boxes)):
                        if pred_idx not in matched_pred:
                            fp += 1

                    for true_idx in range(len(true_boxes)):
                        if true_idx not in matched_true:
                            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1_score, fp, fn


def train(dataset, val_dataset, num_epochs, path_best, path_last, csv_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = get_model(num_classes=2)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    best_f1 = float('-inf')

    avg_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_false_positives = []
    val_false_negatives = []
    val_f1_scores = []

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Avg Training Loss", "Validation Accuracy", "Precision", "Recall", "F1", "False Positives", "False Negatives"])

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        i = 0

        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()

            if i % 1 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {losses.item():.4f}")
            i += 1

        avg_epoch_loss = epoch_loss / len(data_loader)
        avg_losses.append(avg_epoch_loss)

        val_accuracy, val_precision, val_recall, val_f1_score, false_positive, false_negative = validate(model, val_loader, device)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1_scores.append(val_f1_score)
        val_false_positives.append(false_positive)
        val_false_negatives.append(false_negative)

        print(f"Epoch [{epoch+1}/{num_epochs}] Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, "
              f"Recall: {val_recall:.4f}, F1: {val_f1_score:.4f}, FP: {false_positive}, FN: {false_negative}")

        if val_f1_score > best_f1:
            best_f1 = val_f1_score
            torch.save(model.state_dict(), path_best)
            print(f"Best model saved with F1: {best_f1:.4f}")

        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_epoch_loss, val_accuracy, val_precision, val_recall, val_f1_score, false_positive, false_negative])

    torch.save(model.state_dict(), path_last)


if __name__ == "__main__":

    dataset1 = DogFaceDataset('datasets\\dataset1', dataset_type='train', transforms=None)
    dataset1_val = DogFaceDataset('datasets\\dataset1', dataset_type='val', transforms=None)
    dataset2 = DogFaceDataset('datasets\\dataset2', dataset_type='train', transforms=None)
    dataset2_val = DogFaceDataset('datasets\\dataset2', dataset_type='val', transforms=None)

    model1_path_best = 'models\\retinaNet_dogface_dataset1_80epochs_best.pth'
    model1_path_last = 'models\\retinaNet_dogface_dataset1_80epochs_last.pth'

    model2_path_best = 'models\\retinaNet_dogface_dataset2_80epochs_best.pth'
    model2_path_last = 'models\\retinaNet_dogface_dataset2_80epochs_last.pth'

    csv_model1 = 'retinaNet_dogface_dataset1_80epochs_statistics.csv'
    csv_model2 = 'retinaNet_dogface_dataset2_80epochs_statistics.csv'

    train(dataset1, dataset1_val, 80, model1_path_best, model1_path_last, csv_model1)
    train(dataset2, dataset2_val, 80, model2_path_best, model2_path_last, csv_model2)

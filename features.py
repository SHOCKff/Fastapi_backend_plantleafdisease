import os
import uuid
from fastapi import UploadFile

TEMP_DIR = "temp"


def save_upload_file(file: UploadFile) -> str:
    """
    Save uploaded image into temp folder.
    Returns saved file path.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)

    # unique filename to avoid collision
    ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"

    file_path = os.path.join(TEMP_DIR, filename)

    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    return file_path

import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import models, transforms
from skimage.feature import local_binary_pattern

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# AlexNet feature extractor
alexnet = models.alexnet(weights="AlexNet_Weights.IMAGENET1K_V1")
alexnet.classifier = torch.nn.Sequential(
    *list(alexnet.classifier.children())[:-1]
)
alexnet.eval()

# ResNet18 feature extractor
resnet18 = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1])
resnet18.eval()

def segment_leaf_kmeans(image_path: str):
    img = cv2.cvtColor(
        np.array(Image.open(image_path).convert("RGB")),
        cv2.COLOR_RGB2BGR
    )

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape((-1, 3)).astype(np.float32)

    _, labels, _ = cv2.kmeans(
        pixels,
        2,
        None,
        (cv2.TERM_CRITERIA_EPS +
         cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    labels = labels.reshape(lab.shape[:2])
    mask = np.uint8(
        labels == np.argmax(np.bincount(labels.flatten()))
    ) * 255

    return cv2.bitwise_and(img, img, mask=mask)

def extract_alexnet_features(image_path: str):
    img = Image.open(image_path).convert("RGB")
    t = transform(img).unsqueeze(0)
    with torch.no_grad():
        f = alexnet(t)
    return f.squeeze().numpy()


def extract_resnet18_features(image_path: str):
    img = Image.open(image_path).convert("RGB")
    t = transform(img).unsqueeze(0)
    with torch.no_grad():
        f = resnet18(t)
    return f.squeeze().numpy()

def extract_lbp_features(segmented_img):
    gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, "uniform")

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=int(lbp.max() + 1),
        range=(0, int(lbp.max() + 1))
    )

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def get_fused_features(image_path: str):
    segmented = segment_leaf_kmeans(image_path)

    alex = extract_alexnet_features(image_path)
    res = extract_resnet18_features(image_path)
    lbp = extract_lbp_features(segmented)

    fused = np.concatenate([alex, res, lbp])

    return fused

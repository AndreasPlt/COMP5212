from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import random
import torchvision

from src.models.model import get_model
import src.kaggle50k_dataset as kaggle50k_dataset
from src.test import test

MODEL_PATH = "checkpoints/NEW_model_ncg43.hpc.itc.rwth-aachen.de_20231208-201848_mobilenet_v3_large_freezed_0_unfreezed_20_epochs.pth"
TSV_PATH = "data/kaggle_dataset/train.tsv"
TEST_TSV_PATH = "data/kaggle_dataset/test.tsv"
IMAGE_PATH = "notebooks/demo_images/1.png"
BATCH_SIZE = 32

def load_model(model_path):
    state_dict = torch.load(model_path, map_location=torch.device('cuda'))
    if type(state_dict) == dict:
        model_state_dict = state_dict["model_state_dict"]
        idx_to_label = state_dict["idx_to_labels"]
        config = state_dict["config"]
    else:
        model_state_dict = state_dict
        idx_to_label = get_idx_to_label(TSV_PATH)
        config = None
    model = get_model(model_name=config["model"]["name"], num_classes=56)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model, idx_to_label, config

def get_idx_to_label(tsv_path):
    labels = []
    tsv_file = open(tsv_path, "r")

    for line in tsv_file:
        if line in ['\n', '\r\n']:
            continue
        label, _ = os.path.split(line.strip())
        labels.append(label)

    idx_to_label = {idx: label for idx, label in enumerate(set(labels))}
    return idx_to_label

def load_img(image_path):
    transform = transforms.Compose([
        transforms.Resize((1536, 662)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = Image.open(image_path).convert("RGB")
    image_transformed = transform(image)
    return image_transformed

def get_random_image(dataset):
    idx = np.random.randint(0, len(dataset))
    image, label = dataset[idx]
    return image, label

def main():
    model, idx_to_label, config = load_model(MODEL_PATH)
    print(f"Loaded model with config: {config}")
    label_to_idx = {label: idx for idx, label in idx_to_label.items()}

    base_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    blur_transforms = transforms.Compose([
        base_transforms,
        transforms.GaussianBlur(kernel_size=5)
    ])

    illumination_transforms = transforms.Compose([
        base_transforms,
        transforms.ColorJitter(brightness=(0.3,0.6))
    ])

    occlusion_transforms = transforms.Compose([
        base_transforms,
        transforms.RandomErasing(p=1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
    ])

    base_dataset = kaggle50k_dataset.Kaggle50K(TEST_TSV_PATH, base_transforms)
    blur_dataset = kaggle50k_dataset.Kaggle50K(TEST_TSV_PATH, blur_transforms)
    illumination_dataset = kaggle50k_dataset.Kaggle50K(TEST_TSV_PATH, illumination_transforms)
    occlusion_dataset = kaggle50k_dataset.Kaggle50K(TEST_TSV_PATH, occlusion_transforms)

    print(f"Loaded datasets with {len(base_dataset)} images each")

    base_dataset.labels_to_idx = label_to_idx
    blur_dataset.labels_to_idx = label_to_idx
    illumination_dataset.labels_to_idx = label_to_idx
    occlusion_dataset.labels_to_idx = label_to_idx

    base_loader = torch.utils.data.DataLoader(base_dataset, batch_size=BATCH_SIZE, shuffle=False)
    blur_loader = torch.utils.data.DataLoader(blur_dataset, batch_size=BATCH_SIZE, shuffle=False)
    illumination_loader = torch.utils.data.DataLoader(illumination_dataset, batch_size=BATCH_SIZE, shuffle=False)
    occlusion_loader = torch.utils.data.DataLoader(occlusion_dataset, batch_size=BATCH_SIZE, shuffle=False)

    top_k = [1, 3, 5]

    print(f"Starting testing...")
    base_accuracies = test(model, base_loader, top_k)
    print(f"Calculated base accuracies: {base_accuracies}")
    blur_accuracies = test(model, blur_loader, top_k)
    print(f"Calculated blur accuracies: {blur_accuracies}")
    illumination_accuracies = test(model, illumination_loader, top_k)
    print(f"Calculated illumination accuracies: {illumination_accuracies}")
    occlusion_accuracies = test(model, occlusion_loader, top_k)
    print(f"Calculated occlusion accuracies: {occlusion_accuracies}")

    print(f"\n--------\n Final accuracies:")
    print("Base accuracies: ", base_accuracies)
    print("Blur accuracies: ", blur_accuracies)
    print("Illumination accuracies: ", illumination_accuracies)
    print("Occlusion accuracies: ", occlusion_accuracies)

if __name__ == "__main__":
    main()
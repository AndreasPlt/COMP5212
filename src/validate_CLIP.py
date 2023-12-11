 
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import yaml

from main import parse_args, get_dataloader, write_manifest
from util.filter_countries import filter_countries
import kaggle50k_dataset

model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

args = parse_args()
config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)

valid_countries = filter_countries(config["data"]["min_images"], config["data"]["dir"])
labels = valid_countries

# write tsv files
train_out, dev_out, test_out = write_manifest(config, valid_countries)
config['data']['train_manifest'] = train_out
config['data']['dev_manifest'] = dev_out
config['data']['test_manifest'] = test_out

top_k = [1,3,5]

def classify(image, transforms):
    inputs = processor(text=valid_countries, images=image, return_tensors="pt", padding=True)
    inputs['pixel_values'] = inputs['pixel_values'].to(device)

    # do the transformations on the input tensors
    if transforms != None:
        inputs["pixel_values"] = transforms(inputs["pixel_values"])

    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    prediction = logits_per_image.softmax(dim=1)
    confidences = {labels[i]: float(prediction[0][i].item()) for i in range(len(labels))}
    return confidences

def test(dataset, inv_map, transforms):
    total = 0.
    correct = 0.

    correct = len(top_k) * [0.]

    for i in range(len(dataset)):
        image = dataset[i][0]
        label = inv_map[dataset[i][1].item()][20:]
        pred = sorted(classify(image, transforms).items(), key=lambda item: item[1], reverse=True)

        print("------------------------------------------------------")
        print("prediction: " +str(pred[0][0]))
        print("true label: " + label)

        for j, k in enumerate(top_k):
            if label in [x[0] for x in pred[:k]]:
                correct[j] += 1
        total += 1

        for j, k in enumerate(top_k):
            print(f'Accuracy of the model on the test images (top {k}): {((correct[j] / total)*100):.2f}%')

        print("current progress: " + str(i+1) + "/" + str(len(dataset)))

    return [x / total for x in correct]

# force batch size 1
config["training"]["batch_size"] = 1

blur_transform = transforms.GaussianBlur(kernel_size=(5), sigma=(0.1, 5))
occlusion_transform = transforms.RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
illumination_transforms = transforms.ColorJitter(brightness=(0.3,0.6))

base_dataset = kaggle50k_dataset.Kaggle50K(config["data"]["test_manifest"], transform=None)
blur_dataset = kaggle50k_dataset.Kaggle50K(config["data"]["test_manifest"], transform=None)
occlusion_dataset = kaggle50k_dataset.Kaggle50K(config["data"]["test_manifest"], transform=None)
illumination_dataset = kaggle50k_dataset.Kaggle50K(config["data"]["test_manifest"], transform=None)

base_inv_map = {v: k for k, v in base_dataset.labels_to_idx.items()}
blur_inv_map = {v: k for k, v in blur_dataset.labels_to_idx.items()}
occlusion_inv_map = {v: k for k, v in occlusion_dataset.labels_to_idx.items()}
illumination_inv_map = {v: k for k, v in illumination_dataset.labels_to_idx.items()}

base_accuracies = test(base_dataset, base_inv_map, None)
blur_accuracies = test(blur_dataset, blur_inv_map, blur_transform)
occlusion_accuracies = test(occlusion_dataset, occlusion_inv_map, occlusion_transform)
illumination_accuracies = test(illumination_dataset, illumination_inv_map, illumination_transforms)

print("Base accuracies: " + str(base_accuracies))
print("Blur accuracies: " + str(blur_accuracies))
print("Occlusion accuracies: " + str(occlusion_accuracies))
print("Illumination accuracies: " + str(illumination_accuracies))

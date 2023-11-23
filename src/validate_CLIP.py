 
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

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

def classify(image):
    inputs = processor(text=valid_countries, images=image, return_tensors="pt", padding=True)
    inputs['pixel_values'] = inputs['pixel_values'].to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    prediction = logits_per_image.softmax(dim=1)
    confidences = {labels[i]: float(prediction[0][i].item()) for i in range(len(labels))}
    return confidences

# force batch size 1
config["training"]["batch_size"] = 1
dataset = kaggle50k_dataset.Kaggle50K(config["data"]["dev_manifest"], transform=None)
inv_map = {v: k for k, v in dataset.labels_to_idx.items()}

total = 0.
correct = 0.

top_k = [1,3,5]
correct = len(top_k) * [0.]

for i in range(len(dataset)):
    image = dataset[i][0]
    label = inv_map[dataset[i][1].item()][20:]
    pred = sorted(classify(image).items(), key=lambda item: item[1], reverse=True)

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

print("##########################")

for i, k in enumerate(top_k):
    print(f'Accuracy of the model on the test images (top {k}): {((correct[i] / total)*100):.2f}%')
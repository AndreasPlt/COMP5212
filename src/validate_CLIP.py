 
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
    inputs['pixel_values'].to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    prediction = logits_per_image.softmax(dim=1)
    confidences = {labels[i]: float(prediction[0][i].item()) for i in range(len(labels))}
    return confidences

#pred = sorted(classified.items(), key=lambda item: item[1])[0]

# force batch size 1
config["training"]["batch_size"] = 1
dataset = kaggle50k_dataset.Kaggle50K(config["data"]["dev_manifest"], transform=None)
inv_map = {v: k for k, v in dataset.labels_to_idx.items()}

total = 0.
correct = 0.
for i in range(len(dataset)):
    image = dataset[i][0]
    label =  dataset[i][1]
    pred = sorted(classify(image).items(), key=lambda item: item[1])[0]
    #print("data/kaggle_dataset/" +str(pred[0]))
    #print(inv_map[label.item()])
    correct += (("data/kaggle_dataset/" +str(pred[0])) == inv_map[label.item()])
    total += 1
    print(correct/total)

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Kaggle50K(Dataset):
    def __init__(self, tsv_path:str, transform=None):
        self.transform = transform
        self.tsv_path = tsv_path
        self.image_paths, self.labels = self.load_image_paths_and_labels()
        self.labels_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}

    def load_image_paths_and_labels(self):
        image_paths = []
        labels = []

        tsv_file = open(self.tsv_path, "r")

        for line in tsv_file:
            if line in ['\n', '\r\n']:
                continue
            label, image_path = os.path.split(line.strip())
            image_paths.append(image_path)
            labels.append(label)

        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.labels[idx], self.image_paths[idx])
        label_str = self.labels[idx]
        label = self.labels_to_idx[label_str]
        label = torch.LongTensor([label])

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

# Define the transforms to apply to the images
default_transform = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_unfreeze = transforms.Compose([
    #transforms.Resize((192, 83)),
    default_transform
])

# Specify the root directory of your dataset
root_dir = "data/kaggle_dataset/"

# Create an instance of the custom dataset
#dataset = Kaggle50K(root_dir, transform=transform)

# Create an instance of the custom dataset
# dataset = Kaggle50K(root_dir, transform=None)  # No transform for visualization

# Select a random sample from the dataset
# sample_index = 8000  # Change this to the desired index
# sample_image, sample_label = dataset[sample_index]

# print("label: " + str(sample_label))
# sample_image.show()


# Create a DataLoader to handle batching and shuffling
#batch_size = 64
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Iterate over the dataset using the DataLoader
#for images, labels in dataloader:
    # Your training/validation loop here
    # images and labels contain the batch of images and their labels
#    pass
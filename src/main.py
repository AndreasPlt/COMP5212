# imports 
import torch
from torch.utils.data import Dataset, DataLoader
#from torchsummary import summary
import yaml
import os
from kaggle import api

# import project files
import kaggle50k_dataset
from train import train
from test import test
import models.model



from util.filter_countries import filter_countries
from util.manifest_creation import create_manifest

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")

    return parser.parse_args()

def download_kaggle_dataset(config):
    for file in os.listdir(config['data']['dir']):
        if file == ".dataset":
            print("Dataset already downloaded")
            return

    parent_dir, dataset_name = os.path.split(config['data']['dir'])
    api.dataset_download_files(
        dataset='ubitquitin/geolocation-geoguessr-images-50k',
        path=parent_dir,
        unzip=True,
        quiet=False
    )
    
    # move wrong_resolution_files.txt to data/kaggle_dataset
    os.rename(
        os.path.join(config['data']['dir'], "wrong_resolution_files.txt"),
        os.path.join(parent_dir, "compressed_dataset", "wrong_resolution_files.txt"),
    )

    # rename compressed_dataset to kaggle_dataset
    os.rename(
        os.path.join(parent_dir, "compressed_dataset"),
        config['data']['dir'],
    )
    open(os.path.join(config['data']['dir'], ".dataset"), 'a').close()

def get_model(config):
    model_name = config["model"]["name"]
    num_classes = config["model"]["num_classes"]
    pretrained = config["model"]["pretrained"]
    freeze = config["model"]["freeze"]
    unfreeze_last_n = config["model"]["unfreeze_last_n"]

    return models.model.get_model(model_name, num_classes, pretrained, freeze, unfreeze_last_n)

def get_dataloader(config, valid_countries):
    transforms = kaggle50k_dataset.transform
    batch_size = config["training"]["batch_size"]

    train_loader = DataLoader(
        kaggle50k_dataset.Kaggle50K(config["data"]["train_manifest"], transforms),
        batch_size=batch_size,
        shuffle=True,
    )

    dev_loader = DataLoader(
        kaggle50k_dataset.Kaggle50K(config["data"]["dev_manifest"], transforms),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        kaggle50k_dataset.Kaggle50K(config["data"]["test_manifest"], transforms),
        batch_size=batch_size,
        shuffle=True,
    )

    return train_loader, dev_loader, test_loader

def write_manifest(config, valid_countries):
    create_manifest(
        train_split=config["data"]["train_split"],
        dev_split=config["data"]["dev_split"],
        test_split=config["data"]["test_split"],
        seed=config["data"]["split_seed"],
        root_dir=config["data"]["dir"],
        output_dir=config["data"]["dir"],
        subfolders=valid_countries,  
    )

    train_out = os.path.join(config["data"]["dir"], "train.tsv")
    dev_out = os.path.join(config["data"]["dir"], "dev.tsv")
    test_out = os.path.join(config["data"]["dir"], "test.tsv")

    return train_out, dev_out, test_out

def remove_false_sizes(config):
        root_dir = config["data"]["dir"]
        # iterate over all filenames in the wrong resolution files
        with open(os.path.join(root_dir, "wrong_resolution_files.txt"), "r") as f:
            for line in f:
                image = os.path.join(root_dir, line.strip())
                if os.path.isfile(image):
                    os.remove(image)

def main():
    # get arguments
    args = parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    # remove images that have the wrong size
    download_kaggle_dataset(config)
    remove_false_sizes(config)

    # filter countries
    valid_countries = filter_countries(config["data"]["min_images"], config["data"]["dir"])
    config['model']['num_classes'] = len(valid_countries)

    # write tsv files
    train_out, dev_out, test_out = write_manifest(config, valid_countries)
    config['data']['train_manifest'] = train_out
    config['data']['dev_manifest'] = dev_out
    config['data']['test_manifest'] = test_out

    # get data loaders
    train_loader, dev_loader, test_loader = get_dataloader(config, valid_countries)

    model = get_model(config)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # TODO: Fix output for that
    #summary(model, (3, 32, 32))

    print(f"Training {config['model']['name']} for {config['training']['num_epochs']} epochs")
    epoch_losses = train(
        model, 
        optimizer, 
        criterion, 
        train_loader,
        dev_loader, 
        config,
        )

    print(f"Testing {config['model']['name']}")
    accuracy = test(model, test_loader, k=config["training"]["top_k"], device=config["training"]["device"])
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')


if __name__ == "__main__":
    main()

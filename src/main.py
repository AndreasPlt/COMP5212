# imports
import dataloader
import torch
#from torchsummary import summary
from train import train
from test import test
import models.model
import yaml

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")

    return parser.parse_args()

def get_model(config):
    model_name = config["model"]["name"]
    num_classes = config["model"]["num_classes"]
    pretrained = config["model"]["pretrained"]
    freeze = config["model"]["freeze"]

    return models.model.get_model(model_name, num_classes, pretrained, freeze)


def main():
    # get arguments
    args = parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    train_loader, test_loader = dataloader.cifar_loaders(
        dataloader.batch_size,
    )

    model = get_model(config)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # TODO: Fix output for that
    #summary(model, (3, 32, 32))

    print(f"Training {config['model']['name']} for {config['training']['num_epochs']} epochs")
    epoch_losses = train(model, optimizer, criterion, train_loader, config['training']['num_epochs'])

    print(f"Testing {config['model']['name']}")
    test(model, test_loader)

    if args.plot_loss:
        import matplotlib.pyplot as plt
        plt.plot(epoch_losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim(0, 2)
        plt.title(f"{args.model} with {args.activation} activation")
        plt.savefig(f"{args.model}_{args.activation}.png")
        plt.show()

if __name__ == "__main__":
    main()

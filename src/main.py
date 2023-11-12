# imports
import dataloader
import torch
from torchsummary import summary
from train import train
from test import test

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train a linear model.')
    parser.add_argument('model', type=str, choices=['fully-connected', 'convolutional'])
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('')
    parser.add_argument('--activation', type=str, choices=['relu', 'none', 'sigmoid', 'tanh'], default='relu')
    parser.add_argument('--plot_loss', action='store_true')
    args = parser.parse_args()
    return args

def get_model(args):
    pass

def main():
    # get arguments
    args = parse_args()
    train_loader, test_loader = dataloader.cifar_loaders(
        dataloader.batch_size,
    )

    model = get_model(args)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    summary(model, (3, 32, 32))

    print(f"Training {args.model} model with {args.activation} activation for {args.num_epochs} epochs")
    epoch_losses = train(model, optimizer, criterion, train_loader, args.num_epochs)

    print(f"Testing {args.model}")
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
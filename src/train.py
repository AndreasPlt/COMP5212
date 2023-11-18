from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import socket
HOSTNAME = socket.gethostname()
import datetime

from test import test

def train(model, optimizer, criterion, train_loader, dev_loader, config,):
    model.to(config["training"]["device"])#
    criterion.to(config["training"]["device"])
    writer = SummaryWriter("logs")
    progress_bar = tqdm(range(config["training"]["num_epochs"]))
    initial_loss = 0
    epoch_losses = []
    n_batches = len(train_loader)

    print("Calculate initial loss")
    # Calculate initial loss
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(config["training"]["device"])
        labels = labels.to(config["training"]["device"])
        #print("batch " + str(i+1) + "/" + str(n_batches))
        post_fix = {"batch": f"{i+1}/{n_batches}"}
        progress_bar.set_postfix(post_fix)
        outputs = model(images)
        labels = torch.squeeze(labels,dim=1)
        loss = criterion(outputs, labels)
        initial_loss += loss
    initial_loss /= n_batches

    post_fix = {
        "epoch": 0,
        "loss": initial_loss.item()
    }
    progress_bar.set_postfix(post_fix)
    epoch_losses.append(initial_loss.item())

    accuracy = test(model, dev_loader, k=config["training"]["top_k"], device=config["training"]["device"])
    writer.add_scalar(f"Top {config['training']['top_k']} Accuracy (dev)/epoch", accuracy, 0)
    writer.add_scalar("Loss (train)/epoch", initial_loss.item(), 0)

    writer.flush()
    print("Start training")
    # training loop
    for epoch in progress_bar:
        model.train()
        progress_bar.set_description(desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(config["training"]["device"])
            labels = labels.to(config["training"]["device"])
            labels = torch.squeeze(labels,dim=1)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            post_fix = {"batch": f"{i+1}/{n_batches}"}
            progress_bar.set_postfix(post_fix)
            writer.add_scalar("Loss/batch", loss.item(), (epoch*n_batches + i))
            loss.backward()
            optimizer.step()
            total_loss += loss

        total_loss /= n_batches
        writer.add_scalar("Loss (train)/epoch", total_loss.item(), epoch+1)

        accuracy = test(model, dev_loader, k=config["training"]["top_k"], device=config["training"]["device"])
        writer.add_scalar(f"Top {config['training']['top_k']} Accuracy (dev)/epoch", accuracy, epoch+1)
        writer.flush()

        post_fix = {
            "epoch": epoch+1,
            "loss": total_loss.item()
        }

        progress_bar.set_postfix(post_fix)
        epoch_losses.append(total_loss.item())

    writer.close()

    # save model
    save_path = f"checkpoints/model_{HOSTNAME}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{config['training']['num_epochs']}_epochs_.pth"
    torch.save(model.state_dict(), save_path)
    
    return epoch_losses
    

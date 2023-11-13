from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from test import test

def train(model, optimizer, criterion, train_loader, dev_loader, num_epochs, device=torch.device("cpu")):
    model.to(device)
    criterion.to(device)

    writer = SummaryWriter("logs")
    
    progress_bar = tqdm(range(num_epochs))
    initial_loss = 0
    epoch_losses = []

    n_batches = len(train_loader)

    # Calculate initial loss
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        print("batch " + str(i+1) + "/" + str(n_batches))
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
    print("Start training")
    # training loop
    for epoch in progress_bar:
        model.train()
        progress_bar.set_description(desc=f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            labels = torch.squeeze(labels,dim=1)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            writer.add_scalar("Loss/batch", loss.item(), (epoch*n_batches + i))
            loss.backward()
            optimizer.step()
            total_loss += loss

        total_loss /= n_batches
        writer.add_scalar("Loss (train)/epoch", total_loss.item(), epoch)
        accuracy = test(model, dev_loader, device)
        writer.add_scalar("Accuracy (dev)/epoch", accuracy, epoch)
        post_fix = {
            "epoch": epoch+1,
            "loss": total_loss.item()
        }

        progress_bar.set_postfix(post_fix)
        epoch_losses.append(total_loss.item())

    writer.flush()
    writer.close()
    
    return epoch_losses
    
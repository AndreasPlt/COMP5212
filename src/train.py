from tqdm import tqdm
import torch


def train(model, optimizer, criterion, train_loader, num_epochs, device=torch.device("cpu")):
    model.to(device)
    criterion.to(device)
    
    progress_bar = tqdm(range(num_epochs))
    initial_loss = 0
    epoch_losses = []

    n_batches = len(train_loader)

    # Calculate initial loss
    for _, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
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
        progress_bar.set_description(desc=f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        for _, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            labels = torch.squeeze(labels,dim=1)
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss

        total_loss /= n_batches
        post_fix = {
            "epoch": epoch+1,
            "loss": total_loss.item()
        }
        progress_bar.set_postfix(post_fix)
        epoch_losses.append(total_loss.item())
    
    return epoch_losses
    
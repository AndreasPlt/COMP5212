import torch

def test(model, test_loader, device=torch.device("cpu")):
    correct = 0.
    total = 0.
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            prediction = model(images)
            prediction = torch.argmax(prediction, dim=1, keepdim=True)
            update = (prediction.long() == labels)
            correct += update.sum()
            total += images.shape[0]
    accuracy = correct.float() / total
    return accuracy


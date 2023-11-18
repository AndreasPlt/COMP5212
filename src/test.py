import torch

def test(model, test_loader, k=1, device=torch.device("cpu"),):
    correct = 0.
    total = 0.
    model.eval()
    model.to(device)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            prediction = model(images)
            prediction = torch.topk(prediction, k=k, dim=1)[1]
            update = prediction.eq(labels.view(-1, 1).expand_as(prediction))
            correct += update.sum().item()
            total += images.shape[0]
    accuracy = correct / total
    return accuracy


import torch

def test(model, test_loader, k=[1], device=torch.device("cuda"),):
    if not isinstance(k, list):
        k = [k]
    correct = len(k) * [0.]
    total = 0.
    model.eval()
    model.to(device)
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            prediction = model(images)
            for i, top_k in enumerate(k):
                _, top_k_indices = torch.topk(prediction, k=top_k, dim=1)
                correct[i] += top_k_indices.eq(labels.view(-1, 1).expand_as(top_k_indices)).sum().item()
            total += images.shape[0]
    
    accuracies = [c/total for c in correct]
    return accuracies


import torch

def test(model, test_loader):
    correct = 0.
    total = 0.
    with torch.no_grad():
        for images, labels in test_loader:
            prediction = model(images)
            prediction = torch.argmax(prediction, dim=1)
            correct += (prediction.view(-1).long() == labels).sum()
            total += images.shape[0]
    print('Accuracy of the model on the test images: %f %%' % (100 * (correct.float() / total)))


import torch

def test(model, test_loader):
    correct = 0.
    total = 0.
    with torch.no_grad():
        for images, labels in test_loader:
            prediction = model(images)
            prediction = torch.argmax(prediction, dim=1, keepdim=True)
            update = (prediction.long() == labels)
            correct += update.sum()
            total += images.shape[0]
    print('Accuracy of the model on the test images: %f %%' % (100 * (correct.float() / total)))


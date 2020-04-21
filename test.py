import os
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

# call test function 
if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    batch_size = 20
    num_workers = 0
    data_directory = 'dogImages'
    test_directory = os.path.join(data_directory, 'test/')
    standard_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
    transforms_test =  transforms.Compose([transforms.Resize(size=(224,224)),
                                            transforms.ToTensor(), 
                                            standard_normalization])
    test_data = datasets.ImageFolder(test_directory, transform=transforms_test)

    test_loader = torch.utils.data.DataLoader(test_data,
                                                batch_size=batch_size, 
                                                num_workers=num_workers,
                                                shuffle=False)
    criterion = nn.CrossEntropyLoss()
    model = torch.load('model.pt')

    test(test_loader, model, criterion, use_cuda)
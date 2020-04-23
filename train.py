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

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            
            optimizer.step()
            
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            if batch_idx% 100 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                  (epoch, batch_idx + 1, train_loss))
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model,save_path)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            valid_loss_min = valid_loss
    # return trained model
    return model


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    print("use_cuda",use_cuda)
    batch_size = 20
    num_workers = 0
    data_directory = 'dogImages/'
    train_directory = os.path.join(data_directory, 'train/')
    valid_directory = os.path.join(data_directory, 'valid/')
    test_directory = os.path.join(data_directory, 'test/')



    standard_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
    data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        standard_normalization]),
                    'valid': transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        standard_normalization]),
                    'test': transforms.Compose([transforms.Resize(size=(224,224)),
                                        transforms.ToTensor(), 
                                        standard_normalization])
                    }

    train_data = datasets.ImageFolder(train_directory, transform=data_transforms['train'])
    valid_data = datasets.ImageFolder(valid_directory, transform=data_transforms['valid'])
    test_data = datasets.ImageFolder(test_directory, transform=data_transforms['test'])

    train_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=batch_size, 
                                            num_workers=num_workers,
                                            shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                            batch_size=batch_size, 
                                            num_workers=num_workers,
                                            shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=batch_size, 
                                            num_workers=num_workers,
                                            shuffle=False)
    loaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,133) 
    if use_cuda:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)  

    n_epochs = 10

    train(n_epochs, loaders, model, optimizer, criterion, use_cuda, 'model.pt')
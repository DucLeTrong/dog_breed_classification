import cv2
import os
import torch
from PIL import Image
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms   
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

def face_detector(img_path):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def predict_breed(image_path, model):
    '''Predicts the top 3 most likely breeds for a given image.'''
    class_names = torch.load('names.class')
    model.eval()
    image = Image.open(image_path)
    class_names = torch.load('names.class')
    
    loader = transforms.Compose([transforms.Resize((224,224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.485, 0.456, 0.406),      
                                                      (0.229, 0.224, 0.225))])

    image_tensor = loader(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)

    output = model(input)
    softmax = nn.Softmax(dim=1)
    preds = softmax(output)
    top_preds = torch.topk(preds, 3)
    pred_breeds = [class_names[i] for i in top_preds[1][0]]
    confidence = top_preds[0][0]
    
    return pred_breeds, confidence

def human_dog_predictor(img_path, model):
    '''Detects whether an image contains a human or a dog. If the image contains a dog, the 
    most likely breeds are returned. If the image contains a human, the closest resembling dog
    breeds are returned.'''
    
    print("Runing....")
    breeds, confidence = predict_breed(img_path, model)
        
    fig = plt.figure(figsize=(18,9))
    
    if face_detector(img_path):
        title = ("Hey there HUMAN! \n")
        img = mpimg.imread(img_path)
        ax =fig.add_subplot(1,2,1) 
        ax.imshow(img.squeeze(), cmap="gray", interpolation='nearest')       
        resemblance = ""
        for breed, conf in zip(breeds, confidence):
            resemblance += f"  - {breed.split('.')[1]}\n"
        title += (f"Most Resembled Breeds:\n{resemblance}")
        # plt.title(title,loc='left', y=-0.2)
        ec = (0, .6, .1)
        fc = (0, .7, .2)
        ax.text(0, -100, title, size=10, rotation=0,
                    ha="left", va="top", 
                    bbox=dict(boxstyle="round", ec=ec, fc=fc))
        plt.axis('off')
        
        for i, breed in enumerate(breeds):
            subdir = '/'.join(['img', breed])
            file = random.choice(os.listdir(subdir))
            path = '/'.join([subdir, file])
            img = mpimg.imread(path)
            ax = fig.add_subplot(3,2,(i+1)*2)
            ax.imshow(img.squeeze(), cmap="gray", interpolation='nearest')
            # plt.title(breed.split('.')[1])
            ec = (0, .6, .1)
            fc = (0, .7, .2)
            ax.text(0, -20, breed.split('.')[1], size=10, rotation=0,
                    ha="left", va="top", 
                    bbox=dict(boxstyle="round", ec=ec, fc=fc))
            plt.axis('off')
        fig.savefig('result/result.jpg')
        plt.show()
    
    
    elif confidence[0] > 0.3:
        title = ("Hey there DOG! \n")
        
        img = mpimg.imread(img_path)
        ax =fig.add_subplot(1,2,1) 
        ax.imshow(img.squeeze(), cmap="gray", interpolation='nearest')
        plt.axis('off')
        predictions = ""
        for breed, conf in zip(breeds, confidence):
            if conf > 0.005:
                predictions += f"  - {breed.split('.')[1]} ({(conf*100):.0f}%)\n"
        title += (f"Predicted Breed (confidence):\n{predictions}")
        # plt.title(title,loc='left', y=-0.2)
        ec = (0, .6, .1)
        fc = (0, .7, .2)
        ax.text(0, -50, title, size=10, rotation=0,
                    ha="left", va="top", 
                    bbox=dict(boxstyle="round", ec=ec, fc=fc))

        for i, breed in enumerate(breeds):
            subdir = '/'.join(['img', breed])
            file = random.choice(os.listdir(subdir))
            path = '/'.join([subdir, file])
            img = mpimg.imread(path)
            ax = fig.add_subplot(3,2,(i+1)*2)
            ax.imshow(img.squeeze(), cmap="gray", interpolation='nearest')
            # plt.title(breed.split('.')[1])
            plt.axis('off')
            ec = (0, .6, .1)
            fc = (0, .7, .2)
            ax.text(0, -20, breed.split('.')[1], size=10, rotation=0,
                    ha="left", va="top", 
                    bbox=dict(boxstyle="round", ec=ec, fc=fc))
        fig.savefig('result/result.jpg')
        plt.show() 
        
    else:
        print("Hmm. I can't determine what you are. ¯\_(ツ)_/¯")
        img = mpimg.imread(img_path)
        _ = ax.imshow(img)        
        plt.axis('off')
        plt.show()     
    
    print('Done!')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str,
                        default='model.pt')

    parser.add_argument('--img_path', type=str,
                        default='test.jpg')

    args = parser.parse_args()
    model_path = args.model_path
    img_path = args.img_path
    print("Loading.... \n")
    model = torch.load(model_path)
    human_dog_predictor(img_path,model)

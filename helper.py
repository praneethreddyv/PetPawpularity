import numpy as np
import pandas as pd
import torch,os
from torchvision import datasets ,transforms
from torchvision.io import read_image,ImageReadMode
import torch.nn as nn
from torchvision import models
 
 
current_path = os.getcwd()

# getting the current path

model=models.densenet161(pretrained=False)
for param in model.parameters(): #Freezing layers
    param.require_grad=False
model.classifier=nn.Sequential(nn.BatchNorm1d(2208),nn.ReLU(inplace=True),nn.Dropout(p=0.6),nn.Linear(2208,64),nn.BatchNorm1d(64),nn.ReLU(),nn.Linear(64,1))
for param in model.classifier.parameters():
    param.require_grad=True

model.load_state_dict(torch.load(r'weights\model_weights.pth'))

def predictor(img_path): # here image is file name 
    model.eval()
    img= read_image(img_path,ImageReadMode.RGB)
    img=img.unsqueeze(0)
    print(img.shape)
    transform=transforms.Compose([
        transforms.Resize(256),transforms.CenterCrop(224), 
         transforms.ConvertImageDtype(torch.float), transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])])
    img=transform(img)
    
    prediction = model(img) 
    prediction=prediction.to('cpu')
    prediction=prediction.item()
    
    return prediction

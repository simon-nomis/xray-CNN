import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import random
import matplotlib.pyplot as plt


# Boolean value to represent if the data is saved so prevent rerunning some things
saved = True




class data():

    """
    This class gets the data and labels them accordingly and also saves them
    The image size is downscaled to 250 by 250 pixels
    """

    IMG_SIZE = 250
    normal = os.path.join(os.getcwd(), "NORMAL")
    pneumonia = os.path.join(os.getcwd(), "PNEUMONIA")
    data = []
    labels = []

    def append_data(self):
        path = os.getcwd()
        count = 0

        for normal in tqdm(os.listdir(os.path.join(path,"NORMAL"))):
            count += 1
            new_path = os.path.join(path,"NORMAL",normal)
            img = cv2.imread(new_path,0)
            img = cv2.resize(img, (self.IMG_SIZE,self.IMG_SIZE) )/255
            
            self.data.append(img)
            self.labels.append(0)

        for sick in tqdm(os.listdir(os.path.join(path,"PNEUMONIA"))):
            new_path = os.path.join(path,"PNEUMONIA",sick)
            if "virus" in sick:
                img = cv2.imread(new_path,0)
                img = cv2.resize(img, (self.IMG_SIZE,self.IMG_SIZE) ) /255
                self.data.append(img)
                self.labels.append(1)
            else:
                img = cv2.imread(new_path,0)
                img = cv2.resize(img, (self.IMG_SIZE,self.IMG_SIZE) ) /255
                self.data.append(img)
                self.labels.append(2)

        np.save("data.npy", self.data)
        np.save("labels.npy", self.labels)




class dataset(torch.utils.data.Dataset):

    """
    Inherits from the pytorch Dataset class to allow usage of pytorch's DataLoader
    """

    def __init__(self,data,labels):
        self.x = torch.from_numpy(data).float().unsqueeze(1)
        self.y = torch.from_numpy(labels).long()
        
    def __len__(self):
        if len(self.x) == len(self.y):
            return len(self.x)
        raise Exception("length error, input and outputs have different lengths")
        
    def __getitem__(self,index):
        return self.x[index], self.y[index]
    



###############################
#   This block of code loads the data

if not saved:
    data = data()
    data.append_data()


x = np.load("data.npy", allow_pickle=True)
y = np.load("labels.npy", allow_pickle=True)

class loaddata():
    def __init__(self,x,y):
        self.data = x
        self.labels = y

data = loaddata(x,y)

###############################





class model(nn.Module): 

    """
    Inherits from pytorch's Module class
    Inputs are greyscaled images and has three output classes
    """

    def __init__(self):
        super(model, self).__init__()
        
        self.convlayer = nn.Sequential(
            nn.Conv2d(1,64,3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64,32,3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(32,16,3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,2)
        )

        self.linearlayer = nn.Sequential(
        
            nn.Linear(16*29*29,128),
            nn.LeakyReLU(),
            
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            
            nn.Linear(32, 3)

        )

        
    def forward(self,x):
        x = self.convlayer(x)
        x = x.view(-1,16*29*29)
        x = self.linearlayer(x)
        return x
        
###
# Loads a saved model if you have one
bool_model = True


if bool_model:
    model = model()
    model.load_state_dict(torch.load('/home/simon/Desktop/xray/model'))
    model.eval()

else:
    model = model()

###

###
# Used cross entropy loss for multi class classification
# Split the data into three sections

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()


xtrain =  np.array(data.data[:int(len(x)*0.7)])
ytrain =  np.array(data.labels[:int(len(x)*0.7)])

xval =  np.array(data.data[int(len(x)*0.7):int(len(x)*0.85)])
yval =  np.array(data.labels[int(len(x)*0.7):int(len(x)*0.85)])

xtest =  np.array(data.data[int(len(x)*0.85):])
ytest =  np.array(data.labels[int(len(x)*0.85):])


trainset = dataset(xtrain,ytrain)
valset = dataset(xval,yval)
testset = dataset(xtest,ytest)

loadtrain = torch.utils.data.DataLoader(dataset = trainset, batch_size = 64, shuffle = True)
loadval = torch.utils.data.DataLoader(dataset = valset, batch_size = 64, shuffle = True)
loadtest = torch.utils.data.DataLoader(dataset = testset, batch_size = 64, shuffle = True)

###

###
# Trains and evaluates the data

EPOCHS = 20

trainloss = []
valloss = []
testloss = []


for epoch in range(EPOCHS):
    model.train()
    for x, y in tqdm(loadtrain):
        optimizer.zero_grad()
        outputs = model(x)
                
        
        loss = loss_function(outputs, y)
        loss.backward()
        optimizer.step()
        
    trainloss.append(loss.item())
    
    
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data, target in loadval:
            output = model(data)
            val_loss = loss_function(output, target)
            for idx,labels in enumerate(target):
                total += 1
                if output[idx].argmax() == labels:
                    correct += 1
    valloss.append(val_loss.item())

    print(f"Training loss: {loss.item()}, Val_loss: {val_loss.item()}, Val_Acc: {correct/total}")

###


###
# Test set evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in loadtest:
        output = model(data)
        val_loss = loss_function(output, target)
        for idx,labels in enumerate(target):
            total += 1
            if output[idx].argmax() == labels:
                correct += 1
            
            
print(correct/total)



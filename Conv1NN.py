import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim

REBUILD_DATA = True #switch for not rebuding data every time

class DogsVSCats():
    IMG_SIZE = 50
    CATS = "PetImages/Cat" #give immages directory locations
    DOGS = "PetImages/Dog" #^^
    LABELS = {CATS: 0, DOGS: 1} #def dict labels with cats and dogs as keys with val's 1, 0
    training_data = []
    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.LABELS: #itterate over each 1/2 of directory
            print(label) #either dog or cat
            for f in tqdm(os.listdir(label)): #itterate for each immage in each dir
                try:
                    path = os.path.join(label, f) #def path as path to current img in loop
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #def current img var and read it as G.S
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE)) #re-def current img as a 50,50
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]]) #add the tensor of the img and 1hot(for animal label) to class var list


                    if label == self.CATS:  #itt count
                        self.catcount += 1
                    if label == self.DOGS:
                        self.dogcount += 1
                except Exception as e:
                    pass  ##some will error for bad info / corrupt

        np.random.shuffle(self.training_data) #shuffle training data
        np.save("training_data.npy", self.training_data) #save training_data list on disk
        print("Cats:", self.catcount)
        print("Dogs:", self.dogcount)

if REBUILD_DATA: #call if switch on
    dogsvcats = DogsVSCats()
    dogsvcats.make_training_data()
    training_data = np.load("training_data.npy", allow_pickle=True) #for switch*


class Net(nn.Module): #inheret from pytorch class
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5) #layers (in, out, kernel size)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5) #learn more specific features but are in location sensitive featuremap format
## cant feed 2D/not-flat input into linear output so make fake data and get shape to get input number
        x = torch.randn(50,50).view(-1,1,50,50) #def x as fake data to be fed into layers later (convs)
        self._to_linear = None
        self.convs(x) #pass fake data into layrs that are not defined yet

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2) ###512 comes from some unwrapping and fcl's are meant to just avg nets guess to num classes 


    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))  #MaxPool to downsample feature map #relu for linActivation function
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))  ##pool make non-sensitive to location
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        if self._to_linear is None: #only run once
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2] #mult all dimentions of tensor to get flattened shape
        return x

    def forward(self, x): ###
        x = self.convs(x)
        x = x.view(-1, self._to_linear) #reshape x using fake data shape
        x = F.relu(self.fc1(x)) #run new shape x through linear layer
        x = self.fc2(x) #output layer
        return F.softmax(x, dim=1) #output 1 - 0 and normalize along 1 axis ###
net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.001)###lr is how much to respond to error (bounce thing)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50) #featuresets
X = X/255.0 #scaling immage (want 1-0)
y = torch.Tensor([i[1] for i in training_data]) #labels

VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT) #separate some testing data

train_X = X[:-val_size] #slice data ˘˘
train_y = y[:-val_size]

test_X = X[:-val_size:]
test_y = y[:-val_size:]

BATCH_SIZE = 100
EPOCS = 1

for epoch in range(EPOCS): ###run stuff through cpu
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
        batch_y = train_y[i:i+BATCH_SIZE]

        net.zero_grad() ###**
        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

print(loss, 5) ##diff between predicted and actual (used for optimization)

#predict on model
correct = 0
total = 0
with torch.no_grad(): ###
    for i in tqdm(range(len(test_X))):
        real_class =  torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1,1,50,50))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct += 1
        total += 1a
print("Accuracy: ", round(correct/total,3)) ##% correct

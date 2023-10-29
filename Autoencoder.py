from typing import Any, Tuple
import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from numpy import dot
from numpy.linalg import norm
import os
from PIL import Image

class ImageFodlerWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):

        img, label = super(ImageFodlerWithPaths, self).__getitem__(index)
        
        path = self.imgs[index][0]

        return img, label, path 
# 하이퍼파라미터
EPOCH = 10
BATCH_SIZE = 290
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("Using Device:", DEVICE)

PIL_tensor = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),transforms.Grayscale()])
test_imageRoot = Image.open('test3.jpg')

test_imageTensor = PIL_tensor(test_imageRoot)
print(test_imageTensor)
folder_root = './.data/gradu'
ownset = ImageFodlerWithPaths(
    root = folder_root,
    transform = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor(),transforms.Grayscale()])
)

train_loader = torch.utils.data.DataLoader(
    dataset     = ownset,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
    num_workers = 2
)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3),
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.ReLU(),     
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

autoencoder = Autoencoder().to(DEVICE)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
criterion = nn.MSELoss()

def train(autoencoder, train_loader):
    autoencoder.train()
    for step, (x, label,path) in enumerate(train_loader):
        x = x.view(-1, 28*28).to(DEVICE)
        y = x.view(-1, 28*28).to(DEVICE)
        label = label.to(DEVICE)

        encoded, decoded = autoencoder(x)

        loss = criterion(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for epoch in range(1, EPOCH+1):
    train(autoencoder, train_loader)

    print("[Epoch {}]".format(epoch))

images, labels, paths = next(iter(train_loader))

input = test_imageTensor.view(-1,28*28)
input_x = input.to(DEVICE)
input_data,ouput_data = autoencoder(input_x)
input_data = input_data.to("cpu")
inX = input_data.data[:, 0].numpy()
inY = input_data.data[:, 1].numpy()
inZ = input_data.data[:, 2].numpy()


def cos_sim(a,b):
    c = dot(a,b)/(norm(a)*norm(b))
    return c

for xin, yin, zin in zip(inX,inY,inZ):
    input_vec = np.array([xin,yin,zin])
    print(xin,yin,zin)

top = 0
top_count = 0
top_path = ''
count = 0

for image in images:
    print(image.size())
    view_data = image.view(-1,28*28)
    test_x = view_data.to(DEVICE)
    encoded_data, _ = autoencoder(test_x)
    encoded_data = encoded_data.to("cpu")

    X = encoded_data.data[:, 0].numpy()
    Y = encoded_data.data[:, 1].numpy()
    Z = encoded_data.data[:, 2].numpy()
    print(X,Y,Z)
    vec = np.array([X,Y,Z])
    sim = cos_sim (input_vec,vec)
    print("x: {} y: {} z: {} count: {} sim: {} || path: {}".format(X,Y,Z,count,sim,paths[count]))

    if sim>top:
        top = sim
        top_count = count
        top_vector = vec
        top_path = paths[count]
    print(sim)
    count = count +1


print("TOP Vec: {} Count: {} Similarity: {} path: {}".format(top_vector,top_count,top,top_path))
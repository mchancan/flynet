################################################################################
# This file is part of the paper:
#  "A Hybrid Compact Neural Architecture for Visual Place Recognition," 
#  in IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 993-1000,
#  April 2020, doi: 10.1109/LRA.2020.2967324.
#  Project page: https://mchancan.github.io/projects/FlyNet
# 
# Copyright (c) 2020, Marvin Chancán
# Author:
#  Marvin Chancán (mchancanl@uni.pe)
#
# This code is under the MIT License for academic purposes
# (refer to the LICENSE file for details). For commercial
# usage, please contact us via mchancanl@uni.pe
#
###############################################################################

from __future__ import print_function, division

import cv2
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = 'dataset/Nordland-Subset-100/'

train_dir = data_dir + 'summer/' # train
test_dir = data_dir + 'fall/' # test
test_dir2 = data_dir + 'winter/' # test_2

num_classes = 100
img_w = 64
img_h = 32
img_c = 1


# Loading the data

def get_images(img_dir, num_imgs):
    print('Loading images...')
    x = []
    for i in range(num_imgs):
        img = cv2.imread(img_dir+'/'+str(i)+'.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_w,img_h),interpolation = cv2.INTER_AREA)
        x += [img.reshape(img_h,img_w,img_c)]
    return np.asarray(x)/255

x_train = get_images(train_dir, num_classes)
print('Train images shape:', x_train.shape)

x_test = get_images(test_dir, num_classes)
print('Test_1 images shape:', x_test.shape)

x_test2 = get_images(test_dir2, num_classes)
print('Test_2 images shape:', x_test2.shape)


# Nordland dataset class

class NordlandDataset(Dataset):
    """Nordland dataset."""
    
    def __init__(self, data):
        """
        Args:
            data (string): Directory with all the images.
        """
        self.num_images = data.shape[0]
        self.data = data
    
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self.data[idx]
        ids = np.array(idx)
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image), torch.from_numpy(ids)


# Hyper parameters
num_epochs = 200
batch_size = num_classes
learning_rate = 0.001
hidden_size = 64
iter_display = 1

input_size = img_w*img_h


test_dataset = NordlandDataset(data=x_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=4)

test_dataset2 = NordlandDataset(data=x_test2)
test_loader2 = DataLoader(test_dataset2, batch_size=batch_size,
                         shuffle=False, num_workers=4)

train_dataset = NordlandDataset(data=x_train)
dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4)


# FlyNet model, including a hidden-layer (FNA) and a fully-connected (FC) output layer

class FlyNet(nn.Module):
    def __init__(self, input_size=input_size, hidden_size=hidden_size, num_classes=num_classes):
        super(FlyNet, self).__init__()
        self.sampling_ratio = 0.1 # 10% random, sparse connectivity
        self.wta_length = int(hidden_size/2) # 50% WTA
        self.fna_weight = torch.Tensor(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes, bias=False)
        self.reset_fna_weight()
        
    def reset_fna_weight(self):
        # Defining W: binary, sparse matrix
        self.fna_weight = (nn.init.sparse_(self.fna_weight,
                                           sparsity=1-self.sampling_ratio)!=0).float().to(device)
    def fna(self, x):
        firing_rates = torch.matmul(x,self.fna_weight)
        wta_threshold = torch.topk(firing_rates, self.wta_length, dim=1)[0][:,-1].reshape(num_classes,1)
        return (firing_rates>=wta_threshold).float()
        
    def forward(self, x):
        out = self.fna(x)
        out = self.fc(out)
        return out

model = FlyNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Train the model

total_step = len(dataloader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(dataloader):
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images.float())
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        _, argmax = torch.max(outputs, 1)
        accuracy = (labels == argmax.squeeze()).float().mean()
    
    if (epoch+1) % iter_display == 0:
        print ('Epoch [{}/{}], train_loss: {:.4f}, train_acc: {:.2f}'
                .format(epoch+1, num_epochs, loss.item(), accuracy.item()))


# Test the model

def eval_model(model,data_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in data_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            outputs = model(images.float())
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy: {} %'.format(100 * correct / total))
    return outputs.data.cpu().numpy().argmax(axis=1)


y0 = eval_model(model, dataloader)
y1 = eval_model(model, test_loader)
y2 = eval_model(model, test_loader2)


# Plot the resutls

plt.figure(figsize=(15,6))
num_samples = 5
samples_per_class = 3
tol = 11

idxs = np.random.choice(num_classes, num_samples, replace=False)

for cls in range(num_samples):
    idx = idxs[cls]
    for i in range(samples_per_class):
        plt_idx = i * num_samples + cls + 1
        plt.subplot(samples_per_class, num_samples, plt_idx)
        if i == 0:
            plt.imshow(io.imread(train_dir+f'{y0[idx]}.png'))
            plt.title(f'{y0[idx]}', color='blue' if abs(idx-y0[idx])<tol else 'red')
        elif i == 1:
            plt.imshow(io.imread(test_dir+f'{y1[idx]}.png'))
            plt.title(f'{y1[idx]}', color='blue' if abs(idx-y1[idx])<tol else 'red')
        else:
            plt.imshow(io.imread(test_dir2+f'{y2[idx]}.png'))
            plt.title(f'{y2[idx]}', color='blue' if abs(idx-y2[idx])<tol else 'red')
        plt.axis('off')
plt.tight_layout()
plt.savefig('results/demo_flynet_nordland.jpg', dpi = 400)


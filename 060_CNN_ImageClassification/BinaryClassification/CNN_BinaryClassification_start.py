#%% packages
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score

#%% transform, load data
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

batch_size = 32
trainset = torchvision.datasets.ImageFolder(root='data/train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.ImageFolder(root='data/test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

classes = ['Positive', 'Negative']

# %% visualize images
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images, nrow=2))

# %% Neural Network setup
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # BS, 6, 64, 64
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=16*16*16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.sigmoid = nn.Sigmoid()

        # define layers
    def forward(self, x):
        x = self.conv1(x) # BS, 6, 64, 64
        x = self.relu(x)
        x = self.pool(x)  # BS, 6, 32, 32
        x = self.conv2(x) # BS, 16, 32, 32
        x = self.relu(x)
        x = self.pool(x) # BS, 16, 16, 16
        x = self.flatten(x) # BS, 16*16*16
        x = self.fc1(x) # BS, 64
        x = self.relu(x)
        x = self.fc2(x) # BS, 32
        x = self.relu(x)
        x = self.output(x) # BS, 1
        x = self.sigmoid(x)
        return x
        
    

#%% init model
# test = torch.rand(size=(4, 1, 64, 64))
# model = CNN()
# model(test).shape

model = CNN()
lr = 0.01
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 30

# %% training
losses = []
for epoch in range(epochs):
    loss_total = 0
    for X, y in trainloader:
        # zero gradients
        optimizer.zero_grad()
       
        # forward pass
        prediction = model(X)
        # calc losses
        loss = loss_fn(prediction.squeeze(), y.float())

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()
        
        loss_total += loss.item()

    losses.append(loss_total)
    print(f'Epoch: {epoch}, Loss: {loss_total}')

# %%
import seaborn as sns
sns.lineplot(x=range(epochs), y=losses)

# %% test
model.eval()
y_test = []
y_hat = []

with torch.no_grad():
    for X, y in testloader:
        predictions = (model(X)).squeeze()
        probabilities = predictions.round().int()
        y_hat.extend(probabilities.numpy())
        y_test.extend(y.numpy())

# %%
accuracy_score(y_test, y_hat) * 100

# %%
from collections import Counter
Counter(y_test)

# We know that data is balanced, so baseline classifier has accuracy of 50 %.
# %%

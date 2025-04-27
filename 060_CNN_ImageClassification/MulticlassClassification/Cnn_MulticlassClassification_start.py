#%% packages
import torch
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# %% transform and load data
# TODO: set up image transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))

])
# TODO: set up train and test datasets
trainset = torchvision.datasets.ImageFolder('./train', transform=transform)
testset = torchvision.datasets.ImageFolder('./test', transform=transform)

# TODO: set up data loaders
batch_size = 32
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

# %%
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

images, labels = next(iter(trainloader))
imshow(torchvision.utils.make_grid(images, nrow=2))
#%%
# TODO: set up model class
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16*16*16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x) # BS, 6, 64, 64
        x = self.relu(x) 
        x = self.pool(x) # BS, 6, 32, 32
        x = self.conv2(x) # BS, 16, 32, 32
        x = self.relu(x) 
        x = self.pool(x) # BS, 16, 16, 16
        x = self.flatten(x) # BS, 16*16*16
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.output(x)
        # x = self.softmax(x)
        return x

# input = torch.rand((batch_size, 1, 64, 64))
# print(model(input).shape)

# %% loss function and optimizer
model = CNN(num_classes=4)
classes = trainset.classes
num_classes = len(classes)
lr = 0.001
epochs = 30
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# %% training
losses = []
for epoch in range(epochs):
    total_loss = 0
    for X, y in trainloader:
        optimizer.zero_grad()

        prediction = model(X)

        loss = loss_fn(prediction, y)

        loss.backward()

        optimizer.step()
        
        total_loss += loss.item()

    losses.append(total_loss)
    print(f'Epoch: {epoch}, Loss: {total_loss}')

    
#%%
import seaborn as sns
sns.lineplot(x=range(epochs), y=losses)

# %% test
model.eval()

with torch.no_grad():
    y_test = []
    y_hat = []
    for X, y in testloader:
        prediction = model(X)
        predictions = torch.argmax(prediction, dim=1)
        y_hat.extend(predictions.numpy())
        y_test.extend(y.numpy())

acc = accuracy_score(y_pred=y_hat, y_true=y_test) * 100
print(acc)

# %% confusion matrix
cm = confusion_matrix(y_true=y_test, y_pred=y_hat)

ax = sns.heatmap(cm, annot=True, cmap='crest')
ax.set_yticklabels(trainset.classes)
ax.set_xticklabels(trainset.classes)
ax.set_title(f'Confusion Matrix ({acc}%)')

# %%
# load a test image
from PIL import Image
img = Image.open('test/corgi/corgi_11.jpg')
img = transform(img)
img = img.unsqueeze(0)

# predict the class
with torch.no_grad():
    y_hat = model(img).round()
    print(classes[int(y_hat.argmax(dim=1))])
# %%

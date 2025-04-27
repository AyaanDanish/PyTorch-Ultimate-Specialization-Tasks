#%% packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
# %% data import
iris = load_iris()
X = iris.data
y = iris.target

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %% convert to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# %% dataset
class IrisDataset(Dataset):
    def __init__(self, X_train, y_train):
        super().__init__()
        self.X = torch.from_numpy(X_train).float()
        self.y = torch.from_numpy(y_train).long()
        # self.len = self.X.shape[0]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    


# %% dataloader
train_data = IrisDataset(X_train, y_train)
test_data = IrisDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=32)
test_loader = DataLoader(test_data)

 # %% check dims
train_data.X.shape, test_data.X.shape

# %% define class
class Classifier(nn.Module):
    def __init__(self, num_features, num_hidden, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# %% hyper parameters
num_features = train_data.X.shape[1]
num_hidden = 10
num_classes = len(train_data.y.unique())

# %% create model instance
model = Classifier(num_features, num_hidden, num_classes)

# %% loss function
loss_fn = nn.CrossEntropyLoss()

# %% optimizer
lr = 0.01
epochs = 1000
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# %% training
losses = []
for epoch in range(epochs):
    batch_loss = 0
    for X, y in train_loader:
        optimizer.zero_grad()

        prediction = model(X)
        loss = loss_fn(prediction, y)

        loss.backward()

        optimizer.step()
        
        batch_loss += loss.item()

    losses.append(batch_loss)
    print(f'Epoch: {epoch}, Loss: {batch_loss}')


# %% show losses over epochs
sns.lineplot(x=range(epochs), y=losses)

# %% test the model
import numpy as np
model.eval()
X_test = torch.from_numpy(X_test) if isinstance(X_test, np.ndarray) else X_test
with torch.no_grad():
    y_pred = torch.argmax(model(X_test), dim=1)

# %% Accuracy
accuracy_score(y_pred=y_pred, y_true=y_test) * 100

# %%
from collections import Counter
Counter(y_test).most_common()[0][1] / len(y_test)

# %%

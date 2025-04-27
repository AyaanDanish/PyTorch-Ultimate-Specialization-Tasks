#%% packages
from ast import Mult
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import seaborn as sns
import numpy as np
from collections import Counter
# %% data prep
X, y = make_multilabel_classification(n_samples=10000, n_features=10, n_classes=3, n_labels=2)
X_torch = torch.FloatTensor(X)
y_torch = torch.FloatTensor(y)

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X_torch, y_torch, test_size = 0.2)


# %% dataset and dataloader
class MultilabelDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# TODO: create instance of dataset
dataset = MultilabelDataset(X_train, y_train)

# TODO: create train loader
train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# %% model
# TODO: set up model class
# topology: fc1, relu, fc2
# final activation function??
class MultilabelModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=100):
        super(MultilabelModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# TODO: define input and output dim
input_dim = dataset.X.shape[1]
output_dim = dataset.y.shape[1]

# TODO: create a model instance
model = MultilabelModel(input_dim, output_dim, hidden_dim=30)

# %% loss function, optimizer, training loop
# TODO: set up loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()

LR = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

losses = []
slope, bias = [], []
number_epochs = 500

# TODO: implement training loop
for epoch in range(number_epochs):
    for j, (X, y) in enumerate(train_dataloader):
        
        # optimization
        optimizer.zero_grad()

        # forward pass
        y_pred = model(X)


        # compute loss
        loss = loss_fn(y_pred, y)
        
        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

        for name, param in model.named_parameters():
            if param.requires_grad:
                if name == 'linear.weight':
                    slope.append(param.data.numpy()[0][0])
                if name == 'linear.bias':
                    bias.append(param.data.numpy()[0])

    # TODO: print epoch and loss at end of every 10th epoch

    losses.append(float(loss.item()))
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss {loss.item()}')
    
# %% losses
# TODO: plot losses
sns.lineplot(x=range(number_epochs), y=losses)

# %% test the model
# TODO: predict on test set
with torch.no_grad():
    y_test_pred = model(X_test).round()

#%% Naive classifier accuracy
# TODO: convert y_test tensor [1, 1, 0] to list of strings '[1. 1. 0.]'
y_test_str = [str(y) for y in y_test.detach().numpy()]

# TODO: get most common class count
from collections import Counter
most_common = Counter(y_test_str).most_common()[0][1]
# TODO: print naive classifier accuracy
print(f'Naive Classifier Accuracy: {most_common/len(y_test_str)*100}%')

# %% Test accuracy
# TODO: get test set accuracy
print(f'Model Accuracy: {accuracy_score(y_test, y_test_pred)*100:.2f}%')
# %%

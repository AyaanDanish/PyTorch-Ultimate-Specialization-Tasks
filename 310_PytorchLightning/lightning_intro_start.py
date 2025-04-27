#%% packages
import graphlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
import seaborn as sns


import pytorch_lightning as pl

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

#%% convert data to tensor
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1)
y_list = cars.mpg.values
y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)

#%% Dataset and Dataloader
class LinearRegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(dataset = LinearRegressionDataset(X_np, y_np), batch_size=2)



#%%
class LitLinearRegression(pl.LightningModule):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.linear(x)
    
    def configure_optimizers(self):
        learning_rate = 0.02
        optimizer = torch.optim.SGD(self.parameters())
        return optimizer
    
    def training_step(self, train_batch):
        X, y = train_batch
        #forward pass
        y_pred = self.forward(X)

        #calc loss
        loss = self.loss_fn(y_pred, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
#%%

early_stop = EarlyStopping(monitor='train_loss', patience=2, verbose=True, mode='min')
model = LitLinearRegression(input_size=1, output_size=1)
trainer = pl.Trainer(max_epochs=500, log_every_n_steps=2, callbacks=[early_stop])


# %%
trainer.fit(model=model, train_dataloaders=train_loader)
# %%
trainer.current_epoch

# %%
for par in model.parameters():
    print(par)



# %%

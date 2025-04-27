#%% packages
import torch
from torch.utils.data import DataLoader
from torch import nn

import math
import matplotlib.pyplot as plt
import numpy as np
from random import uniform

import seaborn as sns
sns.set_theme(rc={'figure.figsize':(12,12)})
#%% create training data
TRAIN_DATA_COUNT = 1024
theta = np.array([uniform(0, 2 * np.pi) for _ in range(TRAIN_DATA_COUNT)]) # np.linspace(0, 2 * np.pi, 100)
# Generating x and y data
x = 16 * ( np.sin(theta) ** 3 )
y = 13 * np.cos(theta) - 5 * np.cos(2*theta) - 2 * np.cos(3*theta) - np.cos(4*theta)
sns.scatterplot(x=x, y=y)

#%% prepare tensors and dataloader
train_data = torch.Tensor(np.stack((x, y), axis=1))

train_labels = torch.zeros(TRAIN_DATA_COUNT)
train_set = [
    (train_data[i], train_labels[i]) for i in range(TRAIN_DATA_COUNT)
] 

#  dataloader
BATCH_SIZE = 64
train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True
)
#%% initialize discriminator and generator
# TODO: initialize discriminator and generator
discriminator = nn.Sequential(
    nn.Linear(2, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 1),
    nn.Sigmoid(),
)

generator = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

#%%
test_img = torch.randn((BATCH_SIZE, 2))
discriminator(test_img).shape

# %% training
LR = 0.001
NUM_EPOCHS = 2000
loss_function = nn.BCELoss()
optimizer_discriminator = torch.optim.Adam(discriminator.parameters())
optimizer_generator = torch.optim.Adam(generator.parameters())

for epoch in range(NUM_EPOCHS):
    for n, (real_samples, _) in enumerate(train_loader):
        # TODO: Data for training the discriminator
        real_samples_labels = torch.ones((BATCH_SIZE, 1))
        latent_space_samples = torch.randn((BATCH_SIZE, 2))
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((BATCH_SIZE, 1))
        all_samples = torch.cat((real_samples, generated_samples), dim=0)
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels), dim=0) 

        if epoch % 2 == 0:
            # TODO: Training the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()  
            

        if epoch % 2 == 1:
            # TODO: Data for training the generator
            latent_space_samples = torch.randn((BATCH_SIZE, 2))
            # TODO: Training the generator
            generator.zero_grad()
            generated_sample = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_sample)
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()
                    
            
    # Show progress
    if epoch % 10 == 0 and epoch > 0:
        print(epoch)
        print(f"Epoch {epoch}, Discriminator Loss {loss_discriminator}")
        print(f"Epoch {epoch}, Generator Loss {loss_generator}")
        with torch.no_grad():
            latent_space_samples = torch.randn(1000, 2)
            generated_samples = generator(latent_space_samples).detach()
        plt.figure()
        plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
        plt.xlim((-20, 20))
        plt.ylim((-20, 15))
        plt.text(10, 15, f"Epoch {epoch}")
        plt.savefig(f"train_progress/image{str(epoch).zfill(3)}.jpg")
        plt.show()


    

# %% check the result
latent_space_samples = torch.randn(10000, 2)
generated_samples = generator(latent_space_samples)
generated_samples = generated_samples.detach()
plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
plt.text(10, 15, f"Epoch {epoch}")

# %%
for i, data in enumerate(train_loader):
    print(i)
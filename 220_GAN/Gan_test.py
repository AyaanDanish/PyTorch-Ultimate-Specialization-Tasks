#%% packages
import torch
from torch.utils.data import DataLoader
from torch import nn

import math
import matplotlib.pyplot as plt
import numpy as np
from random import uniform
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset
import seaborn as sns
sns.set_theme(rc={'figure.figsize':(12,12)})
# %%
BATCH_SIZE = 64

transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize all images to 64x64
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
])

trainset = datasets.FashionMNIST('./data', download=True, transform=transform)
# train_subset = Subset(trainset, range(0, 1000))
train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

# %%

discriminator = nn.Sequential(
    nn.Conv2d(1, 6, 3), 
    nn.LeakyReLU(), # BS, 6, 62, 62
    nn.Conv2d(6, 16, 3),
    nn.LeakyReLU(), # BS, 16, 60, 60
    nn.Flatten(),
    nn.Linear(16 * 60 * 60, 32),
    nn.LeakyReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

test_discrim = torch.rand((BATCH_SIZE*2, 1, 64, 64))
discriminator(test_discrim).shape

# %%
generator = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Linear(256, 1024),
    nn.ReLU(),
    nn.Unflatten(1, (64, 4, 4)),
    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
    nn.ReLU(), # BS, 32, 6, 6
    nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
    nn.ReLU(), # BS, 32, 6, 6
    nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
    nn.ReLU(), # BS, 32, 6, 6
    nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
    nn.Tanh()
)

test_gen = torch.rand(32, 100)
generator(test_gen).shape
# %%
LR = 0.001
NUM_EPOCHS = 1000
discrim_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LR)
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=LR)
loss_fn = nn.BCELoss()

#%%
def show_image(img):
    img = 0.5 * (img + 1)  # denormalizeA
    # img = img.clamp(0, 1) 
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# %%
for epoch in range(NUM_EPOCHS):
    for idx, (real_images, _) in enumerate(train_loader):
        # prepping data
        discrim_optimizer.zero_grad() # MIGHT NEED TO CHANGE

        real_image_labels = torch.ones((BATCH_SIZE, 1))
        random_noise = torch.randn((BATCH_SIZE, 100))
        generated_images = generator(random_noise)
        generated_image_labels = torch.zeros((BATCH_SIZE, 1))

        all_images = torch.cat((real_images, generated_images))
        all_image_labels = torch.cat((real_image_labels, generated_image_labels)) 

        if epoch % 2 == 0:
            print(idx, 'discrim training')       
            discrim_optimizer.zero_grad()
            # training discrim
            discrim_decision = discriminator(all_images)
            discrim_loss = loss_fn(discrim_decision, all_image_labels)
            discrim_loss.backward()
            discrim_optimizer.step()

        if epoch % 2 == 1:
            print('gen training')
            gen_optimizer.zero_grad() # MIGHT NEED TO CHANGE
            random_noise = torch.randn((BATCH_SIZE, 100))
            generated_images = generator(random_noise)
            discrim_decision = discriminator(generated_images)
            gen_loss = loss_fn(discrim_decision, real_image_labels)
            gen_loss.backward()
            gen_optimizer.step()
        
    if epoch % 10 == 0 and epoch > 0:
        print(epoch)
        print(f"Epoch {epoch}, Discriminator Loss {discrim_decision}")
        print(f"Epoch {epoch}, Generator Loss {gen_loss}")
        # with torch.no_grad():
        #     random_noise = torch.randn((BATCH_SIZE, 100))
        #     generated_images = generator(random_noise).detach()
        
        # show_image(generated_images)

# %%

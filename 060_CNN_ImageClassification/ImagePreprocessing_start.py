#%%
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# %% import image
img = Image.open('kiki.jpg')
img

# %% compose a series of steps
preprocessing_steps = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomRotation(50),
    transforms.CenterCrop(200),
    transforms.Grayscale(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3422], std=[0.2596])
])

x = preprocessing_steps(img)
x
# %%
x.mean(), x.std()
# %%

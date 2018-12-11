import os
import numpy as np
import torch
from torchvision import datasets, transforms
from src import utils as ut
from src import preprocess as prep
from src.models.networks import Generator, ConditionalGenerator
from torchvision.utils import save_image

try:
	os.mkdir('images/gans')
except FileExistsError:
	pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# WGAN (epoch 30)
model_name = 'model=standard_conditional_wasserstein_gp_gan_epochs=50'
wgan = ConditionalGenerator(name=model_name).to(device)
ut.load_model_by_name(wgan, 30)
Z = torch.randn(10, 1, wgan.dim_z).repeat(1, 10, 1).reshape(100, wgan.dim_z).to(device)
Y = torch.arange(10).repeat(10).to(device)
X = torch.clamp((wgan(Z, Y) + 1) / 2., 0, 1)
save_image(X, 'images/gans/wgan_generated.png',nrow=10)

# SPECTRAL (epoch 50)
model_name = 'model=spectral_conditional_nonsaturating_gan_epochs=50'
sngan = ConditionalGenerator(name=model_name).to(device)
ut.load_model_by_name(sngan, 50)
Z = torch.randn(10, 1, sngan.dim_z).repeat(1, 10, 1).reshape(100, sngan.dim_z).to(device)
Y = torch.arange(10).repeat(10).to(device)
X = torch.clamp((sngan(Z, Y) + 1) / 2., 0, 1)
save_image(X, 'images/gans/spectral_generated.png',nrow=10)

# CLASS WGAN (epoch 50)
model_names = ['model=standard_wasserstein_gp_gan_epochs=50_label={}'.format(i) for i in range(10)]
G = [Generator(name).to(device) for name in model_names]
for i in range(len(G)): ut.load_model_by_name(G[i], 50)
Z = torch.randn(10, G[0].dim_z).to(device)
X = torch.clamp((G[0](Z) + 1) / 2., 0, 1)
for label in range(1, 10):
	Z = torch.randn(10, G[label].dim_z).to(device)
	NewX = torch.clamp((G[label](Z) + 1) / 2., 0, 1)
	X = torch.cat((X, NewX), 1)
X = X.reshape(100, 28, 28).unsqueeze(1)
save_image(X, 'images/gans/class_wgan_generated.png',nrow=10)

# CLASS SPECTRAL (epoch 50)
model_names = ['model=spectral_nonsaturating_gan_epochs=50_label={}'.format(i) for i in range(10)]
G = [Generator(name).to(device) for name in model_names]
for i in range(len(G)): ut.load_model_by_name(G[i], 50)
Z = torch.randn(10, G[0].dim_z).to(device)
X = torch.clamp((G[0](Z) + 1) / 2., 0, 1)
for label in range(1, 10):
	Z = torch.randn(10, G[label].dim_z).to(device)
	NewX = torch.clamp((G[label](Z) + 1) / 2., 0, 1)
	X = torch.cat((X, NewX), 1)
X = X.reshape(100, 28, 28).unsqueeze(1)
save_image(X, 'images/gans/class_spectral_generated.png',nrow=10)
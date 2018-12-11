import os
import numpy as np
import torch
from torchvision import datasets, transforms
from src import utils as ut
from src.models.FSVAE import FSVAE
from src.models.networks import Generator, ConditionalGenerator
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image

def save_images(data, augmentor, aug_epoch):
	first_out_dir = 'generated_data/{}'.format(augmentor)
	try:
	    os.mkdir(first_out_dir)
	except FileExistsError:
	    pass

	second_out_dir = '{}/{}'.format(first_out_dir, aug_epoch)
	try:
		os.mkdir(second_out_dir)
	except FileExistsError:
		return

	for i, img in enumerate(data):
		img = img.unsqueeze(0)
		save_image(img, '{}/{}_{}.png'.format(second_out_dir, augmentor, i))

def fsvae_augment(augment, augmentor, device):
	fsvae = FSVAE(name='model=fsvae').to(device)
	ut.load_model_by_name(fsvae, 1000000)

	with torch.no_grad():
		# Generate augment number of examples for each class
		Z = ut.duplicate(fsvae.sample_z(augment), 10)
		NewY = torch.eye(10).view(100, -1).repeat(1, augment).view(10, -1).t()
		NewY = torch.tensor(NewY, device=device)
		M = torch.clamp((fsvae.compute_mean_given(Z, NewY) + 1) / 2., 0, 1).view(augment * 10, 28, 28)
		if augment == 6000: save_images(M, augmentor, 0)
		NewX = (M * 255.).byte()
		NewY = torch.max(NewY, 1)[1].long()

		return NewX, NewY

def gan_augment(g, augment, augmentor, aug_epoch, device):
	ut.load_model_by_name(g, aug_epoch)

	with torch.no_grad():
		BATCH = 100
		Z = torch.randn(BATCH, 1, g.dim_z).repeat(1, 10, 1).reshape(BATCH * 10, g.dim_z).to(device)
		Y = torch.arange(10).repeat(BATCH).to(device)
		NewX = torch.clamp((g(Z, Y) + 1) / 2., 0, 1).squeeze(1)
		NewX = torch.Tensor.numpy(NewX.cpu())

		for _ in range(1, augment//BATCH):
			Z = torch.randn(BATCH, 1, g.dim_z).repeat(1, 10, 1).reshape(BATCH * 10, g.dim_z).to(device)
			Y = torch.arange(10).repeat(BATCH).to(device)
			X = torch.clamp((g(Z, Y) + 1) / 2., 0, 1).squeeze(1)
			NewX = np.concatenate((NewX, torch.Tensor.numpy(X.cpu())), 0)	

		NewX = torch.from_numpy(NewX)
		if augment == 6000: save_images(NewX, augmentor, aug_epoch)
		NewX = (NewX * 255.).byte()

		return NewX, Y.long()

def class_gan_augment(G, augment, augmentor, aug_epoch, device):
	for i in range(len(G)): ut.load_model_by_name(G[i], aug_epoch)

	with torch.no_grad():
		Z = torch.randn(augment, G[0].dim_z).to(device)
		Y = torch.zeros(augment)
		NewX = torch.clamp((G[0](Z) + 1) / 2., 0, 1).squeeze(1)
		NewX = torch.Tensor.numpy(NewX.cpu())

		for label in range(1, 10):
			Z = torch.randn(augment, G[label].dim_z).to(device)
			Y = torch.cat((Y, torch.ones(augment) * label), 0)
			X = torch.clamp((G[label](Z) + 1) / 2., 0, 1).squeeze(1)
			NewX = np.concatenate((NewX, torch.Tensor.numpy(X.cpu())), 0)

		NewX = torch.from_numpy(NewX)
		if augment == 6000: save_images(NewX, augmentor, aug_epoch)
		NewX = (NewX * 255.).byte()

		return NewX, Y.long()

def augment_data(train_loader, augment, augmentor, aug_epoch, batch_size, device):
	# Get original data
	X, Y = train_loader.dataset.train_data, train_loader.dataset.train_labels

	# Get generated data
	if augmentor == 'fsvae':
		NewX, NewY = fsvae_augment(augment, augmentor, device)
	elif augmentor == 'wgan':
		model_name = 'model=standard_conditional_wasserstein_gp_gan_epochs=50'
		wgan = ConditionalGenerator(name=model_name).to(device)
		NewX, NewY = gan_augment(wgan, augment, augmentor, aug_epoch, device)
	elif augmentor == 'spectral':
		model_name = 'model=spectral_conditional_nonsaturating_gan_epochs=50'
		sngan = ConditionalGenerator(name=model_name).to(device)
		NewX, NewY = gan_augment(sngan, augment, augmentor, aug_epoch, device)
	elif augmentor == 'c_wgan':
		model_names = ['model=standard_wasserstein_gp_gan_epochs=50_label={}'.format(i) for i in range(10)]
		G = [Generator(name).to(device) for name in model_names]
		NewX, NewY = class_gan_augment(G, augment, augmentor, aug_epoch, device)
	elif augmentor == 'c_spectral':
		model_names = ['model=spectral_nonsaturating_gan_epochs=50_label={}'.format(i) for i in range(10)]
		G = [Generator(name).to(device) for name in model_names]
		NewX, NewY = class_gan_augment(G, augment, augmentor, aug_epoch, device)
	else:
		raise NotImplementedError

	NewData = torch.cat((X, NewX.to(X.device)), 0)
	NewLabels = torch.cat((Y, NewY.to(Y.device)), 0)

	# Shuffle new dataset for randomness
	perm = torch.randperm(NewLabels.shape[0])
	NewData = NewData[perm]
	NewLabels = NewLabels[perm]

	# Update train loader with new data
	train_loader.dataset.train_data = NewData
	train_loader.dataset.train_labels = NewLabels

	return train_loader


def get_fashion_mnist_data(device, batch_size=100, download=True, augment=0, augmentor='fsvae', aug_epoch=50, preprocess=transforms.ToTensor()):
	train_loader = torch.utils.data.DataLoader(
		datasets.FashionMNIST('data', train=True, download=download, transform=preprocess),
		batch_size=batch_size,
		shuffle=True)
	test_loader = torch.utils.data.DataLoader(
		datasets.FashionMNIST('data', train=False, download=download, transform=preprocess),
		batch_size=batch_size,
		shuffle=True)

	if augment > 0: return augment_data(train_loader, augment, augmentor, aug_epoch, batch_size, device), test_loader
	else: return train_loader, test_loader

def get_fashion_mnist_data_classes(device, batch_size=100, download=True):
	train_loaders = []
	preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,), (.5,))])
	dataset = datasets.FashionMNIST('data', train=True, download=download, transform=preprocess)
	for label in range(10):
		indices = (dataset.train_labels == label).nonzero().squeeze(1)
		sampler = SubsetRandomSampler(indices)
		train_loader = torch.utils.data.DataLoader(
			dataset,
			batch_size=batch_size,
			sampler=sampler)
		train_loaders.append(train_loader)
	return train_loaders

import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

preprocess = transforms.ToTensor()
train_loader = torch.utils.data.DataLoader(
	datasets.FashionMNIST('data', train=True, download=True, transform=preprocess),
	batch_size=1,
	shuffle=True)

# Visualize data with labels
# indices = (train_loader.dataset.train_labels == 0).nonzero()
# perm = torch.randperm(indices.shape[0])
# indices = indices[perm][:10].squeeze(1)
# Images = train_loader.dataset.train_data[indices]
# print(Images.shape)
# for label in range(1, 100):
# 	indices = (train_loader.dataset.train_labels == label).nonzero()
# 	perm = torch.randperm(1)
# 	indices = indices[perm][:10].squeeze(1)
# 	imgs = train_loader.dataset.train_data[indices]
# 	Images = torch.cat((Images, imgs), 0)

indices = (train_loader.dataset.train_labels == 0).nonzero()
perm = torch.randperm(indices.shape[0])
Images = train_loader.dataset.train_data[indices[perm][0]]
for label in range(1, 100):
	label = label % 10
	indices = (train_loader.dataset.train_labels == label).nonzero()
	perm = torch.randperm(indices.shape[0])
	imgs = train_loader.dataset.train_data[indices[perm][0]]
	Images = torch.cat((Images, imgs), 0)

Images = Images.view(100, 1, 28, 28).float() / 255.
save_image(Images, 'images/original_images.png', nrow=10)


# torch.utils.data.DataLoader(TensorDataset(tensor.from_numpy(numpy_array)), batch_size=100, shuffle=True)


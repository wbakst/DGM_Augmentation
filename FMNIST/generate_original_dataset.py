import os
import numpy as np
import torch
from torchvision import datasets, transforms
from src import utils as ut
from src import preprocess as prep
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, test_loader = prep.get_fashion_mnist_data(device, batch_size=1)

out_dir = 'generated_data/original_images'
try:
    os.mkdir(out_dir)
except FileExistsError:
    pass

for i, (x, y) in enumerate(train_loader):
	img = ((x + 1) / 2.).squeeze(0)
	save_image(img, '{}/{}.png'.format(out_dir, i))
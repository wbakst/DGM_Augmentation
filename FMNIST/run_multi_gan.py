import argparse
import os
import time
import torch
import torchvision
import tqdm
from pprint import pprint

from src import preprocess as prep
from src import utils as ut
from src.models import GAN
from src.models import networks
from src.train import train_gan

parser = argparse.ArgumentParser(description='Trains a simple GAN.')
parser.add_argument('--epochs',    default=30,              type=int, help='number of epochs to run for')
parser.add_argument('--batch',     default=100,             type=int, help='batch size for training')
parser.add_argument('--loss_type', default='nonsaturating', type=str, help='loss to train the gan with (nonsaturating, wasserstein_gp)')
parser.add_argument('--d_type',    default='standard',      type=str, help='discriminator to train (standard, spectral)')
args = parser.parse_args()
layouts = [
    [('model={:s}', '{}_{}_gan'.format(args.d_type, args.loss_type)),
     ('epochs={:02d}', args.epochs),
     ('label={}', i)]
for i in range(10)]
model_names = ['_'.join([t.format(v) for (t, v) in layout]) for layout in layouts]
pprint(vars(args))
for i, name in enumerate(model_names):
  print('Model name (label={}):'.format(i), name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get dataset split into classes
train_loaders = prep.get_fashion_mnist_data_classes(device, batch_size=args.batch)

#########################################
############### Generator ###############
#########################################

G = [networks.Generator(model_names[i]).to(device) for i in range(10)]

#########################################
############# Discriminator #############
#########################################

if args.d_type == 'standard':
    D = [networks.Discriminator().to(device) for _ in range(10)]
elif args.d_type == 'spectral':
    D = [networks.SpectralDiscriminator().to(device) for _ in range(10)]
else:
    raise NotImplementedError

#########################################
################# Loss ##################
#########################################

if args.loss_type == 'nonsaturating':
    loss = GAN.loss_nonsaturating
elif args.loss_type == 'wasserstein_gp':
    loss = GAN.loss_wasserstein_gp
else:
    raise NotImplementedError

#########################################
############### Training ################
#########################################

for i in range(10):
  train_gan(g=G[i],
            d=D[i],
            loss=loss,
            train_loader=train_loaders[i],
            device=device,
            tqdm=tqdm,
            epochs=args.epochs,
            model_name=model_names[i])
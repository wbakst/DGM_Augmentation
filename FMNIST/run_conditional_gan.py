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
parser.add_argument('--epochs',    default=30,                          type=int, help='number of epochs to run for')
parser.add_argument('--batch',     default=100,                         type=int, help='batch size for training')
parser.add_argument('--loss_type', default='conditional_nonsaturating', type=str, help='loss to train the gan with (nonsaturating, wasserstein_gp)')
parser.add_argument('--d_type',    default='standard',                  type=str, help='discriminator to train (standard, spectral)')
args = parser.parse_args()
layout = [
    ('model={:s}', '{}_{}_gan'.format(args.d_type, args.loss_type)),
    ('epochs={:02d}', args.epochs),
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = prep.get_fashion_mnist_data(device, batch_size=args.batch)

#########################################
############### Generator ###############
#########################################

g = networks.ConditionalGenerator(model_name).to(device)

#########################################
############# Discriminator #############
#########################################

if args.d_type == 'standard':
    d = networks.ConditionalDiscriminator().to(device)
elif args.d_type == 'spectral':
    d = networks.ConditionalSpectralDiscriminator().to(device)
else:
    raise NotImplementedError

#########################################
################# Loss ##################
#########################################

if args.loss_type == 'conditional_nonsaturating':
    loss = GAN.conditional_loss_nonsaturating
elif args.loss_type == 'conditional_wasserstein_gp':
    loss = GAN.conditional_loss_wasserstein_gp
else:
    raise NotImplementedError

#########################################
############### Training ################
#########################################

train_gan(g=g,
          d=d,
          loss=loss,
          train_loader=train_loader,
          device=device,
          tqdm=tqdm,
          epochs=args.epochs,
          model_name=model_name,
          conditional=True)
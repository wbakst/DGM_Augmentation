import argparse
import numpy as np
import torch
import tqdm
from src import preprocess as prep
from src import utils as ut
from src.train import train_fsvae
from src.models.FSVAE import FSVAE
from pprint import pprint
from torchvision.utils import save_image

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--iter_max',  type=int, default=1000000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000,   help="Save model every n iterations")
parser.add_argument('--batch',     type=int, default=100,     help="Batch size")
args = parser.parse_args()
layout = [
    ('model={:s}',  'fsvae')
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, _ = prep.get_fashion_mnist_data(device, batch_size=args.batch)
fsvae = FSVAE(name=model_name).to(device)

# train_fsvae(model=fsvae,
#       train_loader=train_loader,
#       device=device,
#       tqdm=tqdm.tqdm,
#       iter_max=args.iter_max,
#       iter_save=args.iter_save)

# Visualization Code
MODELS = [FSVAE(name=model_name).to(device) for i in range(10)]
for i, model in enumerate(MODELS):
  checkpoint = (i+1)*100000
  ut.load_model_by_name(model, checkpoint)

for i, model in enumerate(MODELS):
  checkpoint = (i+1)*100000

  # Z = ut.duplicate(model.sample_z(10), 10)
  # Y = torch.eye(10).view(100, -1).repeat(1, 10).view(10, -1).t()
  # Y = torch.tensor(Y, device=device)
  Z = model.sample_z(10).view(100, -1).repeat(1, 10).view(10, -1).t()
  Y = ut.duplicate(torch.eye(10), 10).to(device)
  M = (model.compute_mean_given(Z, Y) + 1) / 2.
  M = torch.clamp(M, 0, 1).view(100, 1, 28, 28)
  save_image(M, 'images/fsvae/fsvae_generated_images_{}.png'.format(checkpoint), nrow=10)

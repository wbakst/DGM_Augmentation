import argparse
import numpy as np
import torch
import tqdm
from src import preprocess as prep
from src import utils as ut
from src.train import train_cnn
from src.models.CNN import BasicCNN, AdvancedCNN
from pprint import pprint

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs',    type=int,  default=10,       help="Number of training epochs")
parser.add_argument('--batch',     type=int,  default=100,      help="Batch size for training")
parser.add_argument('--train',     type=int,  default=1,        help="Flag for training")
parser.add_argument('--eval',      type=int,  default=0,        help="Flag for evaluation. 0 is for train, 1 is for test. Ignored if --train is set to 1")
parser.add_argument('--type',      type=str,  default='basic',  help="Which type of CNN to run. Options: basic, advanced")
parser.add_argument('--augment',   type=int,  default=0,        help="Number of each example to generate")
parser.add_argument('--augmentor', type=str,  default='fsvae',  help='Which generative model to use for augmentation (fsvae, wgan, spectral, c_wgan, c_spectral)')
parser.add_argument('--aug_epoch', type=int,  default=50,       help='Which epoch of augmentor to use for generation (1, 10, 20, 30, 40, 50)')
args = parser.parse_args()
layout = [
    ('model={:s}', '{}_cnn'.format(args.type)),
    ('epochs={:02d}', args.epochs),
]
if args.augment > 0:
  layout.append(('augmented={}', args.augment))
  layout.append(('augmentor={}', args.augmentor))
  layout.append(('aug_epoch={}', args.aug_epoch))
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = prep.get_fashion_mnist_data(device, batch_size=args.batch, augment=args.augment, augmentor=args.augmentor, aug_epoch=args.aug_epoch)
if args.type == 'advanced':
  cnn = AdvancedCNN(model_name).to(device)
elif args.type == 'basic':
  cnn = BasicCNN(model_name).to(device)
else:
  print("Unsupported model type specified. Please specify basic or advanced")
  exit()

if args.train:
    train_cnn(model=cnn,
          train_loader=train_loader,
          device=device,
          tqdm=tqdm,
          epochs=args.epochs)
    ut.evaluate_classifier(cnn, train_loader, device)

else:
  ut.load_model_by_name(cnn, epoch=args.epochs)
  ut.evaluate_classifier(cnn, train_loader if not args.eval else test_loader, device, args.eval)

# Visualization of Data







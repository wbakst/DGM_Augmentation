import argparse
import numpy as np
import torch
import tqdm
from FID.inception import InceptionV3
from src import preprocess as prep
from src import utils as ut
from src.train import train_inception
from pprint import pprint
from torchvision import datasets, transforms
from torchvision.models import inception_v3
from torch.nn import functional as F

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs',    type=int,  default=10,       help="Number of training epochs")
parser.add_argument('--batch',     type=int,  default=30,       help="Batch size for training")
parser.add_argument('--train',     type=int,  default=1,        help="Flag for training")
parser.add_argument('--eval',      type=int,  default=0,        help="Flag for evaluation. 0 is for train, 1 is for test. Ignored if --train is set to 1")
parser.add_argument('--dataset',   type=str,  default='fmnist',   help="Which dataset to use (fmnist, omniglot)")
args = parser.parse_args()
layout = [
    ('model={:s}', 'InceptionV3_{}'.format(args.dataset)),
    ('epochs={:02d}', args.epochs),
    ('dataset={}', args.dataset)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

preprocess = transforms.Compose([
  transforms.Grayscale(3),
  transforms.ToTensor()
])
if args.dataset == 'fmnist':
  train_loader, test_loader = prep.get_fashion_mnist_data(device, batch_size=args.batch, preprocess=preprocess)
else:
  raise NotImplementedError

inception = inception_v3(pretrained=False, num_classes=10).to(device)

if args.train:
    train_inception(model=inception,
          train_loader=train_loader,
          device=device,
          tqdm=tqdm,
          epochs=args.epochs,
          model_name=model_name)


eval_loader = train_loader if not args.eval else test_loader
print('*' * 80)
print("CLASSIFICATION EVALUATION ON {} SET".format("TEST" if not args.train else "TRAINING"))
print('*' * 80)

correct, total = 0, 0
for i, data in enumerate(eval_loader):
    # Extract data
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    inputs = F.interpolate(inputs, size=(299, 299), mode='bilinear')

    outputs = inception(inputs)

    _, preds = torch.max(outputs[0], 1) 

    correct += (labels == preds).float().sum()

    total += inputs.shape[0]
print("Accuracy: {}".format(correct/total))

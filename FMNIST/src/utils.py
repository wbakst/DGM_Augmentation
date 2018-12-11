import os
import torch
from torch.nn import functional as F
from torchvision import transforms

def reset_weights(m):
    try:
        m.reset_parameters()
    except AttributeError:
        pass

def save_model_by_name(model, epoch, model_name=None):
    if model_name is not None:
        save_dir = os.path.join('checkpoints', model_name)
    else:
        save_dir = os.path.join('checkpoints', model.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, 'model-{}.pt'.format(epoch))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))

def load_model_by_name(model, epoch, model_name=None):
    name = model.name if model_name is None else model_name
    file_path = os.path.join('checkpoints',
                             name,
                             'model-{}.pt'.format(epoch))
    state = torch.load(file_path)
    model.load_state_dict(state)
    print("Loaded from {}".format(file_path))

def evaluate_classifier(model, loader, device, test=False):
    print('*' * 80)
    print("CLASSIFICATION EVALUATION ON {} SET".format("TEST" if test else "TRAINING"))
    print('*' * 80)

    correct, total = 0, 0
    for i, data in enumerate(loader):
        # Extract data
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        preds = model.classify(model.forward(inputs))
        correct += (labels == preds).float().sum()

        total += inputs.shape[0]
    print("Accuracy: {}".format(correct/total))

def sample_gaussian(m, v):
    e = torch.randn_like(m)
    z = (e * torch.sqrt(v)) + m
    return z

def log_normal(x, m, v):
    log_prob = torch.distributions.normal.Normal(m, torch.sqrt(v)).log_prob(x).sum(-1)
    return log_prob

def gaussian_parameters(h, dim=-1):
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v

def kl_normal(qm, qv, pm, pv):
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl

def duplicate(x, rep):
    return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])
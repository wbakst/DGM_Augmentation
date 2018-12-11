import torch
from torch.nn import functional as F

def loss_nonsaturating(g, d, x_real, *, device):
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    d_loss = -F.logsigmoid(d(x_real)).mean() + F.binary_cross_entropy_with_logits(d(g(z)), torch.zeros(batch_size).to(device))
    g_loss = -F.logsigmoid(d(g(z))).mean()
    
    return d_loss, g_loss

def conditional_loss_nonsaturating(g, d, x_real, y, *, device):
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    d_loss = -F.logsigmoid(d(x_real, y)).mean() + F.binary_cross_entropy_with_logits(d(g(z, y), y), torch.zeros(batch_size).to(device))
    g_loss = -F.logsigmoid(d(g(z, y), y)).mean()

    return d_loss, g_loss

def loss_wasserstein_gp(g, d, x_real, *, device):
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # Get x_fake
    x_fake = g(z)

    # Calculate penalty
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    r = torch.autograd.Variable((alpha * x_fake) + ((1 - alpha) * x_real), requires_grad=True).to(device)
    grads = torch.autograd.grad(d(r), r, grad_outputs=torch.ones(batch_size).to(device), create_graph=True)[0]
    penalty = 10 * ((grads.view(batch_size, -1).norm(dim=1, p=2) - 1) ** 2).mean()

    # Calculate loss
    d_loss = d(x_fake).mean() - d(x_real).mean() + penalty
    g_loss = -d(x_fake).mean()

    return d_loss, g_loss

def conditional_loss_wasserstein_gp(g, d, x_real, y, *, device):
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # Get x_fake
    x_fake = g(z, y)

    # Calculate penalty
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    r = torch.autograd.Variable((alpha * x_fake) + ((1 - alpha) * x_real), requires_grad=True).to(device)
    grads = torch.autograd.grad(d(r, y), r, grad_outputs=torch.ones(batch_size).to(device), create_graph=True)[0]
    penalty = 10 * ((grads.view(batch_size, -1).norm(dim=1, p=2) - 1) ** 2).mean()

    # Calculate loss
    d_loss = d(x_fake, y).mean() - d(x_real, y).mean() + penalty
    g_loss = -d(x_fake, y).mean()

    return d_loss, g_loss

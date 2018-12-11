import argparse
import numpy as np
import os
from src import utils as ut
import torch
from torch import nn, optim
from torchvision.utils import save_image
from torch.nn import functional as F

def train_cnn(model, train_loader, device, tqdm, epochs=10, model_name='model', reinitialize=False):
    if reinitialize:
        model.apply(ut.reset_weights)

    #########################################
    ############### Optimizer ###############
    #########################################

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    #########################################
    ############# Training Loop #############
    #########################################

    for e in range(1, epochs+1):
        for inputs, labels in tqdm.tqdm(train_loader):
            # Extract data
            inputs, labels = inputs.to(device), labels.to(device)

            # Reset parameter gradients for each loop of optimization
            optimizer.zero_grad()

            # Forward and Backward pass
            outputs = model.forward(inputs)
            loss = model.loss(outputs, labels)
            # acc = (labels == model.classify(outputs)).float().mean()
            loss.backward()

            # Gradient Optimization Step
            optimizer.step()

        # Save model after each epoch
        # ut.save_model_by_name(model, e)
    # Save final model for evaluation
    ut.save_model_by_name(model, epochs)

def train_inception(model, train_loader, device, tqdm, epochs=10, model_name='model', reinitialize=False):
    if reinitialize:
        model.apply(ut.reset_weights)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for e in range(1, epochs+1):
        for inputs, labels in tqdm.tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = F.interpolate(inputs, size=(299, 299), mode='bilinear')

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs[0], labels) + nn.CrossEntropyLoss()(outputs[1], labels)
            loss.backward()

            optimizer.step()
    ut.save_model_by_name(model, epochs, model_name)

def train_fsvae(model, train_loader, device, tqdm, iter_max=np.inf, iter_save=np.inf, model_name='model', reinitialize=False):
    if reinitialize:
        model.apply(ut.reset_weights)

    #########################################
    ############### Optimizer ###############
    #########################################

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    #########################################
    ############# Training Loop #############
    #########################################

    i = 0
    with tqdm(total=iter_max) as pbar:
        while True:
            for batch_idx, (xu, yu) in enumerate(train_loader):
                i += 1
                optimizer.zero_grad()

                xu = xu.to(device).reshape(xu.size(0), -1)
                yu = yu.new(np.eye(10)[yu]).to(device).float()
                loss = model.loss(xu, yu)

                loss.backward()
                optimizer.step()

                pbar.set_postfix(
                    loss='{:.2e}'.format(loss)
                    )
                pbar.update(1)

                if i % iter_save == 0:
                    ut.save_model_by_name(model, i)

                if i == iter_max:
                    return

def train_gan(g, d, loss, train_loader, device, tqdm, epochs, model_name='model', conditional=False, reinintialize=False):
    if reinintialize:
        model.apply(ut.reset_weights)

    #########################################
    ############### Optimizer ###############
    #########################################

    g_optimizer = torch.optim.Adam(g.parameters(), 1e-3, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(d.parameters(), 1e-3, betas=(0.5, 0.999))

    #########################################
    ############# Training Loop #############
    #########################################

    z_test = torch.randn(10, 1, g.dim_z).repeat(1, 10, 1).reshape(100, g.dim_z).to(device)
    y_test = torch.arange(10).repeat(10).to(device)

    global_step = 0
    for e in range(1, epochs+1):
        for x_real, y_real in tqdm.tqdm(train_loader):
            x_real, y_real = x_real.to(device), y_real.to(device)

            if conditional:
                d_loss, g_loss = loss(g, d, x_real, y_real, device=device)
            else:
                d_loss, g_loss = loss(g, d, x_real, device=device)

            d_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            global_step += 1

            # if global_step % 50 == 0:
            #     with torch.no_grad():
            #         g.eval()
            #         if conditional:
            #             x_test = (g(z_test, y_test) + 1) / 2.
            #         else:
            #             x_test = (g(z_test) + 1) / 2.
            #         save_image(x_test, '{}/gans/{}_fake_{}.png'.format('images', model_name, global_step),nrow=10)
            #         g.train()

        ut.save_model_by_name(g, e)
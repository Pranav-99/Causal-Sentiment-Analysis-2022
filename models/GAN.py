from enum import auto
import torch
from torch import nn
import torch.autograd as autograd

import torch.nn.functional as F
import random


class GeneratorNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, nhidden=2):
        super(GeneratorNet,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LeakyReLU())

        for _ in range(nhidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DiscriminatorNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DiscriminatorNet,self).__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.layer1 = nn.Linear(input_dim , hidden_dim)
        self.layer2 = nn.Linear(hidden_dim , output_dim) # softmax is baked into the loss anyway

    def forward(self,x):
        out = F.leaky_relu(self.layer1(x))
        out = self.layer2(out)
        return out


class GANmodel:

    def __init__(self, gan_config):

        self.gan_config = gan_config
        self.device = gan_config['device']
        self.g = GeneratorNet(gan_config['g_input_dim'],
                              gan_config['g_hidden_dim'],
                              gan_config['g_output_dim'],
                              gan_config['g_nhidden'],
                              ).to(self.device)
        self.d = DiscriminatorNet(gan_config['g_output_dim'],
                              gan_config['d_hidden_dim'],
                              gan_config['d_output_dim']
                              ).to(self.device)
 
        # TODO: self.n_d_steps_per_g_step = gan_config['n_d_steps_per_g_step'] 
        self.g_lr = gan_config['g_lr'] 
        self.d_lr = gan_config['d_lr'] 
        self.epochs = gan_config['epochs'] 
        self.gamma = gan_config['gamma']
        self.batch_size = gan_config['batch_size']
        self.mode = gan_config['mode']
        assert self.mode in {'GAN', 'LSGAN', 'WGAN-GP'}
        self.disc_steps = gan_config['disc_steps']

    def discriminator_loss(self, discrim_real, discrim_fake, discrim_interp=None, interp=None, lamb=10.):
        if self.mode == 'WGAN-GP':
            loss = discrim_fake.mean() - discrim_real.mean()

            grad_wrt_interp = autograd.grad(
                outputs=(discrim_interp).sum(),
                inputs=interp,
                create_graph=True,
            )[0]
            grad_norm = torch.norm(grad_wrt_interp.view(grad_wrt_interp.size(0), -1), 2, dim=1)

            grad_loss = lamb * ((grad_norm - 1) ** 2).mean()

            loss += grad_loss
            return loss

        elif self.mode == 'GAN':
            ## TODO: Merge with lsgan, clean up
            loss_fn = torch.nn.BCEWithLogitsLoss()
            targets = torch.ones((len(discrim_real), 1), device=self.device)
            loss_real = loss_fn(discrim_real, targets)
            targets = torch.zeros((len(discrim_fake), 1), device=self.device)
            loss_fake = loss_fn(discrim_fake, targets)

            return loss_real + loss_fake

        elif self.mode == 'LSGAN':
            bce_loss_fn = torch.nn.MSELoss()
            targets = torch.ones((len(discrim_real), 1), device=self.device)
            loss_real = bce_loss_fn(discrim_real, targets)
            targets = torch.zeros((len(discrim_fake), 1), device=self.device)
            loss_fake = bce_loss_fn(discrim_fake, targets)

            return loss_real + loss_fake

        else:
            raise ValueError(f"Unknown GAN mode: {self.mode}")

    def generator_loss(self, discrim_fake):
        if self.mode == 'WGAN-GP':
            return -(discrim_fake).sum() / len(discrim_fake)
        elif self.mode == 'GAN':
            loss_fn = torch.nn.BCEWithLogitsLoss()
            targets = torch.ones((len(discrim_fake), 1), device=self.device)
            loss_fake = loss_fn(discrim_fake, targets)
            return loss_fake
        elif self.mode == 'LSGAN':
            loss_fn = torch.nn.MSELoss()
            targets = torch.ones((len(discrim_fake), 1), device=self.device)
            loss_fake = loss_fn(discrim_fake, targets)
            return loss_fake
        else:
            raise ValueError(f"Unknown GAN mode: {self.mode}")

    def train(self, train_data, validation_data):

        # Train data: Torch tensor (n_train x z_dim)
        # validation data: Torch tensor (n_val x z_dim)

        # train_data = train_data.to(self.device)
        # validation_data = validation_data.to(self.device)

        n_train = train_data.shape[0]
        n_valid = validation_data.shape[0]

        g_optimizer = torch.optim.Adam(self.g.parameters(), lr=self.g_lr)
        d_optimizer = torch.optim.Adam(self.d.parameters(), lr=self.d_lr)
        g_scheduler = torch.optim.lr_scheduler.ExponentialLR(g_optimizer, gamma=self.gamma)
        d_scheduler = torch.optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=self.gamma)

        self.d_losses_train = []
        self.g_losses_train = []
        self.d_losses_val = []
        self.g_losses_val = []

        for epoch in range(self.epochs): 
            self.g.train()
            self.d.train()
            loss_d_train = 0
            loss_g_train = 0
            loss_d_val = 0
            loss_g_val = 0

            for batch_start in range(0, n_train, self.batch_size):

                batch_real = train_data[batch_start : batch_start + self.batch_size].to(self.device)

                if random.random() < 1.0 / self.disc_steps:
                    ### Update Generator ###
                    batch_fake_data = torch.randn(len(batch_real), self.g.input_dim, device=self.device)

                    d_output = self.d(self.g(batch_fake_data))
                    loss_fake_g = self.generator_loss(d_output)

                    loss_g_train += loss_fake_g.detach().cpu().numpy()

                    g_optimizer.zero_grad()
                    loss_fake_g.backward()
                    g_optimizer.step()

                ###  Update discriminator ### 

                batch_fake_data = torch.randn(len(batch_real), self.g.input_dim, device=self.device)
                with torch.no_grad():
                    batch_fake = self.g(batch_fake_data)

                d_output_real = self.d(batch_real)
                d_output_fake = self.d(batch_fake)

                if self.mode == 'WGAN-GP':
                    eps = torch.rand((len(batch_real), 1)).to(self.device)
                    interp = eps * batch_fake + (1 - eps) * batch_real
                    interp = autograd.Variable(interp, requires_grad=True)
                    discrim_interp = self.d(interp)
                else:
                    interp = discrim_interp = None

                loss_d = self.discriminator_loss(d_output_real, d_output_fake, discrim_interp, interp)

                loss_d_train += loss_d.detach().cpu().numpy() 
                d_optimizer.zero_grad()
                loss_d.backward()
                d_optimizer.step()

            self.g.eval()
            self.d.eval()

            for batch_start in range(0, n_valid, self.batch_size):

                batch_real = validation_data[batch_start : batch_start + self.batch_size].to(self.device)
                batch_fake_data = torch.randn(len(batch_real), self.g.input_dim, device=self.device)

                batch_fake = self.g(batch_fake_data)
                d_output_fake = self.d(batch_fake)
                d_output_real = self.d(batch_real)

                if self.mode == 'WGAN-GP':
                    eps = torch.rand((len(batch_real), 1)).to(self.device)
                    interp = eps * batch_fake + (1 - eps) * batch_real
                    interp = autograd.Variable(interp, requires_grad=True)
                    discrim_interp = self.d(interp)
                else:
                    interp = discrim_interp = None

                g_loss = self.generator_loss(d_output_fake)
                d_loss = self.discriminator_loss(d_output_real, d_output_fake, discrim_interp, interp)

                loss_d_val += g_loss.detach().cpu().numpy()
                loss_g_val += d_loss.detach().cpu().numpy()

            g_scheduler.step(loss_g_val/n_valid)
            d_scheduler.step(loss_d_val/n_valid)

            self.d_losses_train.append(loss_d_train/n_train)
            self.g_losses_train.append(loss_g_train/n_train)

            self.d_losses_val.append(loss_d_val/n_valid)
            self.g_losses_val.append(loss_g_val/n_valid) 

            print("Epoch {}:\nTrain dloss: {} gloss: {},\nValid dloss: {} gloss: {}\n".format(
                epoch, self.d_losses_train[-1], self.g_losses_train[-1], self.d_losses_val[-1], self.g_losses_val[-1]))


    def drawsamples(self, N, get_tensor=False):

        random_latents = torch.randn(N, self.g.input_dim, device=self.device)
        with torch.no_grad():
            outputs = self.g(random_latents) 

        if(get_tensor): 
            return outputs
            
        else:
            return outputs.cpu().numpy()


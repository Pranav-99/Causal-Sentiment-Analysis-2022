import torch
from torch import nn
import torch.nn.Functional as F

class GeneratorNet(nn.Module): 

    def __init__(self,input_dim,hidden_dim,output_dim,): 
        super(GeneratorNet,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.layer1 = nn.Linear(input_dim , hidden_dim)
        self.layer2 = nn.Linear(hidden_dim , output_dim)

    def forward(self,x):
        out = F.ReLU(self.layer1(x))
        out = self.layer2(out) 

class DiscriminatorNet(nn.Module):

    def __init__(self,input_dim,hidden_dim,output_dim,): 
        super(DiscriminatorNet,self).__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim 

        self.layer1 = nn.Linear(input_dim , hidden_dim)
        self.layer2 = nn.Linear(hidden_dim , output_dim) # softmax is baked into the loss anyway

    def forward(self,x): 
        out = F.ReLU(self.layer1(x))
        out = self.layer2(out)

class GANmodel: 

    def __init__(self,gan_config):

        self.gan_config = gan_config
        self.device = gan_config['device']
        self.g = GeneratorNet(gan_config['g_input_dim'],
                              gan_config['g_hidden_dim'],
                              gan_config['g_output_dim']
                              ).to(self.device)
        self.d = DiscriminatorNet(gan_config['g_output_dim'],
                              gan_config['d_hidden_dim'],
                              gan_config['d_output_dim']
                              ).to(self.device)
 
        self.n_d_steps_per_g_step = gan_config['n_d_steps_per_g_step'] 
        self.g_lr = gan_config['g_lr'] 
        self.d_lr = gan_config['d_lr'] 
        self.epochs = gan_config['epochs'] 
        self.gamma = gan_config['gamma']
        self.batch_size = gan_config['batch_size'] 


    def train(self,train_data,validation_data): 

        # Train data: Torch tensor (n_train x z_dim)
        # validation data: Torch tensor (n_val x z_dim)
        n_train = train_data.shape[0]
        n_valid = validation_data.shape[0]
        loss_fn = nn.CrossEntropyLoss()
        g_optimizer = torch.optim.Adam(self.g.parameters(), lr=self.lr)
        d_optimizer = torch.optim.Adam(self.d.parameters(), lr=self.lr)
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

            for batch_start in range(0,n_train, self.batch_size):

                self.d_optimizer.zero_grad()
                self.g_optimizer.zero_grad()

                ###  Update discriminator ### 
                
                batch_real = train_data[batch_start : batch_start + self.batch_size]
                labels_real = torch.full((self.batch_size,), 1 , dtype=torch.float, device=self.device)
                
                batch_fake_data = torch.randn(self.batch_size, self.g.input_dim, 1, 1, device=self.device)
                labels_fake = torch.full((self.batch_size,), 0 , dtype=torch.float, device=self.device)

                # TODO : ADD SUPPORT TO SHUFFLE/MIX THE DATA FOR DISCRIMINATOR TRAINING. IS THIS DONE USUALLY? 
                # TODO: ADD SUPPORT FOR TRAINING SEVERAL STEPS OF DISCRIMINATOR FOR ONE GENERATOR STEP

                with torch.no_grad(): 
                    batch_fake = self.g(batch_fake_data)

                d_output_real = self.d(batch_real) 
                loss_real_d = loss_fn(d_output_real , labels_real) 
                loss_real_d.backward()

                loss_d_train += loss_real_d.detach().cpu().numpy() # detach is not inplace so we are not losing gradient information

                d_output_fake = self.d(batch_fake) 
                loss_fake_d = loss_fn(d_output_fake , labels_fake)
                loss_fake_d.backward()

                loss_d_train += loss_fake_d.detach().cpu().numpy()

                d_optimizer.step()

                ### Update Generator ###

                batch_fake_data = torch.randn(self.batch_size, self.g.input_dim, 1, 1, device=self.device)
                labels_fake = torch.full((self.batch_size,), 1 , dtype=torch.float, device=self.device)

                d_output = self.d(self.g(batch_fake_data))
                loss_fake_g = loss_fn(d_output, labels_fake)

                loss_fake_g.backward()

                loss_g_train += loss_fake_g.detach().cpu().numpy()
                
                g_optimizer.step()

            self.g.eval()
            self.d.eval()

            for batch_start in range(0,n_valid, self.batch_size):

                batch_real = validation_data[batch_start : batch_start + self.batch_size]
                labels_real = torch.full((self.batch_size,), 1 , dtype=torch.float, device=self.device)
                
                batch_fake_data = torch.randn(self.batch_size, self.g.input_dim, 1, 1, device=self.device)
                labels_fake_d = torch.full((self.batch_size,), 0 , dtype=torch.float, device=self.device)
                labels_fake_g = torch.full((self.batch_size,), 1 , dtype=torch.float, device=self.device)

                with torch.no_grad():
                    batch_fake = self.g(batch_fake_data)
                    d_output_real = self.d(batch_real)
                    d_output_fake = self.d(batch_fake)
                    loss_d_real = loss_fn(d_output_real , labels_real)
                    loss_d_fake = loss_fn(d_output_fake , labels_fake_d)
                    loss_g_fake = loss_fn(d_output_fake , labels_fake_g)

                    loss_d_val += (loss_d_real.detach().cpu().numpy() + loss_d_fake.detach().cpu().numpy())
                    loss_g_val += (loss_g_fake.detach().cpu().numpy())

            g_scheduler.step(loss_g_val/n_valid)
            d_scheduler.step(loss_d_val/n_valid)


            self.d_losses_train.append(loss_d_train/n_train)
            self.g_losses_train.append(loss_g_train/n_train)

            self.d_losses_val.append(loss_d_val/n_valid)
            self.g_losses_val.append(loss_g_val/n_valid) 



    def drawsamples(self,N,get_tensor=False):

        random_latents = torch.randn(N, self.g.input_dim, 1, 1, device=self.device)
        with torch.no_grad():
            outputs = self.g(random_latents) 

        if(get_tensor): 
            return outputs
            
        else:
            return outputs.cpu().numpy()





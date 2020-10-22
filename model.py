import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain


class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        width = 64
        hidden_layers = [
            nn.Conv2d(width, width, 3, 1, 1),
            nn.ReLU()
        ]
        
        for _ in range(args.n_hidden - 1):
            hidden_layers.append(nn.Conv2d(width, width, 3, 1, 1))
            hidden_layers.append(nn.ReLU())
            
        output_layers = [
            nn.Conv2d(width, 2, 3, 1, 1),
            LambdaLayer(lambda x: torch.mean(x, (-1, -2)))
        ]
        
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Sequential(*output_layers)

    def forward(self, x):
        return self.output(self.hidden(x))


class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        
        self.fc1 = nn.Linear(784, 400)
        self.fc1a = nn.Linear(400, 400)
        self.fc1b = nn.Linear(400, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc3a = nn.Linear(400, 400)
        self.fc3b = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, 784)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc1a(h1))
        h3 = F.relu(self.fc1b(h2))
        return self.fc21(h3), self.fc22(h3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc3a(h3))
        h5 = F.relu(self.fc3b(h4))
        return self.fc4(h5)

    def e_params(self):
        params = self.fc1.parameters()
        for l in [self.fc1a, self.fc1b, self.fc21, self.fc22]:
            params = chain(params, l.parameters())
        return params

    def d_params(self):
        params = self.fc3.parameters()
        for l in [self.fc3a, self.fc3b, self.fc4]:
            params = chain(params, l.parameters())
        return params
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class MLP(nn.Module):
    def __init__(self, n_hidden: int, input_dim: int, hidden_dim: int, output_dim: int):
        super(MLP, self).__init__()

        if n_hidden > 0:
            modules = []
            modules.append(nn.Linear(input_dim, hidden_dim))
            for idx in range(n_hidden - 1):
                if idx < n_hidden:
                    modules.append(nn.ReLU())
                modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.ReLU())
            modules.append(nn.Linear(hidden_dim, output_dim))
        else:
            modules = [nn.Linear(input_dim, output_dim)]

        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        y = self.seq(x)
        if y.shape[-1] == 1:
            y = y[:,0]
        return y

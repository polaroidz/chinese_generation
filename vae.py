import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torch.autograd import Variable
from torch import nn
from skimage import color
import numpy as np
import utils

np.random.seed(0)

latent_space_size = 144

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        self.max1 = nn.MaxPool2d(3)
    
        self.conv2 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        
        self.max2 = nn.MaxPool2d(3)
        
        self.conv3 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(1024)

        self.avg1 = nn.AvgPool2d(3)
        
        self.fc_mu = nn.Linear(1024, latent_space_size)
        self.fc_sig = nn.Linear(1024, latent_space_size)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.max1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.max2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.avg1(x)

        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        
        sigma = self.fc_sig(x)
        sigma = F.softplus(sigma) + 1e-6

        return mu, sigma

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(latent_space_size, 32 * 32)
        self.fc2 = nn.Linear(32 * 32, 48 * 48)
        
    def forward(self, z):
        z = self.fc1(z)
        z = F.relu(z)
        
        z = self.fc2(z)

        return z

class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def sample_latent(self, mu, sigma):
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        return mu + sigma * Variable(std_z, requires_grad=False)

    def decode(self, mu, sigma):
        z = self.sample_latent(mu, sigma)

        return self.decoder(z)

    def forward(self, x):
        mu, sigma = self.encoder(x)

        self.mu = mu
        self.sigma = sigma

        z = self.sample_latent(mu, sigma)

        xt = self.decoder(z)

        xt = xt.view(xt.size(0), 1, 48, 48)

        return xt

def latent_loss(mu, sigma):
    mu_sq = mu ** 2
    sig_sq = sigma ** 2

    return 0.5 * torch.mean(mu_sq + sig_sq - torch.log(sig_sq) - 1)

if __name__ == '__main__':
    batch_size = 12
    nb_batches = 600
    epochs = 100

    model = VariationalAutoEncoder()

    adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    gen = utils.batch_generator(batch_size, nb_batches)

    for epoch in range(epochs):
        for batch in range(nb_batches):
            X, _  = next(gen)
            Xt = model(X)

            loss = mse(Xt, X) #+ latent_loss(model.mu, model.sigma)

            log = " ".join(["Epoch:", str(epoch),
                            "Batch:", str(batch),
                            "Loss:",  str(loss.data[0])])

            print(log)

            adam.zero_grad()
            loss.backward()
            adam.step()
    
        torch.save(model.state_dict(), 'weights/epoch.{}.th'.format(epoch))

    torch.save(model.state_dict(), 'weights/epoch.final.th'.format(epoch))


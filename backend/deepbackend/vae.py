import torch
from torch import nn
import torch.nn.functional as F


class VariationalEncoder(nn.Module):
    def __init__(self, in_channels, latent_dims, device):  
        super(VariationalEncoder, self).__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 8*in_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(8*in_channels),
            nn.ReLU(True),
            nn.Conv2d(8*in_channels, 16*in_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(16*in_channels,),
            nn.ReLU(True),
            nn.Conv2d(16*in_channels, 32*in_channels, 3, stride=2, padding=0)
        )
        self.linear1 = nn.Linear(32*32*32*in_channels, 128)
        self.fc_mu = nn.Linear(128, latent_dims)
        self.fc_sig = nn.Linear(128, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0
        
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = self.encoder_conv(x)
        self.conv_shape = x.shape[2:]
        x = F.adaptive_avg_pool2d(x, 32)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.fc_mu(x)
        sigma = torch.exp(self.fc_sig(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl=0.5 * torch.sum(-1 - sigma + torch.exp(sigma) + mu**2)
        return z    

class Decoder(nn.Module):
    
    def __init__(self, in_channels, latent_dims, device):
        super().__init__()
        self.device = device

        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dims, 128),
            nn.ReLU(True),
            nn.Linear(128, 32*32*32*in_channels),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32*in_channels, 32, 32))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32*in_channels, 16*in_channels, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16*in_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(16*in_channels, 8*in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8*in_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(8*in_channels, in_channels, 3, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x, conv_shape, image_shape):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = F.adaptive_avg_pool2d(x, conv_shape)
        x = self.decoder_conv(x)
        x = F.adaptive_avg_pool2d(x, image_shape)
        x = torch.sigmoid(x)
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, in_channels, latent_dims, device):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(in_channels, latent_dims, device)
        self.decoder = Decoder(in_channels, latent_dims, device)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        z = self.encoder(x)
        return self.decoder(z, self.encoder.conv_shape, x.shape[2:])

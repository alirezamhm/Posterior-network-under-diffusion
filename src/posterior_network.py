import numpy as np
import torch
import lightning as L
from torch import nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
import torch.distributions as tdist

from src.dataset import corruptions
from src.architectures import architectures
from src.normalizing_flow import NormalizingFlow

class PosteriorNetwork(L.LightningModule):
    def __init__(self, args, res, num_classes, class_counts):
        super().__init__()
        self.args = args
        self.res = res
        self.num_classes = num_classes
        self.net = architectures[args.arch](num_classes=args.latent_dim)
        self.corruptions = corruptions
        self.flow = nn.ModuleList([NormalizingFlow(dim=args.latent_dim, flow_length=args.flow_length, flow_type=args.flow_type) for _ in range(num_classes)])
        self.batch_norm = nn.BatchNorm1d(num_features=args.latent_dim)
        self.register_buffer("class_counts", class_counts) # Put tensor on the same device as the model
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epoch)
        return [optimizer], [scheduler]

    def net_forward(self, x):
        return self.batch_norm(self.net(x))

    def forward(self, x):
        z = self.net_forward(x) 
        log_p = torch.stack([self.flow[c].log_prob(z) for c in range(self.num_classes)], dim=1)
        alpha = 1. + self.class_counts * torch.exp(log_p)
        return alpha
    
    def noisy_forward(self, x_noisy):
        flow_length, batch_size, n_noise_sample, c, h, w = x_noisy.shape
        x_noisy = x_noisy.view(flow_length * batch_size * n_noise_sample, c, h, w) # reshape to input to the network
        z_noisy = self.net_forward(x_noisy).view(flow_length, batch_size, n_noise_sample, -1)
        return z_noisy
    
    def statistics(self, alpha, y):
        y_hot = F.one_hot(y, num_classes=self.num_classes).float()
        loss = self.uce_loss(alpha, y_hot)
        probs = F.normalize(alpha, p=1)
        error = (probs.argmax(dim=1) != y).float().mean()
        return loss, error
    
    def uce_loss(self, alpha, y):
        alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, self.num_classes)
        entropy = Dirichlet(alpha).entropy().mean()
        return torch.mean(y*(torch.digamma(alpha_0) - torch.digamma(alpha))) - self.args.regr * entropy

    def gaussian_log_prob(self, z):
        mean = torch.mean(z, dim=0)
        cov = torch.cov(z.T)
        return tdist.MultivariateNormal(mean, cov).log_prob(z)

    def kl_loss(self, z_noisy, y):
        loss = 0
        for cls in range(self.num_classes):
            mask = (y == cls)
            if not mask.any():
                continue
            z_cls = z_noisy[:, mask, :, :].view(self.args.flow_length, -1, self.args.latent_dim) # (flow_length, class_samples, latent_dim) 
            for i in range(self.args.flow_length):
                log_q_z = self.gaussian_log_prob(z_cls[i])
                log_p_z = self.flow[cls].log_prob(z_cls[i], start=i+1)
                loss += (torch.exp(log_q_z)*(log_q_z - log_p_z)).mean()
        return loss/(self.num_classes*self.args.flow_length)

    def calc_noise_levels(self, function='linear', min=0.01, max=1):
        if function == 'linear':
            return torch.linspace(min, max, self.args.flow_length)
        elif function == 'log':
            return torch.logspace(np.log10(min), np.log10(max), self.args.flow_length)
        elif function == 'sqrt':
            return torch.linspace(min**2, max**2, self.args.flow_length)**0.5
        else:
            raise ValueError(f'Unknown function {function}')

    def add_noise(self, x):
        n_noise_sample = 5 # number of noisy samples per image
        # repeat the batch to add noise for each flow
        x = x.unsqueeze(0).unsqueeze(2).repeat(self.args.flow_length, 1, n_noise_sample, 1, 1, 1) # (flow_length, batch, n_noise_sample, c, h, w)
        levels = self.calc_noise_levels(function='linear')
        x += torch.randn_like(x) * levels.view(-1, 1, 1, 1, 1, 1)
        return x
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        alpha = self.forward(x) 
        loss, error = self.statistics(alpha, y)
        if self.args.type == 'posterior-network-diffusion':
            x_noisy = self.add_noise(x)
            z_noisy = self.noisy_forward(x_noisy)
            loss += self.args.kl_reg * self.kl_loss(z_noisy, y)
        self.log('train_loss', loss)
        self.log('train_error', error)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        alpha = self.forward(x)
        loss, error = self.statistics(alpha, y)

        # clean
        if dataloader_idx == 0:
            self.log('val_loss', loss, add_dataloader_idx=False, sync_dist=True)
            self.log('val_error', error, add_dataloader_idx=False, sync_dist=True)

        # corrs 
        if dataloader_idx > 0:
            intensity = 1 + batch_idx // int(50000/(5*self.args.val_batchsize)) # separate the intensity in each corrupt loader
            self.log(f'loss/{self.corruptions[dataloader_idx-1]}_{intensity}', loss, add_dataloader_idx=False, sync_dist=True)
            self.log(f'error/{self.corruptions[dataloader_idx-1]}_{intensity}', error, add_dataloader_idx=False, sync_dist=True)
            
    def process_batch_with_class_flow(self, z, y):
        results = torch.zeros((self.args.flow_length, *z.shape))
        for class_value in range(self.num_classes):
            mask = (y == class_value)
            if mask.any():
                flow = self.flow[class_value]
                results[:, mask] = flow.apply_transfer_layers(z[mask])
        return results
    
    
    
        
import torch
import lightning as L
from torch import nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet

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
        self.register_buffer("class_counts", class_counts) # Puts tensor on the same device as the model

        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
        return optimizer

    def forward(self, x):
        z = self.net(x)
        z = self.batch_norm(z)
        log_p = torch.stack([self.flow[c].log_prob(z) for c in range(self.num_classes)], dim=1)
        alpha = 1. + self.class_counts * torch.exp(log_p)
        return alpha
    
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

    def training_step(self, batch, batch_idx):
        x, y = batch
        alpha = self.forward(x)
        loss, error = self.statistics(alpha, y)
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
            intensity = 1 + batch_idx // 100   # assumes batch_size 100
            self.log(f'loss/{self.corruptions[dataloader_idx-1]}_{intensity}', loss, add_dataloader_idx=False, sync_dist=True)
            self.log(f'error/{self.corruptions[dataloader_idx-1]}_{intensity}', error, add_dataloader_idx=False, sync_dist=True)
    
    
    
        
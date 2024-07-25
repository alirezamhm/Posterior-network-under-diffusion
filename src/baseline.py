import torch
import lightning as L
import torch.nn.functional as F

from src.dataset import corruptions
from src.architectures import architectures

class Baseline(L.LightningModule):
    def __init__(self, args, res, num_classes):
        super().__init__()
        self.args = args
        self.res = res
        self.num_classes = num_classes
        self.net = architectures[args.arch](num_classes=num_classes)
        self.corruptions = corruptions
        
    def forward(self, x):
        self.net(x)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer
    
    def statistics(self, logits, y):
        loss = F.cross_entropy(logits, y)
        error = (logits.argmax(dim=1) != y).float().mean()
        return loss, error
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss, error = self.statistics(logits, y)
        self.log('train_loss', loss)
        self.log('train_error', error)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        logits = self.forward(x)
        loss, error = self.statistics(logits, y)

        # clean
        if dataloader_idx == 0:
            self.log('val_loss', loss, add_dataloader_idx=False, sync_dist=True)
            self.log('val_error', error, add_dataloader_idx=False, sync_dist=True)

        # corrs 
        if dataloader_idx > 0:
            intensity = 1 + batch_idx // 100   # assumes batch_size 100
            self.log(f'loss/{self.corruptions[dataloader_idx-1]}_{intensity}', loss, add_dataloader_idx=False, sync_dist=True)
            self.log(f'error/{self.corruptions[dataloader_idx-1]}_{intensity}', error, add_dataloader_idx=False, sync_dist=True)
    
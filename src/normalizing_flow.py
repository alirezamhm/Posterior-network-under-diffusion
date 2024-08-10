import torch
from torch import nn
import torch.distributions as tdist
from pyro.distributions.transforms.radial import Radial
from pyro.distributions.transforms.affine_autoregressive import affine_autoregressive
from functools import partial

flow_types = {
    'radial': Radial,
    'iaf128': partial(affine_autoregressive, hidden_dims=[128, 128])
}

class NormalizingFlow(nn.Module):
    def __init__(self, dim, flow_length, flow_type='radial'):
        super(NormalizingFlow, self).__init__()
        self.dim = dim
        self.flow_length = flow_length

        self.mean = nn.Parameter(torch.zeros(self.dim), requires_grad=False)
        self.cov = nn.Parameter(torch.eye(self.dim), requires_grad=False)
        
        self.transforms = nn.Sequential(*(flow_types[flow_type](dim) for _ in range(flow_length)))

    def forward(self, z, start=0):
        sum_log_jacobians = 0
        for transform in self.transforms[start:]:
            z_next = transform(z)
            sum_log_jacobians +=  transform.log_abs_det_jacobian(z, z_next)
            z = z_next
        return z, sum_log_jacobians

    def log_prob(self, x, start=0):
        if start == len(self.transforms):
            z, sum_log_jacobians = x, 0
        else:
            z, sum_log_jacobians = self.forward(x, start=start)
        log_prob_z = tdist.MultivariateNormal(self.mean, self.cov).log_prob(z)
        return log_prob_z + sum_log_jacobians
    
    def apply_transfer_layers(self, z):
        layers = []
        for transform in self.transforms:
            z_next = transform(z)
            layers.append(z_next)
            z = z_next
        return torch.stack(layers, dim=0)
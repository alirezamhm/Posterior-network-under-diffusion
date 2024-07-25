import torch
from pyro.distributions.transforms.radial import Radial
from torch import nn
import torch.distributions as tdist


class NormalizingFlow(nn.Module):
    def __init__(self, dim, flow_length):
        super(NormalizingFlow, self).__init__()
        self.dim = dim
        self.flow_length = flow_length

        self.mean = nn.Parameter(torch.zeros(self.dim), requires_grad=False)
        self.cov = nn.Parameter(torch.eye(self.dim), requires_grad=False)
        
        self.transforms = nn.Sequential(*(Radial(dim) for _ in range(flow_length)))

    def forward(self, z):
        sum_log_jacobians = 0
        for transform in self.transforms:
            z_next = transform(z)
            sum_log_jacobians +=  transform.log_abs_det_jacobian(z, z_next)
            z = z_next
        return z, sum_log_jacobians

    def log_prob(self, x):
        z, sum_log_jacobians = self.forward(x)
        log_prob_z = tdist.MultivariateNormal(self.mean, self.cov).log_prob(z)
        return log_prob_z + sum_log_jacobians

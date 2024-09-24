import glob
import numpy as np
import os
import torch
import argparse

from src.dataset import *
from src.utils import load_model

args_dict = {'arch': 'resnet18',
             'type': 'posterior-network',
             'mb': 128,
             'cpus': 4,
             'latent_dim': 4,
             'flow_length': 4,
             'flow_type': 'radial',
             'kl_reg': 0,
             'bn_track_disable': True,
             'bn_remove': False,
             'noise_function': 'linear',
             }
args = argparse.Namespace(**args_dict)

loader_tr, loader_val, res, num_classes, class_counts = load_data('cifar10', mb=args.mb, scale='normal', cpus=args.cpus)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

fn = "j8p8tdvz"
path = glob.glob(f"posterior-network-under-diffusion/{fn}/checkpoints/*.ckpt")
if not path:
    raise FileNotFoundError(f"No files found matching run id: {fn}")
model = load_model(args, path[0], res, num_classes, class_counts, device=device)
model.eval()

zs = []
qs = [[] for _ in range(model.args.flow_length+1)]
ps = [[] for _ in range(model.args.flow_length+1)]
with torch.no_grad():
    for i, (inputs, targets) in enumerate(loader_tr):
        inputs, targets = inputs.to(device), targets.to(device)
        print(f"Batch {i+1}/{len(loader_tr)}")
        inputs_noisy = model.add_noise(inputs)
        z_clean = model.net_forward(inputs)
        z_noisy = model.noisy_forward(inputs_noisy)
        z = torch.cat([z_clean.reshape(1, -1, 1, model.args.latent_dim).expand((1, -1, 5, -1)), z_noisy])
        zs.append(z)
        
        for cls in range(num_classes):
            mask = (targets == cls)
            if not mask.any():
                continue
            z_cls = z[:, mask, :, :].view(model.args.flow_length+1, -1, model.args.latent_dim)
            for i in range(model.args.flow_length+1):
                qs[i].append(torch.exp(model.gaussian_log_prob(z_cls[i])))
                ps[i].append(torch.exp(model.flow[cls].log_prob(z_cls[i], start=i)))
        
zs = torch.cat(zs, dim=1)
qs = torch.stack([torch.cat(q) for q in qs])
ps = torch.stack([torch.cat(p) for p in ps])

output_dir = f"samples/{fn}"
os.makedirs(output_dir, exist_ok=True)

torch.save(zs, os.path.join(output_dir, 'zs.pt'))
torch.save(qs, os.path.join(output_dir, 'qs.pt'))
torch.save(ps, os.path.join(output_dir, 'ps.pt'))
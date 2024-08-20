import numpy as np
import torch
import argparse
import lightning as L
import glob

from src.architectures import architectures
from src.dataset import load_data, data_dirs
from src.baseline import Baseline
from src.posterior_network import PosteriorNetwork, noise_functions
from src.normalizing_flow import flow_types

AUGMENTATIONS = ('','f','c','r','fcr')


parser = argparse.ArgumentParser(description='Posterior network under diffusion')
parser.add_argument('-d', '--data', choices=data_dirs.keys(), default='cifar10')
parser.add_argument('--arch', choices=architectures.keys(), default='resnet18')
parser.add_argument('-t', '--type', choices=['baseline', 'posterior-network', 'posterior-network-diffusion'], default='baseline')
parser.add_argument('-b', '--batchsize', type=int, default=256)
parser.add_argument('-vb', '--val-batchsize', type=int, choices=[100, 200, 500, 1000, 2000], default=100)
parser.add_argument('-e', '--epoch', type=int, default=200)
parser.add_argument('-ld', '--latent-dim', type=int, default=6, help='Latent dimension')
parser.add_argument('-fl', '--flow-length', type=int, default=6, help='Number of flow layers')
parser.add_argument('-ft', '--flow-type', choices=flow_types.keys(), default='radial', help='Type of the normalizing flow')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('-wd', '--weight-decay', type=float, default=0)
parser.add_argument('--regr', type=float, default=1e-5, help='Regularization factor in Bayesian loss')
parser.add_argument('--kl-reg', type=float, default=0, help='Regularization factor in KL divergence')
parser.add_argument('--noise-function', choices=noise_functions.keys(), default='linear', help='Noise function')
parser.add_argument('-fn', type=str, default=None, help='Run id for loading posterior-network-diffusion')
parser.add_argument('--scaling', choices=['normal','uniform'], default='normal', help='Data normalization')
parser.add_argument('-a', '--aug', choices=AUGMENTATIONS, nargs="+", default='', help='Augmentations')
parser.add_argument('--cpus', type=int, default=1)
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('--valfreq', type=int, default=1000, help='Validation frequency over epochs')
args = parser.parse_args()

def args2str(args):
    s = f'{args.data}-{args.arch}-{args.type}'
    if args.fn:
        s += f'-pretrained'
    s += f'-E({args.epoch})' 
    s += f'-S({args.seed})'
    s += f'-L({args.lr:.0e})'
    if args.aug:
        s += f'-A({",".join(args.aug)})'
    if args.type != 'baseline':
        s += f'-LD({args.latent_dim})-FL({args.flow_length})-FT({args.flow_type})'
    if args.type == 'posterior-network-diffusion':
        s += f'-NF({args.noise_function})'
        s += f'-KL({args.kl_reg:.0e})'
    return s


if __name__=='__main__':
    L.pytorch.seed_everything(args.seed, workers=True)

    wandb_logger = L.pytorch.loggers.WandbLogger(project='posterior-network-under-diffusion', config=vars(args), name=args2str(args))

    print('args:', vars(args))
    print('name:', args2str(args))
    if torch.cuda.is_available():
        print(f'cuda {torch.cuda.device_count()}x {torch.cuda.get_device_name(0)}')
    else:
        print('cpu')
    
    # lightning trainer parameter
    trainer_args = {
    	'max_epochs': args.epoch,
		'log_every_n_steps': 50,
		'check_val_every_n_epoch': args.valfreq,
		'num_sanity_val_steps': 1,
		'logger': wandb_logger,
		'enable_progress_bar': False,
		'benchmark': False,
        'deterministic': True,
	}
    
    # load data
    loaders_tr, loaders_val, res, num_classes, class_counts = load_data(args.data, args.batchsize, args.scaling, args.aug, args.cpus, args.val_batchsize)
    
    # create model
    if args.type == 'baseline':
        model = Baseline(args, res, num_classes)
    else:
        model = PosteriorNetwork(args, res, num_classes, class_counts)
        
    if args.fn:
        fn = glob.glob(f"posterior-network-under-diffusion/{args.fn}/checkpoints/*.ckpt")
        if not fn:
            raise FileNotFoundError(f"No files found matching run id: {args.fn}")
        model.load_state_dict(torch.load(fn[0])['state_dict'])
        
    # train model
    trainer = L.Trainer(**trainer_args)
    trainer.fit(model, loaders_tr, loaders_val)
    trainer.validate(model, loaders_val)

    
import numpy as np
import torch
import argparse
import lightning as L

from src.architectures import architectures
from src.dataset import load_data
from src.baseline import Baseline

AUGMENTATIONS = ('','f','c','r','fcr')


parser = argparse.ArgumentParser(description='Posterior network under diffusion')
parser.add_argument('-d', '--data', choices=['cifar10'], default='cifar10')
parser.add_argument('--arch', choices=architectures.keys(), default='resnet18')
parser.add_argument('-t', '--type', choices=['baseline', 'posterior-network'], default='baseline')
parser.add_argument('-b', '--batchsize', type=int, default=128)
parser.add_argument('-e', '--epoch', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--scaling', choices=['normal','uniform'], default='normal', help='Data normalization')
parser.add_argument('-a', '--aug', choices=AUGMENTATIONS, nargs="+", default='', help='Augmentations')
parser.add_argument('--cpus', type=int, default=1)
parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('--valfreq', type=int, default=1000, help='Validation frequency over epochs')
args = parser.parse_args()

def args2str(args):
	s = f'{args.data}-{args.arch}-{args.type}'
	s += f'-E({args.epoch})' 
	if args.aug:
		s += f'-A({','.join(args.aug)})'
	return s


if __name__=='__main__':
    L.pytorch.seed_everything(args.seed)

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
 		'precision': 'bf16-mixed' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else '16-mixed',
		'benchmark': True
	}
    
    # load data
    loaders_tr, loaders_val, res, num_classes, class_counts = load_data(args.data, args.batchsize, args.scaling, args.aug, args.cpus)
    
    # create model
    if args.type == 'baseline':
        model = Baseline(args, res, num_classes)
    elif args.type == 'posterior-network':
        pass
    
    # train model
    trainer = L.Trainer(**trainer_args)
    trainer.fit(model, loaders_tr, loaders_val)
    trainer.validate(model, loaders_val, verbose=True)

    
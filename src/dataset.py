import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms.v2 import *  # v2 transforms


def linearscale(x):
	return (x-0.5)*2

corruptions = (
	'brightness','contrast','defocus_blur','elastic_transform','fog',
	'frost','gaussian_noise','glass_blur','impulse_noise','jpeg_compression',
    'motion_blur','pixelate','shot_noise','snow','zoom_blur',
)

normalizers = {
	'cifar10':  Normalize([0.491, 0.482, 0.446], [0.247, 0.243, 0.261]),
	'tin':      Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}

augmentations = {
	'f': RandomHorizontalFlip(),
	'c': RandomCrop(32, 4),
	'r': RandomRotation(15),
	'fcr': Compose([RandomHorizontalFlip(),RandomCrop(32, 4),RandomRotation(15)]),
	'fcr64': Compose([RandomHorizontalFlip(),RandomCrop(64, 4),RandomRotation(15)]),
}

datashapes = {
	'cifar10':  [(3, 32, 32), 10],
	'tin':      [(3, 64, 64), 200],
}

data_dirs = {
	'cifar10': './data/CIFAR-10/',
	'tin': './data/TIN/',
}

def make_transform(data, scale, aug=[], res=None):
	transforms =  [ ToImage() ]
	transforms += [ augmentations[a] for a in aug ]
	transforms += [ ToDtype(torch.float32, scale=True) ] # transforms [0,255] to [0,1]
	if scale == 'normal':
		transforms += [ normalizers[data] ]
	elif scale == 'uniform':
		transforms += [ Lambda(linearscale) ]
	return Compose(transforms)

def load_data(data, mb=128, scale=None, aug=[], cpus=1):
	res, classes = datashapes[data]
	data_dir = data_dirs[data]

	if data == 'tin':
		aug = [a if a != 'fcr' else 'fcr64' for a in aug if a != 'c'] # remove 'c' and replace 'fcr' with 'fcr64'
	
	data_tr =  DatasetC(data_dir, 'train', make_transform(data, scale, aug, res))
	data_val = DatasetC(data_dir, 'val',   make_transform(data, scale))
	data_c =  [DatasetC(data_dir, c,       make_transform(data, scale)) for c in corruptions]

	loader_tr   =  DataLoader(data_tr,  batch_size=mb,  shuffle=True,  num_workers=cpus, pin_memory=True, persistent_workers=True)	
	loader_val  = [DataLoader(data_val, batch_size=100, shuffle=False, num_workers=cpus, pin_memory=True, persistent_workers=True)]
	loader_val += [DataLoader(data,     batch_size=100, shuffle=False, num_workers=cpus, pin_memory=True, persistent_workers=True) for data in data_c]

	return loader_tr, loader_val, res, classes

class DatasetC(datasets.VisionDataset):
	def __init__(self, root, fold, transform=None):
		super().__init__(root, transform=transform)

		if fold in ('train','val'):
			data_path = os.path.join(root, '%s.npy' % fold)
			target_path = os.path.join(root, '%s_labels.npy' % fold)
		else:
			data_path = os.path.join(root, '%s.npy' % fold)
			target_path = os.path.join(root, 'corr_labels.npy')

		self.data = np.load(data_path)
		self.targets = np.load(target_path)

	def __getitem__(self, index):
		img, targets = self.data[index], self.targets[index]
		if self.transform is not None:
			img = self.transform(img)
		targets = torch.tensor(targets, dtype=torch.long)
		return img, targets
	
	def __len__(self):
		return len(self.data)


import torch
from src.posterior_network import PosteriorNetwork
from src.baseline import Baseline


def load_model(args, path, res, num_classes, class_counts, device=torch.device('cpu')):
    if args.type == 'baseline':
        model = Baseline(args, res, num_classes)
    else:
        model = PosteriorNetwork(args, res, num_classes, class_counts)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device)
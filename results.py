import numpy as np
import torch
import pandas as pd
import wandb
import argparse

from src.dataset import corruptions
from src.posterior_network import PosteriorNetwork
from src.baseline import Baseline


def extract_wandb():
    runs = wandb.Api().runs('alirezamhm/posterior-network-under-diffusion')
    metrics, configs, names = [], [], []
    for run in runs: 
        if run.state == 'finished':
            metrics.append(run.summary._json_dict)
            configs.append({k: v for k,v in run.config.items() if not k.startswith('_')})
            names.append(run.name)
    df = pd.DataFrame({"log": metrics, "config": configs, "name": names})
    df.to_csv('results.csv')

    return metrics, configs, names

def print_summary(metrics, names, limit=None):
    for i in range(limit if limit else len(names)):
        id = names[i]
        try:
            loss = metrics[i]['val_loss']
            error = metrics[i]['val_error']
            
            corr_loss = np.mean([metrics[i][f'loss/{c}_{k+1}'] for c in corruptions for k in range(5)])
            corr_error = np.mean([metrics[i][f'error/{c}_{k+1}'] for c in corruptions for k in range(5)])

            print(f'{i+1:03d} | {id}')
            print(f'loss {loss:.4f} error {error:.4f}  corr_loss {corr_loss:.4f} corr_error {corr_error:.4f}')        
        except:
            pass        


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Posterior network under diffusion')
    parser.add_argument('-l', '--limit', type=int, default=None)
    args = parser.parse_args()
    metrics, configs, names = extract_wandb()
    print_summary(metrics, names, args.limit)
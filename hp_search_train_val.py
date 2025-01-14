import os, random, argparse
import itertools
import numpy as np
import torch
from train import train # Training script
from alpha_eval import eval
from algorithms import algorithms_dict # Dictionary of methods available
from utils.network import net_dict # Dictionary of pre-trained models available
from utils.hp_functions import * # Helper functions to load default hyper-parameters
from utils.misc import *

parser = argparse.ArgumentParser(description='Partial Domain Adaptation')
parser.add_argument('--data_folder', type=str,
                    default='datasets',
                    help="Path to datatasets")
parser.add_argument('--logs_folder', type=str,
                    default='logs_hp_search',
                    help="Path to folder to save the logs")
parser.add_argument('--dset', type=str, 
                    default='office-home',
                    help="Choice of dataset",
                    choices=['office-home', 'visda', 'domainnet'])
parser.add_argument('--method', type=str,
                    default='ar',
                    help="Choice of partial domain adaptation method",
                    choices=algorithms_dict.keys())
parser.add_argument('--net', type=str, 
                    default='ResNet50',
                    help="Choice of neural network architecture",
                    choices=net_dict.keys())
parser.add_argument('--source_domain', type=str, 
                    default='Art',
                    help="Choice of source domain")
parser.add_argument('--target_domain', type=str, 
                    default='Clipart',
                    help="Choice of source domain")
parser.add_argument('--seed', type=int, 
                    default=2020,
                    help="Choice of seed")
parser.add_argument('--sweep_idx', type=int, 
                    default=0,
                    help="Index of hparam sweep")
parser.add_argument('--mode', type=str, 
                    default='train',
                    help="Flag to run training or evaluation",
                    choices=['train', 'eval'])

args = parser.parse_args()
# Adding SCRATCH dir to data and logs folder
scratch_dir = os.path.join(os.environ['SCRATCH'],'DomainAdaptation')
args.data_folder = os.path.join(scratch_dir, args.data_folder)
args.logs_folder = os.path.join(scratch_dir, args.logs_folder)

dset_hp, domains = get_dset_hp(args.dset, args.data_folder)
dset_hp['use_val'] = True

net_hp = get_net_hp_default(dset_hp, args.net)

logger_hp = get_logger_hp_search(args.dset)


if args.source_domain in domains:
    dset_hp['source_domain'] = args.source_domain
else:
    raise ValueError('Not an available domain for the dataset chosen.')
if args.target_domain in domains:
    dset_hp['target_domain'] = args.target_domain
else:
    raise ValueError('Not an available domain for the dataset chosen.')
if dset_hp['source_domain'] == dset_hp['target_domain']:
    raise ValueError('Source and target domains should be different.')

loss_hp = get_loss_hp_default(args.method, dset_hp)
dset_hp = dset_hp_update_paths_task(dset_hp, logger_hp)
search_space = get_search_space(args.method)

train_hp = get_train_hp_search(args.method, dset_hp)
train_hp['seed'] = args.seed
if args.mode == 'train':
    for hidx, hp_params in enumerate(itertools.product(*[iter(search_space[key]) for key in search_space.keys()])):
        if hidx != args.sweep_idx: continue # running one hparam at a time :)
        if args.method == 'ar':
            if hp_params[1] != -hp_params[2]: # restricts up == -low
                continue
        for ix, key in enumerate(search_space.keys()):
            loss_hp[key] = hp_params[ix]

        # Set randoms seed for reproducibility
        set_seeds(train_hp['seed'])

        # Find output_dir
        output_dir = os.path.join(args.logs_folder, args.method, net_hp['net'], dset_hp['name'], dset_hp['task'])     
        for ix, key in enumerate(search_space.keys()):
            output_dir = os.path.join(output_dir, f'{key}_{hp_params[ix]}')
        logger_hp['output_dir'] = os.path.join(output_dir, f"seed_{train_hp['seed']}", 'run_0')
        
        # Train with specified hyper-parameters
        if not os.path.exists(logger_hp['output_dir']):
            print(f"Running hparam config {args.sweep_idx}")
            os.makedirs(logger_hp['output_dir'], exist_ok=True)
            algorithm = algorithms_dict[args.method](dset_hp, loss_hp, train_hp, net_hp, logger_hp)
            train(algorithm)
        else:
            print(f"Skipping hparam config {args.sweep_idx} because dir already exists")

elif args.mode == 'eval':
    tot_configs = len(list(itertools.product(*[iter(search_space[key]) for key in search_space.keys()])))
    for hidx, hp_params in enumerate(itertools.product(*[iter(search_space[key]) for key in search_space.keys()])):
        if hidx != args.sweep_idx: continue # running one hparam at a time :)
        print(f"Running eval for hparam config {1+hidx}/{tot_configs}, for method {args.method}")
        if args.method == 'ar':
            if hp_params[1] != -hp_params[2]: # restricts up == -low
                continue
        for ix, key in enumerate(search_space.keys()):
            loss_hp[key] = hp_params[ix]

        # Set randoms seed for reproducibility
        set_seeds(train_hp['seed'])

        # Find output_dir
        output_dir = os.path.join(args.logs_folder, args.method, net_hp['net'], dset_hp['name'], dset_hp['task'])     
        for ix, key in enumerate(search_space.keys()):
            output_dir = os.path.join(output_dir, f'{key}_{hp_params[ix]}')
        logger_hp['output_dir'] = os.path.join(output_dir, f"seed_{train_hp['seed']}", 'run_0')

        algorithm = algorithms_dict[args.method](dset_hp, loss_hp, train_hp, net_hp, logger_hp)
        algorithm.logger_hp['model_selection_metrics'] += ['alpha']
        eval(algorithm)
        print(f"Eval results saved to {algorithm.logger_hp['output_dir']}")

else:
    raise NotImplementedError
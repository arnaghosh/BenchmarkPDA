import os, glob, argparse
import itertools
import numpy as np
from tqdm import tqdm
from utils.hp_functions import *
from algorithms import algorithms_dict # Dictionary of methods available
from utils.network import net_dict # Dictionary of pre-trained models available
import now

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

res_dict = {}
methods = ['ar', 'pada', 'ba3us', 'jumbot', 'mpot']

for method in tqdm(methods):
    loss_hp = get_loss_hp_default(method, dset_hp)
    dset_hp = dset_hp_update_paths_task(dset_hp, logger_hp)

    search_space = get_search_space(method)
    train_hp = get_train_hp_search(method, dset_hp)
    train_hp['seed'] = args.seed

    for hidx, hp_params in enumerate(itertools.product(*[iter(search_space[key]) for key in search_space.keys()])):
        if method == 'ar':
            if hp_params[1] != -hp_params[2]: # restricts up == -low
                continue
        for ix, key in enumerate(search_space.keys()):
            loss_hp[key] = hp_params[ix]
        output_dir = os.path.join(args.logs_folder, method, net_hp['net'], dset_hp['name'], dset_hp['task'])
        hparam_str = ''
        for ix, key in enumerate(search_space.keys()):
            output_dir = os.path.join(output_dir, f'{key}_{hp_params[ix]}')
            hparam_str += f'{key}_{hp_params[ix]}_'
        hparam_str = hparam_str[:-1]
        output_dir = os.path.join(output_dir, f"seed_{train_hp['seed']}", 'run_0')
        try:
            data = np.load(os.path.join(output_dir,'results_eval_final.npy'),
                        allow_pickle=True).item()
        except:
            continue
        if method not in res_dict.keys():
            res_dict[method] = {}
        res_dict[method][hparam_str] = {
            'source': {'acc': data['s_acc'][0], 
                       'feats_alpha': data['alpha'][0]['source_train']['features']['alpha'],
                       'prefeats_alpha': data['alpha'][0]['source_train']['prefeatures']['alpha'],
                       },
            'val': {'acc': data['s_acc'][0], 
                       'feats_alpha': data['alpha'][0]['source_val']['features']['alpha'],
                       'prefeats_alpha': data['alpha'][0]['source_val']['prefeatures']['alpha'],
                       },
            'target': {'acc': data['t_acc'][0], 
                       'feats_alpha': data['alpha'][0]['test']['features']['alpha'],
                       'prefeats_alpha': data['alpha'][0]['test']['prefeatures']['alpha'],
                       },
        }
        
breakpoint()
fname = f'result_collated_{now.strftime("%d-%m-%Y_%H-%M-%S")}.npy'
np.save(fname,res_dict,allow_pickle=True)
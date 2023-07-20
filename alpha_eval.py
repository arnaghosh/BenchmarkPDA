import os, random, argparse
import numpy as np
import torch
from tqdm import tqdm
from utils.logger import *
from utils.model_selection import *
import fastssl.utils.powerlaw as powerlaw

def get_alpha(feats):
    eigen = powerlaw.get_eigenspectrum(feats)
    alpha, ypred, r2, r2_100 = powerlaw.stringer_get_powerlaw(eigen, trange=np.arange(3, 50))
    return {'eigen': eigen,'alpha': alpha,'ypred': ypred, 'r2': r2, 'r2_100': r2_100}

def eval(algorithm):
    algorithm.logger_hp['filename'] = 'log_alpha_final'
    log_file = Logger(algorithm.logger_hp)
    if algorithm.logger_hp is not None:
        log_file.write(f'Logger Hyper-Parameters', time=False)
        for key in algorithm.logger_hp:
            log_file.write(f'    {key}: {algorithm.logger_hp[key]}', time=False)
    if algorithm.net_hp is not None:
        # Set net load path to final checkpoint saved
        algorithm.net_hp['load_net'] = True
        algorithm.net_hp['load_path'] = os.path.join(algorithm.logger_hp['output_dir'],
                                                 'model_final.pt')
        log_file.write(f'Net Hyper-Parameters', time=False)
        for key in algorithm.net_hp:
            log_file.write(f'    {key}: {algorithm.net_hp[key]}', time=False)
    if algorithm.train_hp is not None:
        log_file.write(f'Training Hyper-Parameters', time=False)
        for key in algorithm.train_hp:
            log_file.write(f'    {key}: {algorithm.train_hp[key]}', time=False)
    if algorithm.dset_hp is not None:
        log_file.write(f'Dataset Hyper-Parameters', time=False)
        for key in algorithm.dset_hp:
            log_file.write(f'    {key}: {algorithm.dset_hp[key]}', time=False)
    if algorithm.loss_hp is not None:
        log_file.write(f'Loss Hyper-Parameters', time=False)
        for key in algorithm.loss_hp:
            log_file.write(f'    {key}: {algorithm.loss_hp[key]}', time=False)
            
    algorithm.set_dsets()
    algorithm.set_dsets_model_selection()
    algorithm.set_dset_loaders()
    algorithm.set_dset_loaders_model_selection()
    algorithm.set_base_network()
    
    algorithm.prep_for_train()
    
    log_results = init_log_results(
        algorithm.train_hp['test_interval'], 
        algorithm.train_hp['max_iterations'], 
        algorithm.net_hp['class_num'],
        algorithm.logger_hp['save_outputs_evl'],
        algorithm.logger_hp['model_selection_metrics'])

    if algorithm.__class__.__name__ == 'SourceOnlyPlus':
        gamma_acc = []

    # correlate with source and target accuracy
    log_file.write("Started metrics eval on last checkpoint")
    if algorithm.dset_hp['use_val']:
        s_prefeatures, s_features, s_logits, s_labels = get_data_features(
                                        algorithm.dset_loaders['source_train'], 
                                        algorithm.base_network)
        v_prefeatures, v_features, v_logits, v_labels = get_data_features(
                                        algorithm.dset_loaders['source_val'], 
                                        algorithm.base_network)
    t_prefeatures, t_features, t_logits, t_labels = get_data_features(
                                        algorithm.dset_loaders['test'], 
                                        algorithm.base_network)

    algorithm.class_weight = get_class_weight(t_logits)

    current_values = {}
    
    for metric in algorithm.logger_hp['model_selection_metrics']:
        if metric == 't_acc':
            current_values[metric] = get_acc(t_logits, t_labels)
        elif metric == 'ent':
            current_values[metric] = get_mean_ent(t_logits)
        elif metric == 'snd':
            current_values[metric] = get_snd(t_logits)
        elif metric == 's_acc':
            current_values[metric] = get_acc(v_logits, v_labels)
        elif metric == 'dev_lr':
            weights = get_importance_weights_lr(s_features, t_features, v_features, algorithm.train_hp['seed'])
            error = get_error(v_logits, v_labels)
            current_values[metric] = get_dev_risk(weights, error)
        elif metric == 'dev_mlp':
            weights = get_importance_weights_mlp(s_features, t_features, v_features, algorithm.train_hp['seed'])
            error = get_error(v_logits, v_labels)
            current_values[metric] = get_dev_risk(weights, error)
        elif metric == 'dev_svm':
            weights = get_importance_weights_svm(s_features, t_features, v_features, algorithm.train_hp['seed'])
            error = get_error(v_logits, v_labels)
            current_values[metric] = get_dev_risk(weights, error)
        elif metric == '1shot_acc':
            _, temp_logits, temp_labels = get_data(algorithm.dset_loaders['test_1shot'], algorithm.base_network)
            current_values[metric] = get_acc(temp_logits, temp_labels)
        elif metric == '1shot_10crop_acc':
            current_values[metric] = get_acc_10crop(algorithm.dset_loaders['test_1shot_10crop'], algorithm.base_network)
        elif metric == '3shot_acc':
            _, temp_logits, temp_labels = get_data(algorithm.dset_loaders['test_3shot'], algorithm.base_network)
            current_values[metric] = get_acc(temp_logits, temp_labels)
        elif metric == '3shot_10crop_acc':
            current_values[metric] = get_acc_10crop(algorithm.dset_loaders['test_3shot_10crop'], algorithm.base_network)
        elif metric == '25random_acc':
            _, temp_logits, temp_labels = get_data(algorithm.dset_loaders['test_25random'], algorithm.base_network)
            current_values[metric] = get_acc(temp_logits, temp_labels)
        elif metric == '25random_10crop_acc':
            current_values[metric] = get_acc_10crop(algorithm.dset_loaders['test_25random_10crop'], algorithm.base_network)
        elif metric == '50random_acc':
            _, temp_logits, temp_labels = get_data(algorithm.dset_loaders['test_50random'], algorithm.base_network)
            current_values[metric] = get_acc(temp_logits, temp_labels)
        elif metric == '50random_10crop_acc':
            current_values[metric] = get_acc_10crop(algorithm.dset_loaders['test_50random_10crop'], algorithm.base_network)
        elif metric == '100random_acc':
            _, temp_logits, temp_labels = get_data(algorithm.dset_loaders['test_100random'], algorithm.base_network)
            current_values[metric] = get_acc(temp_logits, temp_labels)
        elif metric == '100random_10crop_acc':
            current_values[metric] = get_acc_10crop(algorithm.dset_loaders['test_100random_10crop'], algorithm.base_network)
        elif metric == 'alpha':
            # compute alpha for prefeatures and features for train, val and target sets here!
            current_values[metric] = {'source_train': {}, 'source_val': {}, 'test': {}}
            try:
                current_values[metric]['source_train']['prefeatures'] = get_alpha(
                    s_prefeatures
                )
                current_values[metric]['source_train']['features'] = get_alpha(
                    s_features
                )
            except:
                pass
            try:
                current_values[metric]['source_val']['prefeatures'] = get_alpha(
                    v_prefeatures
                )
                current_values[metric]['source_val']['features'] = get_alpha(
                    v_features
                )
            except:
                pass
            try:
                current_values[metric]['test']['prefeatures'] = get_alpha(
                    t_prefeatures
                )
                current_values[metric]['test']['features'] = get_alpha(
                    t_features
                )
            except:
                pass

        else:
            raise NotImplementedError
                
    log_temp = {'iterations': -1, 'class_weights': algorithm.class_weight.cpu()}

    for metric in current_values:
        log_temp[metric] = current_values[metric]

    # update log_results
    log_results = update_log_results(log_results, log_temp)

    update_log_file(log_results, log_file)
    
    np.save(os.path.join(algorithm.logger_hp['output_dir'], 
                         'results_eval_final.npy'), 
            log_results, allow_pickle=True)
    log_file.write('Finished Eval')
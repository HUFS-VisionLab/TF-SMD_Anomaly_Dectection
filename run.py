import os
import glob
import shutil
import argparse

import numpy as np
import tensorflow as tf
import torch

from plot import save_loss, compare_loss 
from model import Model

from datasets_loader import DatasetsLoader


def calc_roc(score, target_name, min_max):
    fpr_list = []
    tpr_list = []
    min_threshold, max_threshold = min_max
    interval = (max_threshold - min_threshold) / 20
    
    thres = min_threshold
    for i in range(21):
        true_negative = 0
        all_negative = 0
        for name, loss_list in score.items():
            if  name == target_name:
                if len(args.targets) == 1:
                    tpr_list.append(len([loss for loss in loss_list if loss < thres]) / len(loss_list))
            else:
                all_negative += len(loss_list)
                true_negative += len([loss for loss in loss_list if loss > thres])
        fpr_list.append(1-(true_negative/all_negative))
        
        thres += interval

    
    roc = np.array([fpr_list, tpr_list])
    return roc

    
        

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', type=int, default=3, help='-')
    parser.add_argument('--model_type', type=int, default=2, help='-')
    parser.add_argument('--no_bidirectional', action='store_true', default=False, help='-')
    parser.add_argument('--data_version', type=str, default='2019-2', help='-')
    parser.add_argument('--data_type', type=str, default='timesteps_64_mel_80', help='-')
    parser.add_argument('--augment', action='store_true', default=False, help='-')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='-')
    parser.add_argument('--beta_1', type=float, default=0.9, help='-')
    parser.add_argument('--beta_2', type=float, default=0.999, help='-')
    parser.add_argument('--epsilon', type=float, default=1e-08, help='-')
    parser.add_argument('--epochs', type=int, default=3000, help='-')
    parser.add_argument('--batch_size', type=int, default=None, help='-')
    parser.add_argument('-l', '--targets', nargs='*', default=None, help='-')
    parser.add_argument('--model_path', type=str, default='./model', help='-')
    parser.add_argument('--inference', action='store_true', default=False, help='-')
    parser.add_argument('--loss_log_scale', type=int, default=0, help='-')
    args, unknown = parser.parse_known_args()
    
    
    """ Preprare the path of dataset and load hyper parameter of model """
    datasets_loader = DatasetsLoader(targets=args.targets, data_type=args.data_type, augment=args.augment)
    trainPath_dict = datasets_loader.pathList_dict['train']
    testPath_dict = datasets_loader.pathList_dict['test']
    data_type = args.data_type.split('_')
    args.timesteps = int(data_type[1])
    args.inputs_dims = int(data_type[-1])
    
    modelType_dict = {0:'Basic',
                      1:'AutoEncoder',
                      2:'AutoEncoder_context',
                      3:'OneClass',
                      4:'OneClass_condition'}
    args.model_type = modelType_dict[args.model_type]
    
    
    """ Save path of result """
    model_path       = './model'
    figure_path      = './figure'
    model_name       = 'Bi' + args.model_type if args.no_bidirectional != True else args.model_type
    model_name       = f'{model_name}_{args.n_layers}_{args.epochs}'
    target_name      = "_".join(args.targets)
    if args.augment == True:
        args.save_path   = os.path.join(model_path, str(args.data_version), model_name,  f'{args.data_type}_shifted')
        save_figure_path = os.path.join(figure_path, str(args.data_version), model_name, f'{args.data_type}_shifted')
    else:
        args.save_path   = os.path.join(model_path, str(args.data_version), model_name,  f'{args.data_type}')
        save_figure_path = os.path.join(figure_path, str(args.data_version), model_name, f'{args.data_type}')
        
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(save_figure_path, exist_ok=True)
    args.save_path   = os.path.join(args.save_path, target_name)
    save_figure_path = os.path.join(save_figure_path, target_name)

    if args.inference == False:
        with tf.Graph().as_default():
            model = Model(args)

            print("Training...")
            for name, path_list in trainPath_dict.items():
                model.train(path_list)
                save_loss(model.loss_list,  args.save_path)
    else:
        with tf.Graph().as_default():
            model = Model(args)
            
            model.load_weights()
            print("model loaded")
            print("Test...")

            # Evaluate
            loss_dict = {}
            threashold = None

            y_true = []  # binary
            y_score = [] # l2 loss
            
            # Get threshold from sample data
            for name, path_list in testPath_dict.items():
                if  name == target_name:
                    loss_list = model.inference(path_list)
                    min_threshold = np.min(loss_list)
                    threshold = np.mean(loss_list) * 1.2

            # Figure with threshold
            max_threshold = 0
            for name, path_list in testPath_dict.items():
                loss_list = model.inference(path_list)
                max_threshold = max(max_threshold, np.max(loss_list))
                if  name == target_name:
                    if len(args.targets) == 1:
                        name = datasets_loader.datasets[name]['name'] # If not trained target, use full name. Not important
                else:
                    name =datasets_loader.datasets[name]['name']

                loss_dict[name] = loss_list
            compare_loss(loss_dict,  args.save_path, threshold, save_figure_path, args.loss_log_scale)
            
            roc = calc_roc(loss_dict, datasets_loader.datasets[target_name]['name'] , min_max=(min_threshold, max_threshold)) # evaluate with ROC
            np.save(f'{save_figure_path}_roc.npy', roc)
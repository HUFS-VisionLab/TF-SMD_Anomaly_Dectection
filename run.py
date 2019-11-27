import os
import glob
import json
import shutil
import argparse

import numpy as np
import tensorflow as tf
import torch

from utils.metric import get_score
from plot import save_loss, compare_loss 
from datasets_loader import DatasetsLoader

from model import Model

    
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--targets', nargs='*', default=None, help='-')
    parser.add_argument('--n_layers', type=int, default=3, help='-')
    parser.add_argument('--model_type', type=int, default=2, help='-')
    parser.add_argument('--no_bidirectional', action='store_true', default=False, help='-')
    parser.add_argument('--data_type', type=str, default='timesteps_64_mel_80', help='-')
    parser.add_argument('--augment', type=str, default='00', help='-')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='-')
    parser.add_argument('--beta_1', type=float, default=0.9, help='-')
    parser.add_argument('--beta_2', type=float, default=0.999, help='-')
    parser.add_argument('--epsilon', type=float, default=1e-08, help='-')
    parser.add_argument('--epochs', type=int, default=3000, help='-')
    parser.add_argument('--batch_size', type=int, default=None, help='-')
    parser.add_argument('--model_path', type=str, default='./model', help='-')
    parser.add_argument('--inference', action='store_true', default=False, help='-')
    parser.add_argument('--loss_log_scale', type=int, default=0, help='-')
    args, unknown = parser.parse_known_args()
    
    
    """ Preprare the path of dataset and load hyper parameter of model """
    datasets_loader = DatasetsLoader(targets=args.targets, data_type=args.data_type, is_augment=args.augment)
    data_version = datasets_loader.version
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
    
    
    augment_detail = {'0' : 'normal', '1' : 'shifted', '2' : 'both'}
    is_trainAug = augment_detail[args.augment[0]]
    is_testAug = augment_detail[args.augment[1]]
    
    
    """ Save path of result """
    model_path       = './model'
    figure_path      = './figure'
    model_name       = 'Bi' + args.model_type if args.no_bidirectional != True else args.model_type
    model_name       = f'{model_name}_{args.n_layers}_{args.epochs}'
    target_name      = "_".join(args.targets) # EX) args.targets = [Class0, Class1] --> target_name : 'Class0-name_Class1-name'
    
    args.save_path   = os.path.join(model_path, str(data_version), model_name,  f'{args.data_type}_train_{is_trainAug}')
    os.makedirs(args.save_path, exist_ok=True)
    args.save_path   = os.path.join(args.save_path, target_name)
    
    if args.inference == False:
        with tf.Graph().as_default():
            model = Model(args)

            print("Training...")
            for name, path_list in trainPath_dict.items():
                model.train(path_list)
                save_loss(model.loss_list,  args.save_path)
    else:
        with tf.Graph().as_default():
            save_figure_path = os.path.join(figure_path, str(data_version), model_name, f'{args.data_type}_train_{is_trainAug}_test_{is_testAug}')
            os.makedirs(save_figure_path, exist_ok=True)
            save_figure_path = os.path.join(save_figure_path, target_name)
            
            model = Model(args)
            
            model.load_weights()
            print("model loaded")
            print("Test...")

            # Evaluate
            loss_dict = {}

            max_loss = 0 
            for name, path_list in testPath_dict.items():
                loss_list = model.inference(path_list)

                if name == target_name:
                    avg_loss = np.mean(loss_list)
                    if len(args.targets) == 1:
                        name = datasets_loader.datasets[name]['name']
                        target_name = name
                else: 
                    # If not trained target, use full name. Not important
                    name =datasets_loader.datasets[name]['name']
                    
                loss_dict[name] = loss_list
                
            
            compare_loss(loss_dict,  args.save_path, avg_loss, save_figure_path, log_scaling=args.loss_log_scale)
            score_dict = get_score(loss_dict, target_name, args.timesteps, args.inputs_dims)
            with open(f"{save_figure_path}_score.json", 'w', encoding='utf-8') as f:
                json.dump(score_dict, f, ensure_ascii=False, indent=4)
            

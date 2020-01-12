import os
import glob
import json
import shutil
import argparse

import numpy as np
import tensorflow as tf

from utils.metric import get_score
from plot import save_loss, compare_loss, show_laten_space 
from datasets_loader import DatasetsLoader

from ssae import SSAE

    
if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    # Dataset setting
    parser.add_argument('-l', '--targets', nargs='*', default=None, help='-')
    parser.add_argument('--dataset-name', dest='dataset_name', type=str, default='SMD_dataset', help='-')
    parser.add_argument('--seq-len', dest='seq_len', type=int, default=32, help='-')
    parser.add_argument('--dims', type=int, default=80, help='-')
    # Network setting
    parser.add_argument('--n_layers', type=int, default=4, help='-')
    parser.add_argument('--epochs', type=int, default=5000, help='-')
    parser.add_argument('--no_bidirectional', dest='bidirectional', action='store_false', default=True, help='-')
    # Optimizer Setting
    parser.add_argument('--batch_size', type=int, default=None, help='-')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='-')
    parser.add_argument('--beta_1', type=float, default=0.9, help='-')
    parser.add_argument('--beta_2', type=float, default=0.999, help='-')
    parser.add_argument('--epsilon', type=float, default=1e-08, help='-')
    # Option
    parser.add_argument('--model_path', type=str, default='./model', help='-')
    parser.add_argument('--test', action='store_true', default=False, help='-')
    parser.add_argument('--threshold_weight', type=int, default=2.0, help='-')
    args, unknown = parser.parse_known_args()
    
    
    """ Preprare the path of dataset and load hyper parameter of model """
    datasets_loader = DatasetsLoader(args)
    dataset_detail = 'seqlen_{}_mels_{}'.format(args.seq_len, args.dims)
    
    trainPath_list = datasets_loader.pathList_dict['train']
    trainPath_list = sorted(trainPath_list, key=os.path.getsize)
    testPath_dict = datasets_loader.pathList_dict['test']
    testPath_dict = dict(sorted(testPath_dict.items()))
    
    print("Targets :", args.targets, "n_data :", len(trainPath_list))
    for name, path_list in testPath_dict.items():
        print("Test :", name, "n_data :", len(path_list))
    
    
    """ Save path of result """
    model_path       = './model'
    figure_path      = './figure'
    model_name       = 'SSAE_{}_{}'.format(args.n_layers, args.epochs)
    target_name      = "_".join(args.targets) # EX) args.targets = [Class0, Class1] --> target_name : 'Class0-name_Class1-name'
    
    args.save_path   = os.path.join(model_path, args.dataset_name, model_name,  dataset_detail)
    os.makedirs(args.save_path, exist_ok=True)
    args.save_path   = os.path.join(args.save_path, target_name)
    
    with tf.Graph().as_default():
        tf.set_random_seed(123456789)
        
        model = SSAE(args)
    
        if args.test == False:
            model.train(trainPath_list)
            save_loss(model.loss_list,  args.save_path)
        else:
            save_figure_path = os.path.join(figure_path, args.dataset_name, model_name, dataset_detail)
            os.makedirs(save_figure_path, exist_ok=True)
            save_figure_path = os.path.join(save_figure_path, target_name)

            model.load_weights()
            print("model loaded")
            print("Test...")

            # Evaluate
            loss_dict = {}
            laten_vector_dict = {}

            # calculate mean l2 loss of train data
            loss_list, _ = model.test(trainPath_list)
            avg_loss = np.mean(loss_list)

            for name, path_list in testPath_dict.items():
                loss_list, laten_vector_list = model.test(path_list)
                loss_dict[name] = loss_list
                laten_vector_dict[name] = laten_vector_list


            compare_loss(loss_dict,  args.save_path, avg_loss, save_figure_path, weight=args.threshold_weight)
            show_laten_space(laten_vector_dict, args.save_path, save_figure_path)

            score_dict = get_score(loss_dict, target_name, avg_loss, args.seq_len, args.dims,
                                   weight=args.threshold_weight)
            with open(f"{save_figure_path}_score.json", 'w', encoding='utf-8') as f:
                json.dump(score_dict, f, ensure_ascii=False, indent=4)

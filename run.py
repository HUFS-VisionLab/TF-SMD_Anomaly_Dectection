import os
import glob
import shutil
import argparse

import numpy as np
import tensorflow as tf

from plot import save_loss, compare_loss 
from model import Model


def makedir(path):
    try:
        os.mkdir(path)
        print("Make dir: %s" %(path))
    except:
        print("Already exist: %s" %(path))
        
    
def initialize(args):
    dataset = {
                "AT" : "AT2-IN88-SINK",
                "M1" : "M-3708",
                "M2" : "M-4478",
                "NA1" : "NA-9289-MAIN",
                "NA2" : "NA-9473",
                "ST" : "ST-4214-GE"
               }
    
    
    trainset_name = 'train_' + '_'.join(args.target_category)
    

    dataset_path = f'./data_{args.data_type}'
    experiment = os.path.join(dataset_path, 'experiment')
    experiment_tr = os.path.join(experiment, trainset_name)
    makedir(experiment)
    makedir(experiment_tr)

    count = 0
    for category in args.target_category:
        fileslist = glob.glob(os.path.join(dataset_path, 'train', dataset[category], '*.npy'))
        for file in fileslist:
            index = str(count).zfill(4)
            shutil.copy(file, os.path.join(experiment_tr, f'{index}.npy'))
            count += 1
            
    timesteps, input_dim = np.load(file).shape  # get hyper parameters from saved data
            
    testPath_dict = {}
    for category, category_name in dataset.items():
        src_path = 'train' if category not in args.target_category else 'test'
        
        src = os.path.join(dataset_path, src_path, category_name)
        dst = os.path.join(experiment, category_name)

        shutil.copytree(src, dst) if not os.path.exists(dst) else None
        
        testPath_dict[category_name] = src
        
    return experiment_tr, testPath_dict, timesteps, input_dim


if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', type=int, default=3, help='-')
    parser.add_argument('--is_bidirectional', type=int, default=1, help='-')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='-')
    parser.add_argument('--beta_1', type=float, default=0.9, help='-')
    parser.add_argument('--beta_2', type=float, default=0.999, help='-')
    parser.add_argument('--epsilon', type=float, default=1e-08, help='-')
    parser.add_argument('--epochs', type=int, default=1000, help='-')
    parser.add_argument('--batch_size', type=int, default=None, help='-')
    parser.add_argument('-l', '--target_category', nargs='*', default=["AT"], help='-')
    parser.add_argument('--data_type', type=str, default='mel', help='-')
    parser.add_argument('--model_path', type=str, default='./model', help='-')
    parser.add_argument('--inference', action='store_true', default=False, help='-')
    args, unknown = parser.parse_known_args()
    
    """ Save path of result """
    model_path  = './model'
    figure_path = './figure'
    layer_type    = 'LSTM' if args.is_bidirectional == 0 else 'BiLSTM'
    model_name  = f'{layer_type}_{args.n_layers}_{args.epochs}'
    target_name =  "_".join(args.target_category) + f"_{args.data_type}"
    args.save_path = os.path.join(model_path, model_name, target_name)
    save_figure_path = os.path.join(figure_path, model_name, target_name)
    makedir(model_path)
    makedir(figure_path)
    makedir(os.path.join(model_path, model_name))
    makedir(os.path.join(figure_path, model_name))
    makedir(args.save_path)
     
    
    """ Preprare the path of dataset and load hyper parameter of model """
    args.inputPath, args.testPath_dict, args.timesteps, args.input_dim = initialize(args)

    
    with tf.Graph().as_default():
        model = Model(args)

        if args.inference == False:
            model.train()
            save_loss(model.loss_list,  args.save_path)
            
            print("Is inferring...")
            loss_dict = {}
            for key, testPath in args.testPath_dict.items():
                loss_list = model.inference(testPath)
                
                loss_dict[key] = loss_list
                
            compare_loss(loss_dict,  args.save_path, save_figure_path)
            print("Inference completed")
            
        else:
            print("Is inferring...")
            model.load_weights()
            print("model loaded")
            
            loss_dict = {}
            for key, testPath in args.testPath_dict.items():
                loss_list = model.inference(testPath)
                
                loss_dict[key] = loss_list
                
            compare_loss(loss_dict,  args.save_path, save_figure_path)
            print("Inference completed")
            
    shutil.rmtree(f'./data_{args.data_type}/experiment', ignore_errors=True)

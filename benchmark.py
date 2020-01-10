import os
import glob
import json
import argparse
import numpy as np

from plot import score_table, custom_table


def class_table(args):
    targetDir_list = glob.glob(os.path.join('./figure', args.dataset_name, f'*_{args.n_layers}_{args.epochs}'))
    for target_dir in targetDir_list:
        model_name = os.path.basename(target_dir)
        save_figure_path = os.path.join('./benchmark', args.data_version, model_name)
        os.makedirs(save_figure_path, exist_ok=True)
        
        data_type = "seqlen_{}_mels_{}".format(args.seq_len, args.dims)
        
        target_path = os.path.join(target_dir, data_type)
        save_figure_path = os.path.join(save_figure_path, data_type)
        
        if os.path.exists(target_path):
            scoreDict_list = []
            jsonPath_list = sorted(glob.glob(os.path.join(target_path, '*_score.json')))
            for json_path in jsonPath_list:
                with open(json_path, 'r') as f:
                    score_dict = json.load(f)
                    scoreDict_list.append(score_dict)

            score_table(scoreDict_list, save_figure_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-layers', dest='n_layers', type=int, default=4, help='-')
    parser.add_argument('--epochs', type=int, default=5000, help='-')
    parser.add_argument('--dataset-name', dest='dataset_name', type=str, default='SMD_dataset', help='-')
    parser.add_argument('--seq-len', dest='seq_len', type=int, default=32, help='-')
    parser.add_argument('--dims', type=int, default=80, help='-')
    args, unknown = parser.parse_known_args()
    
    class_table(args)
        
    scoreDict_list = None
    
    save_path = os.path.join('./benchmark')
    save_name = 'method-table'
        
        
    if scoreDict_list != None:
        custom_table(scoreDict_list, save_path, save_name)
        
     

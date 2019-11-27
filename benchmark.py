import os
import glob
import json
import argparse
import numpy as np

from plot import score_table

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', type=int, default=3, help='-')
    parser.add_argument('--epochs', type=int, default=3000, help='-')
    parser.add_argument('--data_version', type=str, default='2019-1', help='-')
    parser.add_argument('--data_type', type=str, default='timesteps_64_mel_80', help='-')
    parser.add_argument('--augment', nargs='*', default=['0', '0'], help='-')
    args, unknown = parser.parse_known_args()
    
    data_type = args.data_type.split('_')
    args.timesteps = int(data_type[1])
    args.inputs_dims = int(data_type[-1])
    
    augment_detail = {'0' : 'normal', '1' : 'shifted', '2' : 'both'}
    is_trainAug = augment_detail[args.augment[0]]
    is_testAug = augment_detail[args.augment[1]]
    
    targetDir_list = glob.glob(os.path.join('./figure', args.data_version, f'*_{args.n_layers}_{args.epochs}'))
    for target_dir in targetDir_list:
        model_name = os.path.basename(target_dir)
        save_figure_path = os.path.join('./benchmark', args.data_version, model_name)
        os.makedirs(save_figure_path, exist_ok=True)
        
        target_path = os.path.join(target_dir, f'{args.data_type}_train_{is_trainAug}_test_{is_testAug}')
        save_figure_path = os.path.join(save_figure_path, f'{args.data_type}_train_{is_trainAug}_test_{is_testAug}')
        
        scoreDict_list = []
        jsonPath_list = glob.glob(os.path.join(target_path, '*_score.json'))
        for json_path in jsonPath_list:
            with open(json_path, 'r') as f:
                score_dict = json.load(f)
                scoreDict_list.append(score_dict)
                
        score_table(scoreDict_list, save_figure_path)

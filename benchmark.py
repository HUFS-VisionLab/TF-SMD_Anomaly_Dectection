import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def show_roc_curve(fpr, tpr, roc_auc, target_list, title, save_path, is_average=True):
    # Plot all ROC curves
    lw = 2
    plt.figure()
    if is_average:
        plt.plot(fpr["macro"], tpr["macro"], 
                 label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

    for target in target_list:
        plt.plot(fpr[target], tpr[target], lw=lw,
                 label='ROC curve of {0} (area = {1:0.2f})'.format(target, roc_auc[target]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title}')
    plt.legend(loc="lower right")
    plt.savefig(f"{save_path}/{title}.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', type=int, default=3, help='-')
    parser.add_argument('--epochs', type=int, default=3000, help='-')
    parser.add_argument('--data_version', type=int, default=2019, help='-')
    parser.add_argument('--data_type', type=str, default='timesteps_64_mel_80', help='-')
    args, unknown = parser.parse_known_args()
    
    
    save_path = os.path.join('./benchmark', args.data_type, f'{args.n_layers}_{args.epochs}')
    os.makedirs(save_path, exist_ok=True)
    
    targetDir_list = glob.glob(os.path.join('./figure', str(args.data_version), f'*_{args.n_layers}_{args.epochs}'))
    
    model_fpr = {}
    model_tpr = {}
    model_roc_auc = {}
    for dir_path in targetDir_list:
        model_name = os.path.basename(dir_path)
        target_path = os.path.join(dir_path, args.data_type)

        path_list = glob.glob(os.path.join(target_path, '*_roc.npy'))
        
        fpr = {}
        tpr = {}
        total_fpr = 0
        total_tpr = 0
        roc_auc = {}
        for path in path_list:
            target_name = os.path.basename(path).split('_')[0]
            fpr[target_name], tpr[target_name] = np.load(path)
            total_fpr += fpr[target_name] / len(path_list)
            total_tpr += tpr[target_name] / len(path_list)
            roc_auc[target_name] = auc(fpr[target_name], tpr[target_name])
            
            
        fpr['macro'] = model_fpr[model_name] = total_fpr
        tpr['macro'] = model_tpr[model_name] = total_tpr
        roc_auc['macro'] = model_roc_auc[model_name] = auc(fpr['macro'], tpr['macro'])
        
        # Plot all ROC curves
        target_list = [os.path.basename(path).split('_')[0] for path in path_list]
        show_roc_curve(fpr, tpr, roc_auc, target_list=target_list, title=model_name, save_path=save_path)
    """
    target_list = [os.path.basename(dir_path) for path in targetDir_list]
    show_roc_curve(model_fpr, model_tpr, model_roc_auc, target_list=target_list, title='Networks', save_path=save_path, is_average=False)
    """
    
    
            

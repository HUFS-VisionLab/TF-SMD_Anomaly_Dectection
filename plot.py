import os
import matplotlib.pyplot as plt
import numpy as np

def save_loss(loss_list, model_path):
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(loss_list, color='blue', linestyle="-", label="loss")
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig(f"{model_path}/loss.png")
    plt.close()
    
    
def compare_loss(loss_dict, model_path, value, save_figure_path, weight=1.2, log_scaling=0):
    plt.clf()
    plt.rcParams['font.size'] = 15
    
    xlabel = "index of data"
    ylabel = "loss -- y=log10(x+10)" if log_scaling==1 else "loss"
    
    for key, loss_list in loss_dict.items():
        if log_scaling == 1:
            loss_list = np.log10(np.array(loss_list) + 10)
        
        plt.scatter(range(len(loss_list)), loss_list, label=key, s=10)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if value != None:
        threshold = value * weight
        plt.axhline(y=threshold, color='r', linestyle='-')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size' : 12})
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    
    plt.savefig(f"{model_path}/compare_loss.png")
    plt.savefig(f"{save_figure_path}_compare_loss.png")
    
    plt.ylim(0, value * 5)
    plt.savefig(f"{model_path}/compare_loss_zoom.png")
    plt.savefig(f"{save_figure_path}_compare_loss_zoom.png")

    plt.close()
    
    
def score_table(scoreDict_list, save_path):
    avg_TPR = 0
    avg_FPR = 0
    avg_F1_score = 0
    
    n_class = len(scoreDict_list)
    for score_dict in scoreDict_list:
        avg_TPR += score_dict['TPR'] / n_class
        avg_FPR += score_dict['FPR'] / n_class
        avg_F1_score += score_dict['F1_SCORE'] / n_class
        
    scoreDict_list.append(
        {
            'name' : 'average',
            'weight' : score_dict['weight'],
            'seq_length' : score_dict['seq_length'],
            'dims' : score_dict['dims'],
            'TPR' : round(avg_TPR, 2),
            'FPR' : round(avg_FPR, 2),
            'F1_SCORE' : round(avg_F1_score, 2)
        }
    )
    
    
    col_labels = ['TPR', 'FPR', 'F1-score']
    row_labels = [score_dict['name'] for score_dict in scoreDict_list]
    table_vals = [[score_dict['TPR'], score_dict['FPR'], score_dict['F1_SCORE']] for score_dict in scoreDict_list]
    the_table = plt.table(cellText=table_vals,
                          colWidths=[0.1] * 3,
                          rowLabels=row_labels,
                          colLabels=col_labels,
                          loc='center')

    the_table.auto_set_font_size(False)
    the_table.scale(2, 2)
    the_table.set_fontsize(14)
    plt.title('Score')
    
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)
    plt.savefig(f"{save_path}.png")

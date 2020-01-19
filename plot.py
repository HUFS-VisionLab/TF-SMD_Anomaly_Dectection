import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE


def save_loss(loss_list, model_path):
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(loss_list, color='blue', linestyle="-", label="loss")
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig(f"{model_path}/loss.png")
    plt.close()
    
    
def compare_loss(loss_dict, model_path, value, save_figure_path, weight):
    plt.clf()
    plt.rcParams['font.size'] = 15
    
    xlabel = "index of data"
    ylabel = "loss"
    
    if value != None:
        threshold = value * weight
        plt.axhline(y=threshold, color='r', linestyle='-', label='Threshold')
    
    for name, loss_list in loss_dict.items():
        plt.scatter(range(len(loss_list)), loss_list, label=name, s=10)
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size' : 12})
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    
    plt.savefig(f"{model_path}/compare_loss.png")
    plt.savefig(f"{save_figure_path}_compare_loss.png")
    
    plt.ylim(0, threshold * 2)
    plt.savefig(f"{model_path}/compare_loss_zoom.png")
    plt.savefig(f"{save_figure_path}_compare_loss_zoom.png")

    plt.close()

    
def show_laten_space(context_vectors_dict, model_path, save_figure_path):
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.xlabel("Dim_1")
    plt.ylabel("Dim_2")
    
    name_list = list(context_vectors_dict.keys())
    context_vectors_list = list(context_vectors_dict.values())
    
    del context_vectors_dict
    
    total_context_vectors = np.concatenate(context_vectors_list, axis=0) # Shape=(n_total_data, context_dims)

    x = TSNE(n_components=2).fit_transform(total_context_vectors)
    y = []
    for i, context_vectors in enumerate(context_vectors_list):
        y += [i for j in range(len(context_vectors))]
    y = np.array(y)
    
    del context_vectors_list
    
    #colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, label in zip(range(len(name_list)), name_list):
        plt.scatter(x[y == i, 0], x[y == i, 1], label=label, s=10)
    #plt.legend()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size' : 12})
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    
    plt.savefig(f"{model_path}/2d_space.png")
    plt.savefig(f"{save_figure_path}_2d_space.png")
    
    
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
            'name' : 'Average',
            'weight' : score_dict['weight'],
            'seq_length' : score_dict['seq_length'],
            'dims' : score_dict['dims'],
            'TPR' : round(avg_TPR, 3),
            'FPR' : round(avg_FPR, 3),
            'F1_SCORE' : round(avg_F1_score, 3)
        }
    )
    
    
    col_labels = ['Class','TPR', 'FPR', 'F1-score']
    #row_labels = [score_dict['name'] for score_dict in scoreDict_list]
    table_vals = [[score_dict['name'], score_dict['TPR'], score_dict['FPR'], score_dict['F1_SCORE']] \
                  for score_dict in scoreDict_list]
    the_table = plt.table(cellText=table_vals,
                          colWidths=[0.15] * 4,
                          #rowLabels=row_labels,
                          colLabels=col_labels,
                          loc='center')

    the_table.auto_set_font_size(False)
    the_table.scale(2, 2)
    the_table.set_fontsize(14)
    
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)
    plt.savefig(f"{save_path}.png")
    
    
def custom_table(scoreDict_list, save_path, save_name):
    plt.rcParams["figure.figsize"] = (12,12)
    col_labels = list(scoreDict_list[0].keys())
    table_vals = [list(score_dict.values()) for score_dict in scoreDict_list]
    
    the_table = plt.table(cellText=table_vals,
                          colWidths=[0.15] * len(col_labels),
                          #rowLabels=row_labels,
                          colLabels=col_labels,
                          loc='center')

    the_table.auto_set_font_size(False)
    the_table.scale(1, 3)
    the_table.set_fontsize(14)
    
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)
    plt.savefig(os.path.join(save_path, f'{save_name}.png'))
    
    
    

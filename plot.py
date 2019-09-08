import os
import matplotlib.pyplot as plt


def save_loss(loss_list, model_path):
    plt.clf()
    plt.rcParams['font.size'] = 15
    plt.plot(loss_list, color='blue', linestyle="-", label="loss")
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig(f"{model_path}/loss.png")
    plt.close()
    
    
def compare_loss(loss_dict, model_path ,save_figure_path):
    plt.clf()
    plt.rcParams['font.size'] = 15
    
    for key, loss_list in loss_dict.items():
        plt.scatter(range(len(loss_list)), loss_list, label=key, s=10)
        
    plt.xlabel("the number of data")
    plt.ylabel("loss")
    plt.ylim(0, 1)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size' : 12})
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    plt.savefig(f"{model_path}/compare_loss.png")
    plt.savefig(f"{save_figure_path}_compare_loss.png")
    plt.close()
